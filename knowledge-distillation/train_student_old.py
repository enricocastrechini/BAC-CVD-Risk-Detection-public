from __future__ import print_function

import os
import socket
import argparse
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset.mammo_bac import get_mammo_dataloaders, get_mammo_dataloaders_sample
from helper.util import adjust_learning_rate
from helper.loops import train_distill as train, validate
from distiller_zoo import DistillKL
from crd.criterion import CRDLoss

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    
    # dataset
    parser.add_argument('--dataset', type=str, default='mammo', choices=['mammo'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'convnexttiny', 
                                 'convnextsmall', 'convnextbase', 'convnextlarge'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='crd', choices=['kd', 'crd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', type=float, default=0.5, help='momentum for non-parametric updates')

    opt = parser.parse_args()

    # set the path according to the environment
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/your/model_save_path'
        opt.tb_path = '/path/to/your/tensorboard_log'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if "convnext" in segments[0]:
        # Assuming the model names are of the form 'convnexttiny', 'convnextsmall', etc.
        model_name = segments[0]
    elif segments[0] != 'wrn':
        model_name = segments[0]
    else:
        model_name = segments[0] + '_' + segments[1] + '_' + segments[2]
    return model_name


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    state_dict = torch.load(model_path)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} 
    # Load the state dictionary into the model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'[initialize_weight] missing_keys={missing}')
    print(f'[initialize_weight] unexpected_keys={unexpected}')
    print('==> done')
    return model

def main():
    best_acc = 0
    best_auc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.distill in ['crd']:
        train_loader_unlabeled, val_loader, n_data = get_mammo_dataloaders_sample(batch_size=opt.batch_size,
                                                                                  num_workers=opt.num_workers,
                                                                                  k=opt.nce_k,
                                                                                  mode=opt.mode)
        train_loader_labeled = None  # Add this line
    else:
        train_loader_labeled, val_loader, train_loader_unlabeled = get_mammo_dataloaders(batch_size=opt.batch_size,
                                                                                         num_workers=opt.num_workers,
                                                                                         is_instance=True)
        n_data = None  # Add this line

    n_cls = 1  # For binary classification, we use a single output neuron

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(1, 3, 1120, 576)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.BCEWithLogitsLoss()  # Binary classification loss
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    # Add other distillation methods if needed

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.Adam(trainable_list.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, teacher_loss, teacher_auc, teacher_aupr = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)
    print('teacher loss: ', teacher_loss)
    print('teacher auc: ', teacher_auc)
    print('teacher aupr: ', teacher_aupr)

    # routine
    for epoch in range(1, opt.epochs + 1):
        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if opt.distill in ['crd']:
            train_acc, train_loss, train_auc_roc, train_aupr = train(epoch, None, train_loader_unlabeled, module_list, criterion_list, optimizer, opt)
        else:
            train_acc, train_loss, train_auc_roc, train_aupr = train(epoch, train_loader_labeled, train_loader_unlabeled, module_list, criterion_list, optimizer, opt)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('train_auc_roc', train_auc_roc, epoch)
        logger.log_value('train_aupr', train_aupr, epoch)

        test_acc, test_loss, test_auc_roc, test_aupr = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_auc_roc', test_auc_roc, epoch)
        logger.log_value('test_aupr', test_aupr, epoch)

        # save the best model
        if test_auc_roc > best_auc:
            best_auc = test_auc_roc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_auc': best_auc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'auc_roc': test_auc_roc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        
        # update scheduler
        scheduler.step()

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best auc:', best_auc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
