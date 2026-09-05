"""
The general training framework adapted for the mammography dataset.
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import pandas as pd
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.mammo_bac import get_mammogram_dataloaders, get_mammogram_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
from helper.util import reproducibility

from helper.misc import save_checkpoint, load_checkpoint


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('Argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='number of workers to use')
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')  
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='mammo', choices=['mammo'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'convnext_tiny', 
                                 'convnext_small', 'convnext_base', 'convnext_large'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--resume_s', type=str, default=None, help='student model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
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
    parser.add_argument('--nce_m', default=0.5, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

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
        # model_name = segments[0]
        model_name = segments[0] + '_' + segments[1]
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

    # for k, v in model.state_dict().items():
    #     print(k)
    # print("--------------------------------------------------------------------------------")
    # for k, v in state_dict.items():
    #     print(k)

    # Load the state dictionary into the model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # missing, unexpected = model.load_state_dict(state_dict['model'], strict=False)

    print(f'[initialize_weight] missing_keys={missing}')
    print(f'[initialize_weight] unexpected_keys={unexpected}')
    print('==> done')
    return model

def get_class_distribution(labels):
    labels = labels.astype(np.int64)  # Convert labels to integers
    return np.bincount(labels)

def get_class_weights(labels):
    class_count = get_class_distribution(labels)
    class_freq = class_count / np.sum(class_count)
    class_weights = 1.0 / class_freq
    norm_weights = class_weights / np.sum(class_weights)
    return norm_weights



def main():

    reproducibility(seed=42)
    
    best_acc = 0

    best_auc_roc = 0

    opt = parse_option()

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'mammo':
        if opt.distill in ['crd']:
            train_loader, val_loader, test_loader, n_data = get_mammogram_dataloaders_sample(batch_size=opt.batch_size,
                                                                                num_workers=opt.num_workers,
                                                                                k=opt.nce_k,
                                                                                mode=opt.mode)
        else:
            train_loader, val_loader, test_loader = get_mammogram_dataloaders(batch_size=opt.batch_size,
                                                                num_workers=opt.num_workers)
        
        n_cls = 1  # Binary classification

        # # Get the labels from the training data
        # train_labels = np.concatenate([labels.numpy() for _, labels in train_loader], axis=0)
        # train_labels = np.array(train_labels).astype(np.float32)
        
        # # Compute class weights
        # norm_weights = get_class_weights(train_labels)
        # # class_weights_tensor = torch.tensor(norm_weights, dtype=torch.float32)

        # Get the labels from the training data
        train_labels = []
        for batch in train_loader:
            # Assuming the dataloader returns data, labels, and possibly other info
            if len(batch) == 2:
                data, labels = batch
            elif len(batch) > 2:
                data, labels, *others = batch  # Unpack only the first two elements
            # print("Extracted labels:", labels.numpy())  # Print statement to confirm extraction
            train_labels.append(labels.numpy())
        train_labels = np.concatenate(train_labels, axis=0)
        train_labels = np.array(train_labels).astype(np.float32)
        
        # Compute class weights
        norm_weights = get_class_weights(train_labels)
        class_weights_tensor = torch.tensor(norm_weights, dtype=torch.float32)

    else:
        raise NotImplementedError(opt.dataset)

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
    wt = class_weights_tensor[1]/class_weights_tensor[0]
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=wt)  # Use BCEWithLogitsLoss balanced
    # criterion_cls = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss 
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        trainable_list.append(criterion_kd)  # add this as some parameters in VIDLoss need to be updated
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.Adam(trainable_list.parameters(), lr=opt.learning_rate)  # Use Adam optimizer

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    # optimizer = optim.SGD(trainable_list.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)


    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True
    
    # Load checkpoint if resume is specified
    if opt.resume_s:
        start_epoch, _ = load_checkpoint(opt.resume_s, model_s, optimizer)
    else:
        start_epoch = 1
      

    # validate teacher accuracy
    teacher_acc, teacher_loss, teacher_auc_roc, teacher_auc_pr = validate(test_loader, model_t, criterion_cls, opt)
    print('teacher test accuracy: ', teacher_acc)
    print('teacher test loss: ', teacher_loss)
    print('teacher test auc roc: ', teacher_auc_roc)
    print('teacher test auc pr: ', teacher_auc_pr)
    


    # routine
    for epoch in range(start_epoch, opt.epochs + 1):
        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        # Stage 1: Contrastive Learning with Pseudo-labeled Data
        train_acc, train_loss, auc_roc, auc_pr = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('train_auc_roc ', auc_roc, epoch)
        logger.log_value('train_auc_pr', auc_pr, epoch)
    
        # Validation
        test_acc, test_loss, test_auc_roc, test_auc_pr = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_auc_roc', test_auc_roc, epoch)
        logger.log_value('test_auc_pr', test_auc_pr, epoch)

        # save the best model
        if test_auc_roc > best_auc_roc:
            best_auc_roc = test_auc_roc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_auc_roc': best_auc_roc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # # regular saving
        # if epoch % opt.save_freq == 0:
        #     print('==> Saving...')
        #     state = {
        #         'epoch': epoch,
        #         'model': model_s.state_dict(),
        #         'auc_roc': test_auc_roc,
        #     }
        #     save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
        #     torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            save_checkpoint(f'ckpt_epoch_{epoch}.pth', opt, epoch, f'{test_auc_roc:.4f}', model_s.state_dict(), optimizer.state_dict())


        scheduler.step()

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # # save model
    # state = {
    #     'opt': opt,
    #     'model': model_s.state_dict(),
    # }
    # save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    # torch.save(state, save_file)

    # save the last model
    save_checkpoint(f'{opt.model_s}_last.pth', opt, epoch, f'{test_auc_roc:.4f}', model_s.state_dict(), optimizer.state_dict())



if __name__ == '__main__':
    main()