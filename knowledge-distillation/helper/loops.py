from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
from sklearn.metrics import roc_auc_score, average_precision_score


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_targets = []
    all_outputs = []

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target.unsqueeze(1).float())

        acc1 = accuracy(output, target.unsqueeze(1).float())
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))

        all_targets.extend(target.detach().cpu().numpy().tolist())
        all_outputs.extend(output.sigmoid().detach().cpu().numpy().tolist())

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    # Calculate AUC-ROC and AUC-PR
    auc_roc = roc_auc_score(all_targets, all_outputs)
    auc_pr = average_precision_score(all_targets, all_outputs)

    print(' * Acc@1 {top1.avg:.3f} AUC-ROC {auc_roc:.3f} AUC-PR {auc_pr:.3f}'.format(top1=top1, auc_roc=auc_roc, auc_pr=auc_pr))

    return top1.avg, losses.avg, auc_roc, auc_pr


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    # Wrap the models with DataParallel
    model_s = torch.nn.DataParallel(model_s)
    model_t = torch.nn.DataParallel(model_t)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_targets = []
    all_outputs = []

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target.unsqueeze(1).float())
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1 = accuracy(logit_s, target.unsqueeze(1).float())
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))

        all_targets.extend(target.detach().cpu().numpy().tolist())
        all_outputs.extend(logit_s.sigmoid().detach().cpu().numpy().tolist())

        # print(f'Batch {idx} - Targets: {target.tolist()}, Predictions: {logit_s.sigmoid().tolist()}')


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    # Calculate AUC-ROC and AUC-PR
    auc_roc = roc_auc_score(all_targets, all_outputs)
    auc_pr = average_precision_score(all_targets, all_outputs)

    print(' * Acc@1 {top1.avg:.3f} AUC-ROC {auc_roc:.3f} AUC-PR {auc_pr:.3f}'.format(top1=top1, auc_roc=auc_roc, auc_pr=auc_pr))

    return top1.avg, losses.avg, auc_roc, auc_pr


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    all_targets = []
    all_outputs = []

    # Wrap the model with DataParallel
    model = torch.nn.DataParallel(model)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.unsqueeze(1).float())

            # measure accuracy and record loss
            acc1 = accuracy(output, target.unsqueeze(1).float())
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].item(), input.size(0))

            all_targets.extend(target.detach().cpu().numpy().tolist())
            all_outputs.extend(output.sigmoid().detach().cpu().numpy().tolist())
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))
                
        # Calculate AUC-ROC and AUC-PR
        auc_roc = roc_auc_score(all_targets, all_outputs)
        auc_pr = average_precision_score(all_targets, all_outputs)

        print(' * Acc@1 {top1.avg:.3f} AUC-ROC {auc_roc:.3f} AUC-PR {auc_pr:.3f}'.format(top1=top1, auc_roc=auc_roc, auc_pr=auc_pr))


    return top1.avg, losses.avg, auc_roc, auc_pr
