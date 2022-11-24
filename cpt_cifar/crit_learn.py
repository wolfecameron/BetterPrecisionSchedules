from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import wandb

import os
import shutil
import argparse
import time
import logging

import models
from modules.data import *
from fvcore.nn import FlopCountAnalysis

import util_swa

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR training with CPT')
    parser.add_argument('--exp-name', type=str, default='cl_cnn_00')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--arch', metavar='ARCH', default='cifar10_resnet_38',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_38)')
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--def-iters', default=8000, type=int)
    parser.add_argument('--iters', default=64000, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm_up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval_every', default=390, type=int,
                        help='evaluate model every (default: 1000) iterations')

    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    #parser.add_argument('--num_bits_schedule', default=None, type=int, nargs='*',
    #                    help='schedule for weight/act precision')
    #parser.add_argument('--num_grad_bits_schedule', default=None, type=int, nargs='*',
    #                    help='schedule for grad precision')

    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')

    parser.add_argument('--swa_start', type=float, default=None, help='SWA start step number')
    parser.add_argument('--swa_freq', type=float, default=1170,
                        help='SWA model collection frequency')
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--tags', type=str, action='append', default=None)
    args = parser.parse_args()
    
    if args.use_wandb:
        wandb_run = wandb.init(
                project='cnn-quant',
                entity='cameron-research',
                name=args.exp_name,
                tags=args.tags
        )
        wandb_run.define_metric(
                name=f'Training Loss',
                step_metric='Iteration',
        )
        wandb_run.define_metric(
                name=f'Training Accuracy',
                step_metric='Iteration',
        )
        wandb_run.define_metric(
                name=f'Test Accuracy',
                step_metric='Iteration',
        )
        wandb_run.define_metric(
                name=f'Bits',
                step_metric='Iteration',
        )
        wandb_run.define_metric(
                name=f'Grad Bits',
                step_metric='Iteration',
        )
        wandb_run.define_metric(
                name=f'Learning Rate',
                step_metric='Iteration',
        )
        wandb.config = args.__dict__

    return args


def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {}'.format(
            args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.__dict__[args.arch](args.pretrained)
    model = model.to(device)

    if args.swa_start is not None:
        print('SWA training')
        swa_model = torch.nn.DataParallel(models.__dict__[args.arch](args.pretrained)).cuda()
        swa_n = 0

    else:
        print('SGD training')

    best_prec1 = 0
    best_iter = 0

    best_swa_prec = 0
    best_swa_iter = 0

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            if args.swa_start is not None:
                swa_state_dict = checkpoint['swa_state_dict']
                if swa_state_dict is not None:
                    swa_model.load_state_dict(swa_state_dict)
                swa_n_ckpt = checkpoint['swa_n']
                if swa_n_ckpt is not None:
                    swa_n = swa_n_ckpt
                best_swa_prec_ckpt = checkpoint['best_swa_prec']
                if best_swa_prec_ckpt is not None:
                    best_swa_prec = best_swa_prec_ckpt

            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    if args.swa_start is not None:
        swa_loader = prepare_train_data(dataset=args.dataset,
                                        datadir=args.datadir,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cr = AverageMeter()

    end = time.time()

    i = args.start_iter
    while i < args.iters:
        for input, target in train_loader:
            # measuring data loading time
            data_time.update(time.time() - end)

            model.train()
            adjust_learning_rate(args, optimizer, i)

            adjust_precision(args, i)

            i += 1

            # use FLOPS * (bit / 32)^2 to compute effective FLOPS (proportional to # bit-opts)
            # here, we seem to exclude FLOPS because it is constant
            # calculate both backward + forward pass cost, backward is 2x cost of forward (included twice in sum)
            fw_cost = args.num_bits * args.num_bits / 32 / 32
            eb_cost = args.num_bits * args.num_grad_bits / 32 / 32
            gc_cost = eb_cost
            cr.update((fw_cost + eb_cost + gc_cost) / 3)
            target = target.squeeze().long().to(device)
            input_var = Variable(input).to(device)
            target_var = Variable(target).to(device)

            # compute output
            output = model(input_var, args.num_bits, args.num_grad_bits)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, = accuracy(output.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print log
            if i % args.print_freq == 0:
                logging.info("Num bit {}\t"
                             "Num grad bit {}\t".format(args.num_bits, args.num_grad_bits))
                logging.info("Iter: [{0}/{1}]\t"
                             "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                             "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                             "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                             "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                             "Training FLOPS ratio: {cr.val:.6f} ({cr.avg:.6f})\t".format(
                    i,
                    args.iters,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    cr=cr
                )
                )


            if args.swa_start is not None and i >= args.swa_start and i % args.swa_freq == 0:
                util_swa.moving_average(swa_model, model, 1.0 / (swa_n + 1))
                swa_n += 1
                util_swa.bn_update(swa_loader, swa_model, args.num_bits, args.num_grad_bits)
                prec1 = validate(args, test_loader, swa_model, criterion, i, swa=True)

                if prec1 > best_swa_prec:
                    best_swa_prec = prec1
                    best_swa_iter = i

                print("Current Best SWA Prec@1: ", best_swa_prec)
                print("Current Best SWA Iteration: ", best_swa_iter)

            if (i % args.eval_every == 0 and i > 0) or (i == args.iters):
                with torch.no_grad():
                    prec1 = validate(args, test_loader, model, criterion, i)
                
                # update wandb metrics
                if args.use_wandb:
                    wandb.log({
                        'Training Loss': losses.avg,
                        'Training Accuracy': top1.avg,
                        'Test Accuracy': prec1,
                        'Bits': args.num_bits,
                        'Grad Bits': args.num_grad_bits,
                        'Iteration': i, 
                    })
                
                is_best = prec1 > best_prec1
                if is_best:
                    best_prec1 = prec1
                    best_iter = i
                
                    
                    #checkpoint_path = os.path.join(args.save_path, 'results.pth')
                    save_checkpoint = {
                        'iter': i,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'train_mets': (losses.avg, top1.avg, cr.avg, cr.val)
                        #'swa_state_dict': swa_model.state_dict() if args.swa_start is not None else None,
                        #'swa_n': swa_n if args.swa_start is not None else None,
                        #'best_swa_prec': best_swa_prec if args.swa_start is not None else None,
                    }
                    #shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                    #                                              'checkpoint_latest'
                    #                                              '.pth.tar'))
                    torch.save(save_checkpoint, os.path.join(args.save_path, 'best_results.pth'))
                
                print("Current Best Prec@1: ", best_prec1)
                print("Current Best Iteration: ", best_iter)
                print("Current cr val: {}, cr avg: {}".format(cr.val, cr.avg))

                if i == args.iters:
                    save_checkpoint = {
                        'iter': i,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'train_mets': (losses.avg, top1.avg, cr.avg, cr.val)
                    }
                    torch.save(save_checkpoint, os.path.join(args.save_path, 'final_results.pth'))
                    break


def validate(args, test_loader, model, criterion, step, swa=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().to(device)
        input_var = Variable(input, volatile=True).to(device)
        target_var = Variable(target, volatile=True).to(device)

        # compute output
        output = model(input_var, args.num_bits, args.num_grad_bits)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=losses, top1=top1
                )
            )

    if not swa:
        logging.info('Step {} * Prec@1 {top1.avg:.3f}'.format(step, top1=top1))
    else:
        logging.info('Step {} * SWA Prec@1 {top1.avg:.3f}'.format(step, top1=top1))

    return top1.avg



def test_model(args):
    # create model
    model = models.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        prec1 = validate(args, test_loader, model, criterion, args.start_iter)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def  update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_precision(args, _iter):
    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]
    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

    if _iter <= args.def_iters:
        args.num_bits = num_bit_min
    else:
        args.num_bits = num_bit_max
    args.num_grad_bits = num_grad_bit_max
    
    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(_iter, args.num_bits,
                                                                                                  args.num_grad_bits))


def adjust_learning_rate(args, optimizer, _iter):
    if args.lr_schedule == 'piecewise':
        first_dec = int(0.5 * args.iters)
        sec_dec = int(0.75 * args.iters)
        if first_dec <= _iter < sec_dec:
            lr = args.lr * (args.step_ratio ** 1)
        elif _iter >= sec_dec:
            lr = args.lr * (args.step_ratio ** 2)
        else:
            lr = args.lr

    elif args.lr_schedule == 'piecewise-no-def-decay':
        non_def_iters = int(args.iters - args.def_iters)
        _nd_iter = int(_iter - args.def_iters)
        first_dec = int(0.5 * non_def_iters)
        sec_dec = int(0.75 * non_def_iters)
        if first_dec <= _nd_iter < sec_dec:
            lr = args.lr * (args.step_ratio ** 1)
        elif _nd_iter >= sec_dec:
            lr = args.lr * (args.step_ratio ** 2)
        else:
            lr = args.lr

    else:
        raise NotImplementedError(f'{args.lr_schedule} is not a supported lr schedule.')

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    if args.use_wandb:
        wandb.log({
            'Learning Rate': lr,
            'Iteration': _iter, 
        })
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
