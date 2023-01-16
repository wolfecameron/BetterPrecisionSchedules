from __future__ import print_function

import torch
import wandb
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np

import os
import shutil
import argparse
import time
import logging
import math

import models
from modules.data import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__')
                     and callable(models.__dict__[name])
                     )


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch ImageNet training with CPT')
    parser.add_argument('--exp-name', type=str, default='cl_cnn_00')
    parser.add_argument('--dir', help='annotate the working directory')
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: cifar10_resnet_38)')
    parser.add_argument('--dataset', '-d', type=str, default='imagenet',
                        choices=['cifar10', 'cifar100','imagenet'],
                        help='dataset choice')
    parser.add_argument('--datadir', default='/home/yf22/dataset', type=str,
                        help='path to dataset')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=120, type=int,
                        help='number of epochs (default: 90)')
    parser.add_argument('--def-epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr_schedule', default='piecewise', type=str,
                        help='learning rate schedule')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--save_folder', default='save_checkpoints',
                        type=str,
                        help='folder to save the checkpoints')

    parser.add_argument('--num_bits',default=0,type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits',default=0,type=int,
                        help='num bits for gradient')

    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--tags', type=str, action='append', default=None)
    args = parser.parse_args()

    if args.use_wandb:
        wandb_run = wandb.init(
                project='cnn-quant',
                entity='cameron-research',
                name=args.exp_name,
                tags=args.tags,
                config={},
        )
        wandb_run.define_metric(
                name=f'Training Loss',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Training Accuracy',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Test Accuracy',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Num Bits',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Num Grad Bits',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Learning Rate',
                step_metric='Epoch',
        )
        wandb.config.update(args)

    return args


def main():
    args = parse_args()
    global save_path
    save_path = args.save_path = os.path.join(args.save_folder, "{}_num_bit_{}_{}_grad_bit_{}_{}".format(
            args.arch, args.cyclic_num_bits_schedule[0], args.cyclic_num_bits_schedule[1],
            args.cyclic_num_grad_bits_schedule[0], args.cyclic_num_grad_bits_schedule[1]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    run_training(args)


def run_training(args):
    model = models.__dict__[args.arch](args.pretrained)
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    # track when your model achieves the best performance
    best_prec1 = 0
    best_epoch = 0

    # the the input size to your network is not changing, this is good to set to True
    # it will run a benchmark to figure out the best way to handle this size as fast as possible
    # cudnn.benchmark = False

    print(f'Getting dataloaders...')
    train_loader = prepare_train_data(dataset=args.dataset,
                                      datadir=args.datadir+'/train',
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    datadir=args.datadir+'/val',
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    all_lrs = []
    all_bits = []
    all_grad_bits = []
    all_loss = []
    all_acc = []
    all_test_acc = []
    print(f'Running training...')
    for _epoch in range(args.epoch):
        # track metrics during training
        training_loss = 0
        training_acc = 0

        # adjust the learning rate per epoch
        lr = adjust_learning_rate(args, optimizer, _epoch)
        cyclic_adjust_precision(args, _epoch)
        if args.use_wandb:
            wandb.log({
                'Epoch': _epoch,
                'Learning Rate': lr,
                'Num Bits': args.num_bits,
                'Num Grad Bits': args.num_grad_bits,
            })
        else:
            all_lrs.append(lr)
            all_bits.append(args.num_bits)
            all_grad_bits.append(args.num_grad_bits)

        print(f'\nRunning Epoch {_epoch}...')
        print(f'\tLearning Rate: {lr}')
        print(f'\tNum Bits: {args.num_bits}')
        print(f'\tNum Grad Bits: {args.num_grad_bits}')

        model.train()
        for i, (inp, tgt) in enumerate(train_loader):
            try:
                inp = inp.cuda()
                tgt = tgt.long().cuda()

                # compute output
                output = model(inp, args.num_bits, args.num_grad_bits)
                loss = criterion(output, tgt)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure accuracy and record loss
                with torch.no_grad():
                    prec1, = accuracy(output.data, tgt, topk=(1,))
                    training_loss += loss.item()
                    training_acc += prec1.item()  
                    if (i  % 10) == 0:
                        print(f'\rIter. [{i}/{len(train_loader)}]: Loss {loss:.4f}; Train Acc {prec1:.4f}', end='')
            except:
                print('Skipped iteration')

        epoch_loss = training_loss / len(train_loader)
        epoch_acc = training_acc / len(train_loader)
        with torch.no_grad():
            prec1 = validate(args, test_loader, model, criterion)

        print(f'\nEpoch {_epoch} Results...')
        print(f'\tLoss: {epoch_loss:.4f}')
        print(f'\tTrain Acc: {epoch_acc:.4f}')
        print(f'\tTest Acc: {prec1:.4f}')

        if args.use_wandb:
            wandb.log({
                'Epoch': _epoch,
                'Training Loss': epoch_loss,
                'Training Accuracy': epoch_acc,
                'Test Accuracy': prec1,
            })
        else:
            all_loss.append(epoch_loss)
            all_acc.append(epoch_acc)
            all_test_acc.append(prec1)

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            best_epoch = _epoch
    
    if not args.use_wandb:
        results = {
            'lrs': all_lrs,
            'bits': all_bits,
            'grad_bits': all_grad_bits,
            'loss': all_loss,
            'acc': all_acc,
            'test_acc': all_test_acc,
        }
        save_fp = os.path.join(args.save_folder, 'result.pth')
        torch.save(results, save_fp)


def validate(args, test_loader, model, criterion):
    val_loss = 0.
    val_acc = 0.
    model.eval()
    for i, (inp, tgt) in enumerate(test_loader):
        inp = inp.cuda()
        tgt = tgt.long().cuda()

        # compute output
        output = model(inp, args.num_bits, args.num_grad_bits)
        loss = criterion(output, tgt)

        # measure accuracy and record loss
        prec1, = accuracy(output.data, tgt, topk=(1,))
        val_acc += float(prec1)
        val_loss += float(loss)
    return val_acc / len(test_loader)


def cyclic_adjust_precision(args, _epoch):
    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]
    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]
    assert num_grad_bit_min == num_grad_bit_max

    if _epoch < args.def_epochs:
        args.num_bits = num_bit_min
    else:
        args.num_bits = num_bit_max
    args.num_grad_bits = num_grad_bit_max

def adjust_learning_rate(args, optimizer, _epoch):
    if args.lr_schedule == 'piecewise':
        first_dec = int(0.5 * args.epoch)
        sec_dec = int(0.75 * args.epoch)
        if first_dec <= _epoch < sec_dec:
            lr = args.lr * (args.step_ratio ** 1)
        elif _epoch >= sec_dec:
            lr = args.lr * (args.step_ratio ** 2)
        else:
            lr = args.lr

    elif args.lr_schedule == 'piecewise-no-def-decay':
        assert args.epoch > args.def_epochs
        non_def_epochs = int(args.epoch - args.def_epochs)
        _nd_epoch = int(_epoch - args.def_epochs)
        first_dec = int(0.5 * non_def_epochs)
        sec_dec = int(0.75 * non_def_epochs)
        if first_dec <= _nd_epoch < sec_dec:
            lr = args.lr * (args.step_ratio ** 1)
        elif _nd_epoch >= sec_dec:
            lr = args.lr * (args.step_ratio ** 2)
        else:
            lr = args.lr

    elif args.lr_schedule == 'linear':
        t = _epoch / args.epoch
        lr_ratio = 0.01
        if t < 0.5:
            lr = args.lr
        elif t < 0.9:
            lr = args.lr * (1 - (1 - lr_ratio) * (t - 0.5) / 0.4)
        else:
            lr = args.lr * lr_ratio

    elif args.lr_schedule == 'anneal_cosine':
        lr_min = args.lr * (args.step_ratio ** 2)
        lr_max = args.lr
        lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(_epoch / args.epoch * 3.141592653))

    else:
        raise NotImplementedError(f'{args.lr_schedule} is not a supported lr schedule.')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



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
