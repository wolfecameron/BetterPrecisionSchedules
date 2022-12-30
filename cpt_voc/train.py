import argparse
import collections
import os

import wandb

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet.model import qresnet18, qresnet101
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from quant_scheds import cyclic_adjust_precision

assert torch.__version__.split('.')[0] == '1'


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--exp_name', help='Name of the experiment', type=str, required=True)
    parser.add_argument('--tags', type=str, action='append', default=None)
    parser.add_argument('--use_wandb', help='Whether to log training metrics to wandb', action='store_true', default=False)


    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--save_model', help='Whether or not to persist the model to a file', action='store_true', default=False)
    parser.add_argument('--save-path', default=None, type=str, help='Path to which to save results')
    parser.add_argument('--eval_len', help='How frequently to evaluate at end of epoch', type=int, default=5)

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=1e-5)
    parser.add_argument('--use_lr_decay', action='store_true', default=False, help='Whether or not to decay lr')
    parser.add_argument('--batch_size', help='Number of images to include in each batch', type=int, default=2)
    parser.add_argument('--workers', help='Number of workers to use for data loading', type=int, default=8)

    # cpt params
    parser.add_argument('--num_bits', help='num bits for weight and activation', default=0,type=int)
    parser.add_argument('--num_grad_bits', help='num bits for gradient', default=0, type=int)
    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    parser.add_argument('--num_cyclic_period', help='number of cyclic period for precision', default=32, type=int)
    parser.add_argument('--flip-vertically', action='store_true', default=False)
    parser.add_argument('--precision_schedule', default='cos_growth', type=str)
    parser = parser.parse_args(args)


    if parser.use_wandb:
        wandb_run = wandb.init(
                project='cnn-quant',
                entity='cameron-research',
                name=parser.exp_name,
                tags=parser.tags,
                config={},
        )
        wandb_run.define_metric(
                name=f'Training Loss',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Classification Loss',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Regression Loss',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Test mAP',
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
        wandb.config.update(parser)


    if parser.csv_train is None:
        raise ValueError('Must provide --csv_train when training on COCO,')
    if parser.csv_classes is None:
        raise ValueError('Must provide --csv_classes when training on COCO,')
    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                    transform=transforms.Compose([Normalizer(), Resizer()]))

    
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler)
    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = qresnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = qresnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    assert torch.cuda.is_available(), "Error: cannot find the GPU!"
    retinanet = retinanet.cuda()

    # cycle length for CPT
    cyclic_period = int((parser.epochs * len(dataloader_train)) / parser.num_cyclic_period) + 1

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    all_mAPs = []
    all_lrs = []
    all_bits = []
    all_grad_bits = []
    all_losses = []
    for epoch_num in range(parser.epochs):
        print(f'Running Epoch {epoch_num} / {parser.epochs}')

        # optionally decay lr
        if parser.use_lr_decay and epoch_num == int(parser.epochs * 0.75):
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.1

        # log lr/precision to wandb 
        for pg in optimizer.param_groups:
            _clr = pg['lr']
            break
        if parser.use_wandb:
            wandb.log({
                'Epoch': epoch_num,
                'Learning Rate': _clr,
                'Num Bits': parser.num_bits,
                'Num Grad Bits': parser.num_grad_bits,
            })
        else:
            all_lrs.append(_clr)
            all_bits.append(parser.num_bits)
            all_grad_bits.append(parser.num_grad_bits)

        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []
        epoch_class_loss = []
        epoch_reg_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # update the precision according to CPT schedule
            _citer = epoch_num * len(dataloader_train) + iter_num
            cyclic_adjust_precision(parser, _citer, cyclic_period)

            optimizer.zero_grad()
            classification_loss, regression_loss = retinanet(
                    [data['img'].cuda().float(), data['annot'].cuda()],
                    parser.num_bits, parser.num_grad_bits)
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))
            epoch_class_loss.append(float(classification_loss))
            epoch_reg_loss.append(float(regression_loss))

            if (iter_num % 20) == 0:
                print(f'\rEpoch: {epoch_num} | Iteration: {iter_num} / {len(dataloader_train)} | Loss: {float(np.mean(loss_hist)):.2f}', end='')
            del classification_loss
            del regression_loss
        print('') # add newline for terminal output after making it through an epoch
        
        if (epoch_num % parser.eval_len) == 0 or (epoch_num == parser.epochs - 1):
            # if parser.dataset == 'coco':
            #     print('Evaluating dataset')
            #     coco_eval.evaluate_coco(dataset_val, retinanet)
            if parser.csv_val is not None:
                print('Evaluating dataset')
                APs = csv_eval.evaluate(dataset_val, retinanet, num_bits=parser.num_bits, num_grad_bits=parser.num_grad_bits)
                APs = [x[0] for x in APs.values()]
                mAP = float(np.mean(APs))
                if parser.use_wandb:
                    wandb.log({'Epoch': epoch_num, 'Test mAP': float(mAP)})
                all_mAPs.append(mAP)  

        # scheduler.step(np.mean(epoch_loss))
        
        if parser.use_wandb:
            wandb.log({
                'Epoch': epoch_num,
                'Training Loss': float(np.mean(epoch_loss)),
                'Classification Loss': float(np.mean(epoch_class_loss)),
                'Regression Loss': float(np.mean(epoch_reg_loss)),
            })
        else:
            all_losses.append(float(np.mean(epoch_loss)))

        if parser.save_model:
            torch.save(retinanet, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

    retinanet.eval()

    if parser.save_model:
        torch.save(retinanet, 'model_final.pt')

    # optionally save the results of training
    if not parser.use_wandb and parser.save_path is not None:
        results = {
            'all_mAPs': all_mAPs,
            'all_lrs': all_lrs,
            'all_bits': all_bits,
            'all_grad_bits': all_grad_bits,
            'all_losses': all_losses,
        }
        torch.save(results, os.path.join(parser.save_path, f'{parser.exp_name}.pth'))

    print(f"Final Test mAP: {all_mAPs[-1]:.4f}")
    print(f"Best Test mAP: {max(all_mAPs):.4f}")


if __name__ == '__main__':
    main()
