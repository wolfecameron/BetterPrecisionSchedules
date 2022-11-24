# training script for ogbn arxiv

import argparse

import wandb
import torch
import torch.nn.functional as F

from models import QGCN, QGAT, QGATPlus

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import numpy as np
from quant_scheds import (
    calc_cos_decay,
    calc_cos_growth,
    calc_demon_decay,
    calc_demon_growth,
    calc_exp_decay,
    calc_exp_growth,
    calc_linear_decay,
    calc_linear_growth,
)

def train(args, g, model, train_idx, optimizer, labels):
    model.train()

    optimizer.zero_grad()
    out = model(g.ndata['feat'], args.num_bits, args.num_grad_bits)[train_idx]
    loss = F.nll_loss(out, labels[train_idx].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()

def test(args, g, model, labels, split_idx, evaluator):
    with torch.no_grad():
        model.eval()

        out = model(g.ndata['feat'], args.num_bits, args.num_grad_bits)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': labels[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': labels[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': labels[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--exp-name', type=str, default='gnn_quant_00')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    
    parser.add_argument('--arch', type=str, default='gnn', choices=['gnn', 'gat', 'gat-plus'])
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--merge', type=str, choices=['proj', 'cat', 'mean'], default='mean')
    parser.add_argument('--use-layer-norm', action='store_true', default=False)
    parser.add_argument('--use-res-conn', action='store_true', default=False)
    parser.add_argument('--norm-attn', action='store_true', default=False)
    parser.add_argument('--use-classif-layer', action='store_true', default=False)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='fixed')
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--eval_every', default=25, type=int,
                        help='evaluate model every (default: 25) iterations')
    
    # cpt_params
    parser.add_argument('--precision_schedule', default='cos_growth', type=str)
    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    parser.add_argument('--num_cyclic_period', default=1, type=int,
                        help='number of cyclic period for precision, same for weights/activation and gradients')
    parser.add_argument('--flip-vertically', action='store_true', default=False)
    
    # GNN-specific cpt params
    parser.add_argument('--quant-norm', action='store_true', default=False)
    parser.add_argument('--quant-agg', action='store_true', default=False)

    # these are updated by the scheduling code
    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--tags', type=str, action='append', default=None)
    args = parser.parse_args()

    if args.use_wandb:
        wandb_run = wandb.init(project="gnn-quant", entity="cameron-research", name=args.exp_name,
                tags=args.tags)
        wandb_run.define_metric(
                name=f'Loss',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Test Accuracy',
                step_metric='Epoch',
        )
        wandb_run.define_metric(
                name=f'Validation Accuracy',
                step_metric='Epoch',
        )
        wandb.config = args.__dict__

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    evaluator = Evaluator(name='ogbn-arxiv')

    g, labels = dataset[0]
    g = dgl.to_bidirected(g, copy_ndata=True)
    labels = labels.to(device)

    # normalize the data
    feat = g.ndata['feat']
    feat = (feat - feat.mean(0)) / feat.std(0)
    g.ndata['feat'] = feat
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    g = g.to(device)
    n_features = feat.size()[-1]
    n_classes = dataset.num_classes
    if args.arch == 'gnn':
        model = QGCN(g, n_features, args.n_hidden, n_classes, args.n_layers,
                F.relu, args.dropout, quant_norm=args.quant_norm,
                quant_agg=args.quant_agg).to(device)
    elif args.arch == 'gat':
        model = QGAT(g, n_features, args.n_hidden, n_classes, args.n_heads,
                args.dropout, quant_agg=args.quant_agg, dpt_inp=args.dpt_inp,
                merge=args.merge, dpt_attn=args.dpt_attn).to(device) 
    elif args.arch == 'gat-plus':
        model = QGATPlus(g, n_features, args.n_hidden, n_classes, args.n_heads,
                p=args.dropout, quant_agg=args.quant_agg, merge=args.merge,
                use_layer_norm=args.use_layer_norm, use_res_conn=args.use_res_conn,
                norm_attn=args.norm_attn, use_classif_layer=args.use_classif_layer).to(device)
    else:
        raise NotImplementedError()
    vals, tests = [], []
    optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.n_epochs):
        if args.verbose:
            print(f'Running epoch {epoch}')

        # update learning rate
        adjust_learning_rate(args, optimizer, epoch)

        # update quantization
        cyclic_period = int(args.n_epochs / args.num_cyclic_period) + 1
        cyclic_adjust_precision(args, epoch, cyclic_period)

        loss = train(args, g, model, train_idx, optimizer, labels)
        if args.use_wandb:
            wandb.log({
                    f"Loss": loss,
                    f"Epoch": epoch,
            })
        if epoch % args.test_freq == 0 or epoch == args.n_epochs:
            result = test(args, g, model, labels, split_idx, evaluator)
            _, val, tst = result
            vals.append(val)
            tests.append(tst)
            if args.verbose:
                print(f'\tVal Acc: {vals[-1]:.4f}')
                print(f'\tTest Acc: {tests[-1]:.4f}')
            if args.use_wandb:
                wandb.log({
                        f"Test Accuracy": tests[-1],
                        f"Epoch": epoch,
                })
                wandb.log({
                        f"Validation Accuracy": vals[-1],
                        f"Epoch": epoch,
                })

    print(f"Final Test Accuracy: {tests[-1]:.4f}")
    print(f"Best Val Accuracy: {max(vals):.4f}")
    print(f"Best Test Accuracy: {max(tests):.4f}")


def cyclic_adjust_precision(args, _epoch, cyclic_period):
    assert len(args.cyclic_num_bits_schedule) == 2
    assert len(args.cyclic_num_grad_bits_schedule) == 2

    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]

    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

    if args.precision_schedule == 'fixed':
        assert num_bit_min == num_bit_max
        assert num_grad_bit_min == num_grad_bit_max
        args.num_bits = num_bit_min
        args.num_grad_bits = num_grad_bit_min
    elif args.precision_schedule == 'cos_decay':
        num_period = int(_epoch / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_cos_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_cos_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_cos_decay(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_cos_decay(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'cos_growth':
        args.num_bits = calc_cos_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_cos_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'demon_decay':
        num_period = int(_epoch / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_demon_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_demon_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_demon_decay(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True, flip_vertically=args.flip_vertically)
            args.num_grad_bits = calc_demon_decay(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=args.flip_vertically)
    elif args.precision_schedule == 'demon_growth':
        args.num_bits = calc_demon_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_demon_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'exp_decay':
        num_period = int(_epoch / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_exp_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_exp_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_exp_decay(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True, flip_vertically=args.flip_vertically)
            args.num_grad_bits = calc_exp_decay(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=args.flip_vertically)
    elif args.precision_schedule == 'exp_growth':
        args.num_bits = calc_exp_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_exp_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'linear_decay':
        num_period = int(_epoch / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_linear_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_linear_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_linear_decay(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_linear_decay(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'linear_growth':
        args.num_bits = calc_linear_growth(cyclic_period, _epoch, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_linear_growth(cyclic_period, _epoch, num_grad_bit_min, num_grad_bit_max, discrete=True)
    else:
        raise NotImplementedError(f'{args.precision_schedule} is not a supported precision schedule.')
    if _epoch % args.eval_every == 0:
        print('Epoch [{}] num_bits = {} num_grad_bits = {} cyclic precision'.format(_epoch, args.num_bits,
                                                                                              args.num_grad_bits))

def adjust_learning_rate(args, optimizer, _epoch):
    if args.lr_schedule == 'fixed':
        lr = args.lr
    
    elif args.lr_schedule == 'piecewise':
        first_dec = int(0.5 * args.n_epochs)
        sec_dec = int(0.75 * args.n_epochs)
        if first_dec <= _epoch < sec_dec:
            lr = args.lr * 0.1
        elif _epoch >= sec_dec:
            lr = args.lr * 0.01
        else:
            lr = args.lr

    elif args.lr_schedule == 'linear':
        t = _epoch / args.n_epochs
        lr_ratio = 0.01
        if t < 0.5:
            lr = args.lr
        elif t < 0.9:
            lr = args.lr * (1 - (1 - lr_ratio) * (t - 0.5) / 0.4)
        else:
            lr = args.lr * lr_ratio

    elif args.lr_schedule == 'anneal_cosine':
        lr_min = args.lr * 0.1
        lr_max = args.lr
        lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(_epoch / args.n_epochs * 3.141592653))

    else:
        raise NotImplementedError(f'{args.lr_schedule} is not a supported lr schedule.')

    if _epoch % args.eval_every == 0:
        print('Epoch [{}] learning rate = {}'.format(_epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    main()
