# training script for ogbn arxiv

import argparse

import torch
import torch.nn.functional as F

from models import QGCN

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import numpy as np


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
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument('--dropout', type=float, default=0.5)
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
    parser.add_argument('--start-def', type=int)
    parser.add_argument('--end-def', type=int)
    parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for weight/act precision')
    parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                        help='cyclic schedule for grad precision')
    
    # these are updated by the scheduling code
    parser.add_argument('--num_bits', default=0, type=int,
                        help='num bits for weight and activation')
    parser.add_argument('--num_grad_bits', default=0, type=int,
                        help='num bits for gradient')
    args = parser.parse_args()

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

    model = QGCN(g, n_features, args.n_hidden, n_classes, args.n_layers,
            F.relu, args.dropout).to(device)
    vals, tests = [], []
    optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.n_epochs):
        if args.verbose:
            print(f'Running epoch {epoch}')

        # update learning rate
        adjust_learning_rate(args, optimizer, epoch)

        # update quantization
        cyclic_adjust_precision(args, epoch)

        loss = train(args, g, model, train_idx, optimizer, labels)
        if epoch % args.test_freq == 0 or epoch == args.n_epochs:
            result = test(args, g, model, labels, split_idx, evaluator)
            _, val, tst = result
            vals.append(val)
            tests.append(tst)
            if args.verbose:
                print(f'\tVal Acc: {vals[-1]:.4f}')
                print(f'\tTest Acc: {tests[-1]:.4f}')

    print(f"Final Test Accuracy: {tests[-1]:.4f}")
    print(f"Best Val Accuracy: {max(vals):.4f}")
    print(f"Best Test Accuracy: {max(tests):.4f}")


def cyclic_adjust_precision(args, _epoch):
    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]
    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]

    if args.start_def <= _epoch < args.end_def:
        args.num_bits = num_bit_min
    else:
        args.num_bits = num_bit_max
    args.num_grad_bits = num_grad_bit_max
    
    #if _epoch % args.eval_every == 0:
    #    print('Epoch [{}] num_bits = {} num_grad_bits = {}'.format(_epoch, args.num_bits,
    #                                                                            args.num_grad_bits))

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
    
    elif args.lr_schedule == 'cosine':
        lr_min = args.lr * 0.1
        lr_max = args.lr
        lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(_epoch / args.n_epochs * 3.141592653))
    
    elif args.lr_schedule == 'linear':
        t = _epoch / args.n_epochs
        lr_ratio = 0.01
        if t < 0.5:
            lr = args.lr
        elif t < 0.9:
            lr = args.lr * (1 - (1 - lr_ratio) * (t - 0.5) / 0.4)
        else:
            lr = args.lr * lr_ratio

    else:
        raise NotImplementedError(f'{args.lr_schedule} is not a supported lr schedule.')

    if _epoch % args.eval_every == 0:
        print('Epoch [{}] learning rate = {}'.format(_epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__=='__main__':
    main()
