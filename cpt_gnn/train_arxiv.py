# training script for ogbn arxiv

import argparse

import torch
import torch.nn.functional as F

from models import QGCN

from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import dgl
import numpy as np

def train(g, model, train_idx, optimizer, labels):
    model.train()

    optimizer.zero_grad()
    out = model(g.ndata['feat'], 8, 8)[train_idx]
    loss = F.nll_loss(out, labels[train_idx].squeeze())
    loss.backward()
    optimizer.step()

    return loss.item()

def test(g, model, labels, split_idx, evaluator):
    with torch.no_grad():
        model.eval()

        out = model(g.ndata['feat'], 8, 8)
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
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
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
    for epoch in range(1, args.n_epochs + 1):
        if args.verbose:
            print(f'Running epoch {epoch}')
        loss = train(g, model, train_idx, optimizer, labels)
        if epoch % args.test_freq == 0 or epoch == args.n_epochs:
            result = test(g, model, labels, split_idx, evaluator)
            _, val, tst = result
            vals.append(val)
            tests.append(tst)
            if args.verbose:
                print(f'\tVal Acc: {vals[-1]:.4f}')
                print(f'\tTest Acc: {tests[-1]:.4f}')


    print(f"Final Test Accuracy: {tests[-1]:.4f}")
    print(f"Best Val Accuracy: {max(vals):.4f}")
    print(f"Best Test Accuracy: {max(tests):.4f}")

if __name__=='__main__':
    main()
