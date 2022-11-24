import argparse
import wandb
import tqdm
import sklearn.metrics
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

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
from models import QGraphSAGE

def cyclic_adjust_precision(args, _iter, cyclic_period):
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
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_cos_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_cos_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_cos_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_cos_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'cos_growth':
        args.num_bits = calc_cos_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_cos_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'demon_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_demon_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_demon_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_demon_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True, flip_vertically=args.flip_vertically)
            args.num_grad_bits = calc_demon_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=args.flip_vertically)
    elif args.precision_schedule == 'demon_growth':
        args.num_bits = calc_demon_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_demon_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'exp_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_exp_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_exp_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_exp_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True, flip_vertically=args.flip_vertically)
            args.num_grad_bits = calc_exp_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=args.flip_vertically)
    elif args.precision_schedule == 'exp_growth':
        args.num_bits = calc_exp_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_exp_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'linear_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            args.num_bits = calc_linear_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_linear_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            args.num_bits = calc_linear_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            args.num_grad_bits = calc_linear_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif args.precision_schedule == 'linear_growth':
        args.num_bits = calc_linear_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        args.num_grad_bits = calc_linear_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    else:
        raise NotImplementedError(f'{args.precision_schedule} is not a supported precision schedule.')
    


def adjust_learning_rate(args, optimizer, _iter, total_iter):
    if args.lr_schedule == 'fixed':
        lr = args.lr
    
    elif args.lr_schedule == 'piecewise':
        first_dec = int(0.5 * total_iter)
        sec_dec = int(0.75 * total_iter)
        if first_dec <= _iter < sec_dec:
            lr = args.lr * 0.1
        elif _iter >= sec_dec:
            lr = args.lr * 0.01
        else:
            lr = args.lr

    elif args.lr_schedule == 'anneal_cosine':
        lr_min = args.lr * args.lr_ratio
        lr_max = args.lr
        lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos(_iter / total_iter * 3.141592653))

    else:
        raise NotImplementedError(f'{args.lr_schedule} is not a supported lr schedule.')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='OGBN-Products (SAGE)')
parser.add_argument('--exp-name', type=str, default='products_sage_00')

# GNN args
parser.add_argument('--n-hidden', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num-neighbors', type=int, default=4)

# opt args
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr-schedule', type=str, default='fixed')
parser.add_argument('--lr-ratio', type=float, default=0.01)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--batch-size', type=int, default=1024)
parser.add_argument('--test-freq', type=int, default=5)

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
    wandb_run = wandb.init(project="gnn-quant", entity="cameron-research",
            name=args.exp_name, tags=args.tags)
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
    wandb_run.define_metric(
            name=f'Num Bits',
            step_metric='Epoch',
    )
    wandb_run.define_metric(
            name=f'Learning Rate',
            step_metric='Epoch',
    )
    wandb.config = args.__dict__


# download the data
dataset = DglNodePropPredDataset('ogbn-products')
device = torch.device('cuda')

# extract graph/labels
graph, node_labels = dataset[0]
graph = dgl.add_reverse_edges(graph) # graph is not directed
graph.ndata['label'] = node_labels[:, 0]

# extract node features
node_features = graph.ndata['feat']
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()

# train/val/test splits
idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

# create the neighborhood sampler
sampler = dgl.dataloading.NeighborSampler([args.num_neighbors, args.num_neighbors])
train_dataloader = dgl.dataloading.DataLoader(graph, train_nids, sampler,
    device=device, batch_size=args.batch_size, shuffle=True, drop_last=False,
    num_workers=0)
valid_dataloader = dgl.dataloading.DataLoader(graph, valid_nids, sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, device=device)
test_dataloader = dgl.dataloading.DataLoader(graph, test_nids, sampler,
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=0, device=device)

# model and optimizer
model = QGraphSAGE(num_features, args.n_hidden, num_classes,
        dropout=args.dropout, quant_agg=args.quant_agg).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay)

# store iteration counters to track precision and lr schedules
_curr_iter = 0
total_iter = args.n_epochs * len(train_dataloader)
cycle_length = int(total_iter // args.num_cyclic_period) + 1

# training loop
test_accs = []
val_accs = []
for epoch in range(args.n_epochs):

    # training epoch
    model.train()
    epoch_loss = 0.
    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
            # update learning rate and iterations
            adjust_learning_rate(args, opt, _curr_iter, total_iter)
            cyclic_adjust_precision(args, _curr_iter, cycle_length)

            # perform training update
            inputs = mfgs[0].srcdata['feat']
            labels = mfgs[-1].dstdata['label']
            out = model(mfgs, inputs, args.num_bits, args.num_grad_bits)
            loss = F.cross_entropy(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # record metrics
            preds = out.argmax(1).detach().cpu().numpy()
            train_acc = sklearn.metrics.accuracy_score(labels.cpu().numpy(), preds)
            tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % train_acc}, refresh=False)
            epoch_loss += float(loss)

            # increment the iteration
            _curr_iter += 1

    epoch_loss = epoch_loss / len(train_dataloader)
    if args.use_wandb:
        wandb.log({
                f"Loss": epoch_loss,
                f"Epoch": epoch,
        })
    
    # record lr/precision after each epoch
    if args.use_wandb:
        for pg in opt.param_groups:
            _lr = pg['lr']
            break
        _nb = args.num_bits
        wandb.log({
                f"Learning Rate": _lr,
                f"Num Bits": _nb,
                f"Epoch": epoch
        })

    
    model.eval()

    # validation
    predictions = []
    labels = []
    with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
        for input_nodes, output_nodes, mfgs in tq:
            inputs = mfgs[0].srcdata['feat']
            labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(mfgs, inputs, args.num_bits, args.num_grad_bits).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        val_acc = sklearn.metrics.accuracy_score(labels, predictions)
        val_accs.append(val_acc)
        if args.use_wandb:
            wandb.log({
                    f"Validation Accuracy": val_acc,
                    f"Epoch": epoch,
            })
    
    # testing
    if (epoch % args.test_freq) == 0 or epoch == (args.n_epochs - 1): 
        predictions = []
        labels = []
        with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
            for input_nodes, output_nodes, mfgs in tq:
                inputs = mfgs[0].srcdata['feat']
                labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                predictions.append(model(mfgs, inputs, args.num_bits, args.num_grad_bits).argmax(1).cpu().numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            test_acc = sklearn.metrics.accuracy_score(labels, predictions)
            test_accs.append(test_acc)
        if args.use_wandb:
            wandb.log({
                    f"Test Accuracy": test_acc,
                    f"Epoch": epoch,
            })

print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
print(f"Best Val Accuracy: {max(val_accs):.4f}")
print(f"Best Test Accuracy: {max(test_accs):.4f}")