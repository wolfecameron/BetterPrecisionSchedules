import argparse
import time
import math

import wandb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from models import PTB_QLSTM
from modules import data
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


# Add ckp
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--exp-name', type=str, default='gnn_quant_00')
parser.add_argument('--data', type=str, default='/home/exx/data/ptb', # /input
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--precision_schedule', default='cos_growth', type=str)
parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for weight/act precision')
parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for grad precision')
parser.add_argument('--num_cyclic_period', default=1, type=int,
                    help='number of cyclic period for precision, same for weights/activation and gradients')
parser.add_argument('--flip-vertically', action='store_true', default=False)
parser.add_argument('--num_bits', default=0, type=int,
                    help='num bits for weight and activation')
parser.add_argument('--num_grad_bits', default=0, type=int,
                    help='num bits for gradient')
parser.add_argument('--use-wandb', action='store_true', default=False)
parser.add_argument('--tags', type=str, action='append', default=None)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

if args.use_wandb:
    wandb_run = wandb.init(project="lstm-quant", entity="cameron-research",
            name=args.exp_name, tags=args.tags)
    wandb_run.define_metric(
            name=f'Validation PPL',
            step_metric='Epoch',
    )
    wandb_run.define_metric(
            name=f'Test PPL',
            step_metric='Epoch',
    )
    wandb.config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'bptt': args.bptt,
        'dropout': args.dropout,
        'emsize': args.emsize,
        'nhid': args.nhid,
        'lr': args.lr,
        'precision_schedule': args.precision_schedule,
        'cyclic_num_bits_schedule': args.cyclic_num_bits_schedule,
        'cyclic_num_grad_bits_schedule': args.cyclic_num_grad_bits_schedule,
        'num_cyclic_period': args.num_cyclic_period,
        'flip_vertically': args.flip_vertically,
    }

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = PTB_QLSTM(ntokens, args.emsize, args.nhid, args.dropout)
model.cuda()
criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    return Variable(h.data)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    with torch.no_grad():
        data = Variable(source[i:i+seq_len])
        target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden, args.num_bits, args.num_grad_bits)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * float(criterion(output_flat, targets).data)
        hidden = tuple(repackage_hidden(h) for h in hidden) 
    return total_loss / len(data_source)

def train(epoch):
    iter_per_epoch = ((train_data.size(0) - 1) // args.bptt) + 1
    total_iters = args.epochs * (iter_per_epoch)
    cycle_length = (total_iters // args.num_cyclic_period) + 1

    model.train()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    print((train_data.size(0) - 1) // args.bptt)
    raise ""
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        # adjust the precision at every batch batch
        global_iter = (epoch * iter_per_epoch) + batch
        cyclic_adjust_precision(args, global_iter, cycle_length)

        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = tuple(repackage_hidden(h) for h in hidden)
        model.zero_grad()
        output, hidden = model(data, hidden, args.num_bits, args.num_grad_bits)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += float(loss.data)
    
    return total_loss / train_data.size(0) - 1

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
        #args.num_bits = np.rint(num_bit_min +
        #                        0.5 * (num_bit_max - num_bit_min) *
        #                        (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        #args.num_grad_bits = np.rint(num_grad_bit_min +
        #                             0.5 * (num_grad_bit_max - num_grad_bit_min) *
        #                             (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
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


# Loop over epochs.
lr = args.lr
best_val_loss = None
trn_losses = []
val_losses = []
val_ppls = []
test_losses = []
test_ppls = []

# At any point you can hit Ctrl + C to break out of training early.
for epoch in range(args.epochs):
    if args.verbose:
        print(f'\nRunning Epoch {epoch} / {args.epochs}:')
    trn_loss = train(epoch)
    trn_losses.append(trn_loss)

    # validation
    val_loss = evaluate(val_data)
    val_losses.append(val_loss)
    val_ppls.append(math.exp(val_loss))

    # testing
    test_loss = evaluate(test_data)
    test_losses.append(test_loss)
    test_ppls.append(math.exp(test_loss))
    if args.use_wandb:
        wandb.log({
            'Validation PPL': val_ppls[-1],
            'Test PPL': test_ppls[-1],
            'Epoch': epoch,
        })
    if args.verbose:
        print(f'\tValidation PPL: {val_ppls[-1]:.2f}')
        print(f'\tTest PPL: {test_ppls[-1]:.2f}')

    # Anneal the learning rate if no improvement has been seen in the validation dataset.
    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:
        lr /= 4.0

# report results
max_idx = np.argmin(np.array(val_ppls))
best_val_ppl = val_ppls[max_idx]
best_test_ppl = test_ppls[max_idx]
print(f"Validation PPL: {best_val_ppl:.2f}")
print(f"Test PPL: {best_test_ppl:.2f}")
