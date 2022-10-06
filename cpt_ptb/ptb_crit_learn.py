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


# Add ckp
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--exp-name', type=str, default='gnn_quant_00')
parser.add_argument('--data', type=str, default='/home/exx/data/ptb', 
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
parser.add_argument('--def-epochs', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cyclic_num_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for weight/act precision')
parser.add_argument('--cyclic_num_grad_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for grad precision')
parser.add_argument('--num_bits', default=0, type=int,
                    help='num bits for weight and activation')
parser.add_argument('--num_grad_bits', default=0, type=int,
                    help='num bits for gradient')
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()


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
    cyclic_adjust_precision(args, epoch)
    model.train()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt*10)):
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

def cyclic_adjust_precision(args, _epoch):
    assert len(args.cyclic_num_bits_schedule) == 2
    assert len(args.cyclic_num_grad_bits_schedule) == 2
    num_bit_min = args.cyclic_num_bits_schedule[0]
    num_bit_max = args.cyclic_num_bits_schedule[1]
    num_grad_bit_min = args.cyclic_num_grad_bits_schedule[0]
    num_grad_bit_max = args.cyclic_num_grad_bits_schedule[1]
    if _epoch <= args.def_epochs:
        args.num_bits = num_bit_min
    else:
        args.num_bits = num_bit_max
    args.num_grad_bits = num_grad_bit_max



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
    if args.verbose:
        print(f'\tValidation PPL: {val_ppls[-1]:.2f}')
        print(f'\tTest PPL: {test_ppls[-1]:.2f}')

    # Anneal the learning rate if no improvement has been seen in the validation dataset.
    # can add some logic here to only decay after def epochs
    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:
        lr /= 4.0

# report results
max_idx = np.argmin(np.array(val_ppls))
best_val_ppl = val_ppls[max_idx]
best_test_ppl = test_ppls[max_idx]
print(f"Validation PPL: {best_val_ppl:.4f}")
print(f"Test PPL: {best_test_ppl:.4f}")
