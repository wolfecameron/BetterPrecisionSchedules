import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.quantize import QLSTM, QLinear
from torch.autograd import Variable

class PTB_QLSTM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, dropout=0.5):
        super(PTB_QLSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.rnn = QLSTM(ninp, nhid)
        # optionally quantize the output decoder
        #self.quantize_linear = quantize_linear
        #if quantize_linear:
        self.decoder = QLinear(nhid, ntoken)
        #else:
        #    self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.ntoken = ntoken
        self.ninp = ninp
        #self.quantize_linear = quantize_linear

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, num_bits, num_grad_bits):
        # embed the data
        emb = self.drop(self.encoder(input))

        # output and hidden will be quantized in QLinear and next forward pass, respectively
        output, hidden = self.rnn(emb, hidden, num_bits, num_grad_bits)
        output = self.drop(output)
        
        #if self.quantize_linear:
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)), num_bits, num_grad_bits)
        #else:
        #    decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
               
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.nhid).zero_()),
                Variable(weight.new(1, bsz, self.nhid).zero_()))
