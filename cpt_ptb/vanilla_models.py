import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PTB_LSTM(nn.Module):
    def __init__(self, ntoken, ninp, nhid, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        #self.encoder = nn.Embedding(ntoken, ninp) # Token2Embeddings
        self.rnn = LSTM(ninp, nhid)
        # optionally quantize the output decoder
        #self.quantize_linear = quantize_linear
        #if quantize_linear:
        self.decoder = nn.Linear(nhid, ntoken)
        #else:
        #    self.decoder = nn.Linear(nhid, ntoken)
        #self.init_weights()
        self.nhid = nhid
        self.ntoken = ntoken
        self.ninp = ninp
        #self.quantize_linear = quantize_linear

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, emb):
        hidden = (torch.zeros(1, 20, 800).cpu(), torch.zeros(1, 20, 800).cpu())
        # pass directly in as a float for FLOP computation
        # embed the data
        #emb = self.drop(self.encoder(input))

        # output and hidden will be quantized in QLinear and next forward pass, respectively
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        
        #if self.quantize_linear:
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        #else:
        #    decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
               
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, self.nhid).zero_()),
                Variable(weight.new(1, bsz, self.nhid).zero_()))


class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        #self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        #self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        #self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        # use quantized version of weight matrices
        # bias vector contained implicitly within W
        self.W = nn.Linear(input_sz, hidden_sz * 4, bias=True)
        self.U = nn.Linear(hidden_sz, hidden_sz * 4, bias=False)
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states):
        seq_sz, bs, _ = x.size()
        hidden_seq = []
        #if init_states is None:
        #    h_t, c_t = (torch.zeros(1, bs, self.hidden_size).to(x.device), 
        #                torch.zeros(1, bs, self.hidden_size).to(x.device))
        #else:
        assert init_states is not None
        h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :][None, :]
            #gates = x_t @ self.W + h_t @ self.U + self.bias
            gates = self.W(x_t) + self.U(h_t)

            # activation functions/gating are applied in full precision
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :, :HS]), # input
                torch.sigmoid(gates[:, :, HS:HS*2]), # forget
                torch.tanh(gates[:, :, HS*2:HS*3]),
                torch.sigmoid(gates[:, :, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_t, c_t)
