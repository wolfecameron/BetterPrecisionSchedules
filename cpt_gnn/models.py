# has the implementation of all gcn models used for quantization experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from modules.quantize import QGraphConv

class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation,
            dropout, use_layernorm=True):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm

        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))   
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i < len(self.layers) - 1 and self.use_layernorm:
                h = F.layer_norm(h, h.shape)
        return h


class QGCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation,
            dropout, use_layernorm=True):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm

        self.layers.append(QGraphConv(in_feats, n_hidden, activation=activation))   
        for i in range(n_layers - 1):
            self.layers.append(QGraphConv(n_hidden, n_hidden, activation=activation))
        self.layers.append(QGraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, num_bits, num_grad_bits):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h, num_bits, num_grad_bits)
            if i < len(self.layers) - 1 and self.use_layernorm:
                h = F.layer_norm(h, h.shape) # perform layernorm in full precision
        return h
