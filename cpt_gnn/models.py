# has the implementation of all gcn models used for quantization experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

from modules.quantize import QGraphConv, MultiHeadQGATLayer

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
            dropout, use_layernorm=True, quant_norm=False, quant_agg=False):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.use_layernorm = use_layernorm

        # for now -- we apply quantization to all layers
        self.layers.append(QGraphConv(in_feats, n_hidden, activation=activation,
                quant_norm=quant_norm, quant_agg=quant_agg))   
        for i in range(n_layers - 1):
            self.layers.append(QGraphConv(n_hidden, n_hidden, activation=activation,
                    quant_norm=quant_norm, quant_agg=quant_agg))
        self.layers.append(QGraphConv(n_hidden, n_classes,
                quant_norm=quant_norm, quant_agg=quant_agg))
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

class GATPlus(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, p=0.6,
            quant_agg=False, dpt_inp=False, dpt_attn=False, use_layer_norm=False,
            use_res_conn=False, norm_attn=False, project_attn=False, use_classif_layer=False):
        super().__init__()
        self.g = g
        self.layer1 = MultiHeadQGATLayer(in_dim, hidden_dim, num_heads, p=p,
                quant_agg=quant_agg, dpt_inp=dpt_inp, dpt_attn=dpt_attn)
        self.layer2 = MultiHeadQGATLayer(hidden_dim, out_dim, 1, p=p,
                quant_agg=quant_agg, dpt_inp=dpt_inp, dpt_attn=dpt_attn)

    def forward(self, h, num_bits, num_grad_bits):
        h = self.layer1(self.g, h, num_bits, num_grad_bits)
        h = F.elu(h)
        h = self.layer2(self.g, h, num_bits, num_grad_bits)
        return h
    

class QGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, p=0.6,
            quant_agg=False, dpt_inp=False, dpt_attn=False):
        super(QGAT, self).__init__()
        self.g = g
        self.layer1 = MultiHeadQGATLayer(in_dim, hidden_dim, num_heads, p=p,
                quant_agg=quant_agg, dpt_inp=dpt_inp, dpt_attn=dpt_attn)
        self.layer2 = MultiHeadQGATLayer(hidden_dim, out_dim, 1, p=p,
                quant_agg=quant_agg, dpt_inp=dpt_inp, dpt_attn=dpt_attn)

    def forward(self, h, num_bits, num_grad_bits):
        h = self.layer1(self.g, h, num_bits, num_grad_bits)
        h = F.elu(h)
        h = self.layer2(self.g, h, num_bits, num_grad_bits)
        return h
