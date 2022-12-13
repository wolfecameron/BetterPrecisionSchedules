# has the implementation of all gcn models used for quantization experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv, GraphConv

from modules.quantize import QGraphConv, MultiHeadQGATLayer, SAGEQConv

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

class QGraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout=0.2, quant_agg=False):
        super().__init__()
        # implemented with mean aggregator
        self.conv1 = SAGEQConv(in_feats, h_feats, bias=True, quant_agg=quant_agg)
        self.conv2 = SAGEQConv(h_feats, num_classes, bias=True, quant_agg=quant_agg)
        self.h_feats = h_feats
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, mfgs, x, num_bits, num_grad_bits):
        # first SAGE layer
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst), num_bits, num_grad_bits)

        # activation and layer norm in full precision
        h = F.relu(h)
        h = F.layer_norm(h, h.shape)
        h = self.dropout(h)

        # second SAGE layer
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst), num_bits, num_grad_bits)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, mfgs):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        self.h_feats = h_feats
        self.mfgs = mfgs

    def forward(self, x):
        # Lines that are changed are marked with an arrow: "<---"

        h_dst = x[:self.mfgs[0].num_dst_nodes()]  # <---
        h = self.conv1(self.mfgs[0], (x, h_dst))  # <---
        h = F.relu(h)
        h_dst = h[:self.mfgs[1].num_dst_nodes()]  # <---
        h = self.conv2(self.mfgs[1], (h, h_dst))  # <---
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

class QGATPlus(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, p=0.6, merge='proj',
            quant_agg=False, use_layer_norm=False, use_res_conn=False, norm_attn=False,
            use_classif_layer=False):
        super().__init__()
        self.g = g
        self.use_classif_layer = use_classif_layer 

        self.layer1 = MultiHeadQGATLayer(in_dim, hidden_dim, num_heads, p=p,
                merge=merge, quant_agg=quant_agg, use_layer_norm=use_layer_norm,
                use_res_conn=use_res_conn, norm_attn=norm_attn, first_layer=True)
        if self.use_classif_layer:
            self.layer2 = MultiHeadQGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, p=p,
                    merge=merge, quant_agg=quant_agg, use_layer_norm=use_layer_norm,
                    use_res_conn=use_res_conn, norm_attn=norm_attn, first_layer=False)
            self.classif = nn.Linear(hidden_dim, out_dim, bias=False)
        else:
            self.layer2 = MultiHeadQGATLayer(hidden_dim, out_dim, num_heads=1, p=p,
                    merge='cat', quant_agg=quant_agg, use_layer_norm=use_layer_norm,
                    use_res_conn=use_res_conn, norm_attn=norm_attn, first_layer=False)

    def forward(self, h, num_bits, num_grad_bits):
        h = self.layer1(self.g, h, num_bits, num_grad_bits)
        h = F.elu(h)
        h = self.layer2(self.g, h, num_bits, num_grad_bits)
        if self.use_classif_layer:
            h = self.classif(h)
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
