from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
import dgl.function as fn
from dgl.utils import expand_as_pair

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        if flatten_dims is not None:
            x_flat = x.flatten(*flatten_dims) # make it have shape [batch_size, single_flat_dim]
        else:
            x_flat = x
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
  
        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin)

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
        scale = torch.max(scale, min_scale)

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False):
    if qparams:
        if qparams.num_bits:
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace)

    return x


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True,
                  signed=False, stochastic=True):
    if qparams:
        if qparams.num_bits:
            return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                               stochastic)
    elif num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic)

    return x


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            return q_input


class QGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, norm='both', weight=True, bias=True,
            activation=None, allow_zero_in_degree=False, quant_norm=False,
            quant_agg=False):
        super(QGraphConv, self).__init__()
        if norm not in ('none', 'both', 'right', 'left'):
            raise DGLError('Invalid norm value. Must be either "none", "both", "right" or "left".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.quant_norm = quant_norm
        self.quant_agg = quant_agg

        # two main parameters being used are weight and bias
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
    
    def forward(self, graph, feat, num_bits, num_grad_bits, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise ValueError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            aggregate_fn = fn.copy_src('h', 'm')
            assert edge_weight is None
            #if edge_weight is not None:
            #    assert edge_weight.shape[0] == graph.number_of_edges()
            #    graph.edata['_edge_weight'] = edge_weight
            #    aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            # quantize the source features
            feat_src, feat_dst = expand_as_pair(feat, graph)
            #feat_qparams = calculate_qparams(feat_src, num_bits=num_bits,
            #        flatten_dims=None, reduce_dim=None, reduce_type='extreme')
            #qfeat_src = quantize(feat_src, qparams=feat_qparams)

            # normalize by the out degree
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)

                if self.quant_norm:
                    # quantize the norm vector
                    norm_qparams = calculate_qparams(norm, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='mean')
                    qnorm = quantize(norm, qparams=norm_qparams)

                    # quantize the features
                    feat_qparams = calculate_qparams(feat_src, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                    qfeat_src = quantize(feat_src, feat_qparams)
                    
                    # normalize in low precision
                    feat_src = qfeat_src * qnorm
                    feat_src = quantize_grad(feat_src, num_bits=num_grad_bits, flatten_dims=None)
                
                else:
                    feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise ValueError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight
            
            # quantize the weights
            weight_qparams = calculate_qparams(weight, num_bits=num_bits,
                    flatten_dims=None, reduce_dim=None, reduce_type='mean')
            qweight = quantize(weight, qparams=weight_qparams)

            if self._in_feats > self._out_feats:
                # quantized matrix multiplication
                feat_qparams = calculate_qparams(feat_src, num_bits=num_bits,
                        flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                qfeat_src = quantize(feat_src, qparams=feat_qparams)
                if weight is not None:
                    qfeat_src = torch.matmul(qfeat_src, qweight)
                qfeat_src = quantize_grad(qfeat_src, num_bits=num_grad_bits, flatten_dims=None)

                # aggregate node features
                if self.quant_agg:
                    # quantize the features
                    feat_qparams = calculate_qparams(qfeat_src, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                    qfeat_src = quantize(qfeat_src, qparams=feat_qparams)

                    # aggregate low precision features
                    graph.srcdata['h'] = qfeat_src
                    graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                    qrst = graph.dstdata['h']
                    qrst = quantize_grad(qrst, num_bits=num_grad_bits, flatten_dims=None)
                else:
                    graph.srcdata['h'] = qfeat_src
                    graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                    qrst = graph.dstdata['h']
            else:
                # aggregate node features
                if self.quant_agg:
                    # quantize the features
                    feat_qparams = calculate_qparams(feat_src, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                    qfeat_src = quantize(feat_src, qparams=feat_qparams)

                    # aggregate low precision features
                    graph.srcdata['h'] = qfeat_src
                    graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                    rst = graph.dstdata['h']
                    rst = quantize_grad(rst, num_bits=num_grad_bits, flatten_dims=None)
                else:
                    graph.srcdata['h'] = feat_src
                    graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                    rst = graph.dstdata['h']

                # quantized matrix multiplication of features
                rst_qparams = calculate_qparams(rst, num_bits=num_bits,
                        flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                qrst = quantize(rst, qparams=rst_qparams)
                if weight is not None:
                    qrst = torch.matmul(qrst, qweight)
                qrst = quantize_grad(qrst, num_bits=num_grad_bits, flatten_dims=None)
            
            # normalize by the in degree in full precision
            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)

                if self.quant_norm:
                    # quantize the norm vector
                    norm_qparams = calculate_qparams(norm, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='mean')
                    qnorm = quantize(norm, qparams=norm_qparams)

                    # quantize the features
                    rst_qparams = calculate_qparams(qrst, num_bits=num_bits,
                            flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                    qrst = quantize(qrst, rst_qparams)
                    
                    # normalize in low precision
                    qrst = qrst * qnorm
                    qrst = quantize_grad(qrst, num_bits=num_grad_bits, flatten_dims=None)
                else:
                    qrst = qrst * norm

            # quantized addition of the bias 
            if self.bias is not None:
                # quantize the bias
                bias_qparams = calculate_qparams(self.bias, num_bits=num_bits,
                        flatten_dims=None, reduce_dim=None, reduce_type='mean')
                qbias = quantize(self.bias, qparams=bias_qparams)

                # quantize the features
                qrst_qparams = calculate_qparams(qrst, num_bits=num_bits,
                        flatten_dims=None, reduce_dim=None, reduce_type='extreme')
                qrst = quantize(qrst, qparams=qrst_qparams)
                
                # add bias and quantize the gradient
                qrst = qrst + qbias
                qrst = quantize_grad(qrst, num_bits=num_grad_bits, flatten_dims=None)

            if self._activation is not None:
                qrst = self._activation(qrst)

            return qrst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)



if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
