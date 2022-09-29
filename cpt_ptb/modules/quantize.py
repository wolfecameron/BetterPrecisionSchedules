from collections import namedtuple
import math
from typing import List, Tuple, Optional, overload
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction, Function
from torch.nn.utils.rnn import PackedSequence
from torch.nn.modules.rnn import (
    RNNBase,
    apply_permutation
)
from torch import Tensor, _VF

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


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super(QLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.quantize_input = QuantMeasure(shape_measure=(1, 1), flatten_dims=(0, -1))

    def forward(self, x, num_bits, num_grad_bits):
        # quantize the input
        qx = self.quantize_input(x, num_bits)

        # quantize the weights
        weight_qparam = calculate_qparams(self.weight, num_bits=num_bits, flatten_dims=None,
                reduce_dim=None, reduce_type='mean')
        qweight = quantize(self.weight, qparams=weight_qparam)

        # quantize the bias
        if self.bias is not None:
            bias_qparam = calculate_qparams(self.bias, num_bits=num_bits, flatten_dims=None,
                    reduce_dim=None, reduce_type='mean')
            qbias = quantize(self.bias, qparams=bias_qparam)
        else:
            qbias = None

        # run quantized forward pass
        output = F.linear(qx, qweight, qbias)
        output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=None)
        return output


class QLSTM(nn.Module):
    """quantized, single-layer LSTM"""
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        #self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        #self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        #self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))

        # use quantized version of weight matrices
        # bias vector contained implicitly within W
        self.W = QLinear(input_sz, hidden_sz * 4, bias=True)
        self.U = QLinear(hidden_sz, hidden_sz * 4, bias=False)
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states, num_bits, num_grad_bits):
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
            gates = self.W(x_t, num_bits, num_grad_bits) + self.U(h_t, num_bits, num_grad_bits)

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


if __name__ == '__main__':
    #x = torch.rand(2, 3)
    #x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    #print(x)
    #print(x_q)
    LSTM(100, 100)
