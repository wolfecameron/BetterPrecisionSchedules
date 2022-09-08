import argparse

import torch

from vanilla_models import (
    cifar10_resnet_74,
    cifar10_resnet_152,
    cifar10_mobilenet_v2,
)
from fvcore.nn import FlopCountAnalysis
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


def cyclic_adjust_precision(_iter, cyclic_period, flip_vertically,
        precision_schedule, num_bits, num_grad_bits):
    
    num_bit_min = int(num_bits[0])
    num_bit_max = int(num_bits[-1])

    num_grad_bit_min = int(num_grad_bits[0])
    num_grad_bit_max = int(num_grad_bits[-1])

    if precision_schedule == 'fixed':
        assert num_bit_min == num_bit_max
        assert num_grad_bit_min == num_grad_bit_max
        num_bits = num_bit_min
        num_grad_bits = num_grad_bit_min
    elif precision_schedule == 'cos_decay':
        #args.num_bits = np.rint(num_bit_min +
        #                        0.5 * (num_bit_max - num_bit_min) *
        #                        (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        #args.num_grad_bits = np.rint(num_grad_bit_min +
        #                             0.5 * (num_grad_bit_max - num_grad_bit_min) *
        #                             (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi)))
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            num_bits = calc_cos_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_cos_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            num_bits = calc_cos_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_cos_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif precision_schedule == 'cos_growth':
        num_bits = calc_cos_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        num_grad_bits = calc_cos_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif precision_schedule == 'demon_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            num_bits = calc_demon_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_demon_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            num_bits = calc_demon_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True, flip_vertically=flip_vertically)
            num_grad_bits = calc_demon_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=flip_vertically)
    elif precision_schedule == 'demon_growth':
        num_bits = calc_demon_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        num_grad_bits = calc_demon_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif precision_schedule == 'exp_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            num_bits = calc_exp_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_exp_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            num_bits = calc_exp_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True, flip_vertically=flip_vertically)
            num_grad_bits = calc_exp_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True, flip_vertically=flip_vertically)
    elif precision_schedule == 'exp_growth':
        num_bits = calc_exp_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        num_grad_bits = calc_exp_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif precision_schedule == 'linear_decay':
        num_period = int(_iter / cyclic_period)
        if (num_period % 2) == 1:
            num_bits = calc_linear_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_linear_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
        else:
            num_bits = calc_linear_decay(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
            num_grad_bits = calc_linear_decay(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    elif precision_schedule == 'linear_growth':
        num_bits = calc_linear_growth(cyclic_period, _iter, num_bit_min, num_bit_max, discrete=True)
        num_grad_bits = calc_linear_growth(cyclic_period, _iter, num_grad_bit_min, num_grad_bit_max, discrete=True)
    else:
        raise NotImplementedError(f'{precision_schedule} is not a supported precision schedule.')
    return num_bits, num_grad_bits

def compute_flops(flops, schedule): 
    total_flops = 0.
    for nb, ngb in schedule:
        fw_eflop = (nb**2) / (32**2)
        bw_eflop = (2 * nb * ngb) / (32**2)
        total_flops += flops * (fw_eflop + bw_eflop)
    return total_flops

arch = 'resnet18'
num_iter = 64000
num_cycle = 8
cycle_len = (num_iter // num_cycle)
num_grad_bits = '8 8'

prec_scheds = ['linear_growth', 'linear_decay', 'cos_growth', 'cos_decay', 'demon_growth', 'demon_decay', 'exp_growth', 'exp_decay']
num_bit_list = ['6 6']
flips = [True, False]


if arch == 'resnet74':
    model = cifar10_resnet_74()
elif arch == 'resnet152':
    model = cifar10_resnet_152()
elif arch == 'mobilenet':
    model = cifar10_mobilenet_v2()
else:
    raise NotImplementedError
inp = torch.zeros(128, 3, 32, 32).cpu()
flop_obj = FlopCountAnalysis(model, inp)
flops = flop_obj.total()

for num_bits in num_bit_list:
    for sched in prec_scheds:
        if not 'growth' in sched and not 'cos' in sched and not 'linear' in sched:
            for flip in flips:
                qs = [cyclic_adjust_precision(i, cycle_len, flip, sched, num_bits, num_grad_bits) for i in range(num_iter)]
                bitops = compute_flops(flops, qs)
                print(f'{sched} {num_bits} {flip} --> {bitops}') 
        else:
            qs = [cyclic_adjust_precision(i, cycle_len, True, sched, num_bits, num_grad_bits) for i in range(num_iter)]
            bitops = compute_flops(flops, qs)
            print(f'{sched} {num_bits} --> {bitops}') 

# use FLOPS * (bit / 32)^2 to compute effective FLOPS (proportional to # bit-opts)
# here, we seem to exclude FLOPS because it is constant
# calculate both backward + forward pass cost, backward is 2x cost of forward (included twice in sum)
#fw_cost = args.num_bits * args.num_bits / 32 / 32
#eb_cost = args.num_bits * args.num_grad_bits / 32 / 32
#gc_cost = eb_cost
