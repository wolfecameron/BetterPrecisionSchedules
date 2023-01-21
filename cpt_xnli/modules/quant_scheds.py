"""implements different quantization schedules"""

import numpy as np

def cyclic_adjust_precision(args, _iter, cyclic_period):
    num_bit_min = args.num_bits_min
    num_bit_max = args.num_bits_max
    num_grad_bit_min = args.num_grad_bits_min
    num_grad_bit_max = args.num_grad_bits_max

    if args.precision_schedule == 'fixed':
        assert num_bit_min == num_bit_max
        assert num_grad_bit_min == num_grad_bit_max
        args.num_bits = num_bit_min
        args.num_grad_bits = num_grad_bit_min
    elif args.precision_schedule == 'cos_decay':
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
    return args

def calc_cos_decay(cyclic_period, curr_iter, min_val, max_val, discrete=True):
    assert min_val <= max_val
    z = float((curr_iter % cyclic_period)) / cyclic_period
    val = max_val - 0.5*float(max_val - min_val)*(1.0 - np.cos(np.pi*z))
    if discrete:
        val = np.rint(val)
    return val

def calc_cos_growth(cyclic_period, curr_iter, min_val, max_val, discrete=True):
    assert min_val <= max_val
    z = float((curr_iter % cyclic_period)) / cyclic_period
    val = min_val + 0.5*float(max_val - min_val)*(1.0 - np.cos(np.pi*z))
    if discrete:
        val = np.rint(val)
    return val

def calc_demon_decay(cyclic_period, curr_iter, min_val, max_val, discrete=True, flip_vertically=True):
    assert min_val <= max_val
    mod_iter = float(curr_iter % cyclic_period)
    z = float(cyclic_period - mod_iter) / cyclic_period
    if flip_vertically:
        val = min_val + float(max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    else:
        z = 1.0 - z
        val = max_val - float(max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    if discrete:
        val = np.rint(val)
    return val

def calc_demon_growth(cyclic_period, curr_iter, min_val, max_val, discrete=True):
    assert min_val <= max_val
    mod_iter = float(curr_iter % cyclic_period)
    z = float(cyclic_period - mod_iter) / cyclic_period
    val = max_val - float(max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    if discrete:
        val = np.rint(val)
    return val

def calc_exp_decay(cyclic_period, curr_iter, min_val, max_val, exponent=4, discrete=True, flip_vertically=True):
    assert min_val <= max_val
    z = float(curr_iter % cyclic_period) / cyclic_period
    if flip_vertically:
        val = min_val + float(max_val - min_val)*np.exp(-exponent*z)
    else:
        z = 1.0 - z
        val = max_val - float(max_val - min_val)*np.exp(-exponent*z)
    if discrete:
        val = np.rint(val)
    return val

def calc_exp_growth(cyclic_period, curr_iter, min_val, max_val, exponent=4, discrete=True):
    assert min_val <= max_val
    z = float(curr_iter % cyclic_period) / cyclic_period
    val = max_val - float(max_val - min_val)*np.exp(-exponent*z)
    if discrete:
        val = np.rint(val)
    return val

def calc_linear_decay(cyclic_period, curr_iter, min_val, max_val, discrete=True):
    assert min_val <= max_val
    mod_iter = float(curr_iter % cyclic_period)
    val = max_val - (float(max_val - min_val) * (float(mod_iter) / cyclic_period))
    if discrete:
        val = np.rint(val)
    return val

def calc_linear_growth(cyclic_period, curr_iter, min_val, max_val, discrete=True):
    assert min_val <= max_val
    mod_iter = float(curr_iter % cyclic_period)
    val = min_val + (float(max_val - min_val) * (float(mod_iter) / cyclic_period))
    if discrete:
        val = np.rint(val)
    return val
