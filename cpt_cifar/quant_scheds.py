"""implements different quantization schedules"""

import numpy as np

def calc_cos_decay(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    return np.rint(max_val - 0.5*(max_val - min_val)*(1 + np.cos(np.pi * ((curr_iter % cyclic_period) / cyclic_period) + np.pi)))

def calc_cos_growth(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    return np.rint(min_val + 0.5*(max_val - min_val)*(1 + np.cos(np.pi * ((curr_iter % cyclic_period) / cyclic_period) + np.pi)))

def calc_demon_decay(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    mod_iter = curr_iter % cyclic_period
    z = (cyclic_period - mod_iter) / cyclic_period
    val = min_val + (max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    return np.rint(val)

def calc_demon_growth(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    mod_iter = curr_iter % cyclic_period
    z = (cyclic_period - mod_iter) / cyclic_period
    val = max_val - (max_val - min_val) * (z / (1 - 0.9 + 0.9*z))
    return val

def calc_exp_decay(cyclic_period, curr_iter, min_val, max_val, exponent=4):
    assert min_val <= max_val
    return min_val + (max_val - min_val)*np.exp((-exponent*(curr_iter%cyclic_period))/cyclic_period)

def calc_exp_growth(cyclic_period, curr_iter, min_val, max_val, exponent=4):
    assert min_val <= max_val
    return max_val - (max_val - min_val)*np.exp((-exponent*(curr_iter%cyclic_period))/cyclic_period)

def calc_linear_decay(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    mod_iter = curr_iter % cyclic_period
    val = max_val - ((max_val - min_val) * (mod_iter / cyclic_period))
    return val

def calc_linear_growth(cyclic_period, curr_iter, min_val, max_val):
    assert min_val <= max_val
    mod_iter = curr_iter % cyclic_period
    val = min_val + ((max_val - min_val) * (mod_iter / cyclic_period))
    return val
