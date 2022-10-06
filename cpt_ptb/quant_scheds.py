"""implements different quantization schedules"""

import numpy as np

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


if __name__=='__main__':
    import matplotlib.pyplot as plt
    discrete=False
    fv = True
    lw=2
    total_iter = 64000
    #cd = [calc_cos_decay(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    #cg = [calc_cos_growth(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    #dd = [calc_demon_decay(total_iter, i, 3.0, 8.0, discrete=discrete, flip_vertically=fv) for i in range(total_iter)]
    #dg = [calc_demon_growth(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    ed = [calc_exp_decay(total_iter, i, 3.0, 8.0, discrete=discrete, flip_vertically=fv) for i in range(total_iter)]
    eg = [calc_exp_growth(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    #ld = [calc_linear_decay(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    #lg = [calc_linear_growth(total_iter, i, 3.0, 8.0, discrete=discrete) for i in range(total_iter)]
    #plt.plot(cd, label='cos decay', linewidth=lw)
    #plt.plot(cg, label='cos growth', linewidth=lw)
    #plt.plot(dd, label='demon decay', linewidth=lw)
    #plt.plot(dg, label='demon growth', linewidth=lw)
    plt.plot(ed, label='exp decay', linewidth=lw)
    plt.plot(eg, label='exp growth', linewidth=lw)
    #plt.plot(ld, label='linear decay', linewidth=lw)
    #plt.plot(lg, label='linear growth', linewidth=lw)
    leg = plt.legend()
    for legobj in leg.legendHandles:
        legobj.set_linewidth(lw)
    plt.show()
