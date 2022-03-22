# simple script for computing gbitops

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

sched = calc_exp_growth
num_cycles = 16

iters = 64000
cyclic_period = int(iters / num_cycles)
running_sum = 0.
count = 0.
for i in range(iters):
    num_bits = sched(cyclic_period, i, 3.0, 8.0, discrete=True)
    fw_cost = num_bits * num_bits / 32.0 / 32.0
    eb_cost = num_bits * 8.0 / 32.0 / 32.0
    gc_cost = eb_cost
    running_sum += (fw_cost + eb_cost + gc_cost) / 3.0
    count += 1.0

print(f'{str(sched)}, {num_cycles} cycles: {running_sum / count:.4f} avg gbitops')
