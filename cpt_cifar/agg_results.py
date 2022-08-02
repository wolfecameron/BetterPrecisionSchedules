import os

import numpy as np
import torch

res_base = './quant_results/'
base_name = 'cifar100_exp_growth_quant_05'
exclude_str = 'nothing'
model_name = 'cifar100_resnet_74/'
file_name = 'best_results.pth'

all_path = [x for x in os.listdir(res_base) if base_name in x and not exclude_str in x]
all_path = [os.path.join(os.path.join(res_base, x), f'{model_name}{file_name}') for x in all_path]
all_res = [torch.load(x) for x in all_path]
all_p1 = []
for p, r in zip(all_path, all_res):
    all_p1.append(r['best_prec1'])
    print(f'{p} --> {r["best_prec1"]:.2f}')
print(f'Agg Perf: {np.mean(all_p1):.2f} +- {np.std(all_p1):.4f}')
