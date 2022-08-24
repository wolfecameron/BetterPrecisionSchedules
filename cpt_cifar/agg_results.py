import os

import numpy as np
import torch

res_base = './quant_results/'
base_name = 'cifar100_cifar100_resnet_152'
exclude_str = 'max6'
model_name = 'cifar100_resnet_152/'
file_name = 'best_results.pth'

all_path = [x for x in os.listdir(res_base) if base_name in x and not exclude_str in x]
all_path_grouped = {}
for x in all_path:
    if 'vertical' in x or 'horizontal' in x:
        flip = x[x.rfind('_') + 1:]
        exp_str = x[len(base_name) + 1: x.rfind('_') - 2]
        exp_str = exp_str + flip
    else:
        exp_str = x[len(base_name) + 1:-2]
    
    fullp = os.path.join(os.path.join(res_base, x), f'{model_name}{file_name}')
    if not exp_str in all_path_grouped.keys():
        all_path_grouped[exp_str] = [fullp]
    else:
        all_path_grouped[exp_str].append(fullp)

for k in sorted(list(all_path_grouped.keys())):
    fns = all_path_grouped[k]
    all_res = []
    for fn in fns:
        res = all_res.append(torch.load(fn)['best_prec1'])
    all_res = np.array(all_res)
    print(f'\n{k} --> {np.mean(all_res):.2f} +- {np.std(all_res):.2f}')
