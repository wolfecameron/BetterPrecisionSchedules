import os

import numpy as np
import torch

# 'critlearn_{dataset}_{arch}_{lr_sched}_{dit}_{t}/'

res_base = './quant_results/'
base_name = 'cifar10_cifar10_resnet_74_clprobe'
exclude_str = 'nothing'
model_name = 'cifar10_resnet_74/'
perf_file_name = 'best_results.pth'
cost_file_name = 'best_results.pth'

all_path = [x for x in os.listdir(res_base) if base_name in x and not exclude_str in x]
all_path_grouped = {}
for x in all_path:
    if 'vertical' in x or 'horizontal' in x:
        flip = x[x.rfind('_') + 1:]
        exp_str = x[len(base_name) + 1: x.rfind('_') - 2]
        exp_str = exp_str + flip
    else:
        exp_str = x[len(base_name) + 1:-2]
   
    fullp = os.path.join(os.path.join(res_base, x), f'{model_name}')
    if not exp_str in all_path_grouped.keys():
        all_path_grouped[exp_str] = [fullp]
    else:
        all_path_grouped[exp_str].append(fullp)

for k in sorted(list(all_path_grouped.keys())):
    fns = all_path_grouped[k]
    all_cost = []
    all_res = []
    for fn in fns:
        acc_res = torch.load(os.path.join(fn, perf_file_name))
        all_res.append(acc_res['best_prec1'])
        cost_res = torch.load(os.path.join(fn, cost_file_name))
        all_cost.append(cost_res['train_mets'][2])
    all_res = np.array(all_res)
    all_cost = np.array(all_cost)
    print(f'\n{k} --> {np.mean(all_res):.2f} +- {np.std(all_res):.2f}, {np.mean(all_cost):.4f} +- {np.std(all_cost):.4f}')