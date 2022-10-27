import os

import numpy as np
import torch

res_base = './quant_results/'
#exp_name = f'{ds}_{arch}_{ps}_quant_max{max_bit}_{c}_{t}_vertical/'
#{ds}_{arch}_{lr_sched}_quant_{min_bit}_{max_bit}_{dit}_00
base_name = 'imagenet_resnet34'
exclude_str = 'nothing'
model_name = 'imagenet_resnet34/'
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
    
    fullp = os.path.join(res_base, x) #f'{model_name}{file_name}')
    fullp = os.path.join(os.path.join(fullp, os.listdir(fullp)[0]), file_name)
    if not exp_str in all_path_grouped.keys():
        all_path_grouped[exp_str] = [fullp]
    else:
        all_path_grouped[exp_str].append(fullp)

for k in sorted(list(all_path_grouped.keys())):
    fns = all_path_grouped[k]
    all_cost = []
    all_res = []
    for fn in fns:
        res = torch.load(fn)
        all_res.append(res['best_prec1'])
        all_cost.append(res['train_mets'][2])
    all_res = np.array(all_res)
    all_cost = np.array(all_cost)
    print(f'{k} --> {np.mean(all_res):.2f} +- {np.std(all_res):.2f}, {np.mean(all_cost):.4f} +- {np.std(all_cost):.4f}')
