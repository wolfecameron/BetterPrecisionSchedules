import os
import json

gpu = 1
def_iters = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
iters = [x + 1000 for x in def_iters]
trials = 1

dpt = 0.2
lr = 1e-3
lrs = 'cosine'
layers = 2
hid = 512
wd = 0.
eval_len = 25
use_wandb = True
tags = 'gnn_arxiv_critlearn'

num_bits = '3 8'
num_grad_bits = '8 8'
max_bit = num_bits[-1]
min_bit = num_bits[0]
save_name = f'./results/arxiv_crit_learn_{min_bit}_{max_bit}_cosdecay_01.json'

def add_accs_to_results(results, name, accs):
    if name in results.keys():
        results[name]['final_test_acc'].append(accs[0])
        results[name]['best_val_acc'].append(accs[1])
        results[name]['best_test_acc'].append(accs[2])
    else:
        results[name] = {}
        results[name]['final_test_acc'] = [accs[0]]
        results[name]['best_val_acc'] = [accs[1]]
        results[name]['best_test_acc'] = [accs[2]]

    # continually save results as they are updated
    with open(save_name, 'w') as f:
        json.dump(results, f)

results = {}
for it, dit in zip(iters, def_iters):
    exp_name = f'arxiv_critlearn_{dit}'
    for t in range(trials):
        full_exp_name = exp_name + f'_{t}'
        command = (
            f'CUDA_VISIBLE_DEVICES={gpu} python arxiv_crit_learn.py --exp-name {full_exp_name} '
            f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} '
            f'--lr-schedule {lrs} --n-epochs {it} --weight-decay {wd} --eval_every {eval_len} '
            f'--def-epochs {dit} --cyclic_num_bits_schedule {num_bits} '
            f'--cyclic_num_grad_bits_schedule {num_grad_bits} '
        )
        if use_wandb:
            command += f'--use-wandb --tags {tags} '
        os.system(command + ' > arxiv_cl_output.txt')
        with open('arxiv_cl_output.txt', 'r') as f:
            trn_output = f.readlines()
        final_test_acc = float(trn_output[-3][-6:])
        best_val_acc = float(trn_output[-2][-6:])
        best_test_acc = float(trn_output[-1][-6:])
        acc_list = [final_test_acc, best_val_acc, best_test_acc]
        add_accs_to_results(results, exp_name, acc_list) 
