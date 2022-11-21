# runs tests with fixed quantization level on the arxiv OGB dataset

import os
import json

gpu = 1
epochs = 500
dpt = 0.2
lr = 1e-3
lrs = 'anneal_cosine'
layers = 2
hid = 512
wd = 5e-4
eval_len = 25
ngbs = '6 6'

tags = 'axiv_sched_quant1'
qscheds = ['demon_decay', 'demon_growth', 'exp_decay', 'exp_growth', 'linear_decay', 'linear_growth', 'cos_decay', 'cos_growth']
nbs = ['3 6']
cycles = [8]
trials = 3
save_name = './results/arxiv_sched_test_01.json'

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
for nb in nbs:
    for ps in qscheds:
        if ps in ['demon_decay', 'exp_decay']:
            for c in cycles:
                # run vertical flip trials
                exp_name = f'arxiv_{ps}_{nb[0]}_{nb[-1]}_{c}_vertical'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --exp-name {full_exp_name} '
                        f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
                        f'--n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
                        f'--precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngbs} --tags {tags} --flip-vertically '
                        f'--num_cyclic_period {c} '
                    )
                    os.system(command + ' > arxiv1_output.txt')
                    with open('arxiv1_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    final_test_acc = float(trn_output[-3][-6:])
                    best_val_acc = float(trn_output[-2][-6:])
                    best_test_acc = float(trn_output[-1][-6:])
                    acc_list = [final_test_acc, best_val_acc, best_test_acc]
                    add_accs_to_results(results, exp_name, acc_list) 
                    os.remove('arxiv1_output.txt')
                         
                # run horizontal flip trials
                exp_name = f'arxiv_{ps}_{nb[0]}_{nb[-1]}_{c}_horizontal'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --exp-name {full_exp_name} '
                        f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
                        f'--n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
                        f'--precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngbs} --use-wandb --tags {tags} '
                        f'--num_cyclic_period {c} '
                    )
                    os.system(command + ' > arxiv1_output.txt')
                    with open('arxiv1_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    final_test_acc = float(trn_output[-3][-6:])
                    best_val_acc = float(trn_output[-2][-6:])
                    best_test_acc = float(trn_output[-1][-6:])
                    acc_list = [final_test_acc, best_val_acc, best_test_acc]
                    add_accs_to_results(results, exp_name, acc_list) 
                    os.remove('arxiv1_output.txt')
        else:
            for c in cycles:
                # run vertical flip trials
                exp_name = f'arxiv_{ps}_{nb[0]}_{nb[-1]}_{c}'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --exp-name {full_exp_name} '
                        f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
                        f'--n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
                        f'--precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngbs} --use-wandb --tags {tags} '
                        f'--num_cyclic_period {c} '
                    )
                    os.system(command + ' > arxiv1_output.txt')
                    with open('arxiv1_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    final_test_acc = float(trn_output[-3][-6:])
                    best_val_acc = float(trn_output[-2][-6:])
                    best_test_acc = float(trn_output[-1][-6:])
                    acc_list = [final_test_acc, best_val_acc, best_test_acc]
                    add_accs_to_results(results, exp_name, acc_list) 
                    os.remove('arxiv1_output.txt')
