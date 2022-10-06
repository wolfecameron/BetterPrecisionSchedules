# runs tests with fixed quantization level on the arxiv OGB dataset

import os
import json

gpu = 0 
data_path = '/home/exx/data/ptb'
emsize = 800
nhid = 800
lr = 20
clip = 0.25
epochs = 40
bs = 20
bptt = 35
dpt = 0.5

tags = 'ptb_sched_quant0'
qscheds = ['demon_decay', 'demon_growth', 'exp_decay', 'exp_growth', 'linear_decay', 'linear_growth', 'cos_decay', 'cos_growth']
nbs = ['5 8', '5 6']
ngbs = ['8 8', '6 6']
cycles = [8]
trials = 2
save_name = f'./results/ptb_scheds_00.json'

def add_accs_to_results(results, name, accs):
    if name in results.keys():
        results[name]['val_ppl'].append(accs[0])
        results[name]['test_ppl'].append(accs[1])
    else:
        results[name] = {}
        results[name]['val_ppl'] = [accs[0]]
        results[name]['test_ppl'] = [accs[1]]

    # continually save results as they are updated
    with open(save_name, 'w') as f:
        json.dump(results, f)

results = {}
for nb, ngb in zip(nbs, ngbs):
    for ps in qscheds:
        if ps in ['demon_decay', 'exp_decay']:
            for c in cycles:
                # run vertical flip trials
                exp_name = f'ptb_{ps}_{nb[0]}_{nb[-1]}_{c}_vertical'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_ptb.py --exp-name {full_exp_name} '
                        f'--data {data_path} --emsize {emsize} --nhid {nhid} --lr {lr} '
                        f'--clip {clip} --epochs {epochs} --batch_size {bs} --bptt {bptt} '
                        f'--dropout {dpt} --precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngb} --num_cyclic_period {c} --use-wandb '
                        f'--tags {tags} --flip-vertically'
                    )
                    os.system(command + ' > ptb_sched_output.txt')
                    with open('ptb_sched_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    val_ppl = float(trn_output[-2].split(' ')[2])
                    test_ppl = float(trn_output[-1].split(' ')[2])
                    acc_list = [val_ppl, test_ppl]
                    add_accs_to_results(results, exp_name, acc_list) 
                
                # run horizontal flip trials
                exp_name = f'ptb_{ps}_{nb[0]}_{nb[-1]}_{c}_horizontal'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_ptb.py --exp-name {full_exp_name} '
                        f'--data {data_path} --emsize {emsize} --nhid {nhid} --lr {lr} '
                        f'--clip {clip} --epochs {epochs} --batch_size {bs} --bptt {bptt} '
                        f'--dropout {dpt} --precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngb} --num_cyclic_period {c} --use-wandb '
                        f'--tags {tags} '
                    )
                    os.system(command + ' > ptb_sched_output.txt')
                    with open('ptb_sched_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    val_ppl = float(trn_output[-2].split(' ')[2])
                    test_ppl = float(trn_output[-1].split(' ')[2])
                    acc_list = [val_ppl, test_ppl]
                    add_accs_to_results(results, exp_name, acc_list) 

        else:
            for c in cycles:
                # run vertical flip trials
                exp_name = f'ptb_{ps}_{nb[0]}_{nb[-1]}_{c}'
                for t in range(trials):
                    full_exp_name = exp_name + f'_{t}'
                    command = (
                        f'CUDA_VISIBLE_DEVICES={gpu} python train_ptb.py --exp-name {full_exp_name} '
                        f'--data {data_path} --emsize {emsize} --nhid {nhid} --lr {lr} '
                        f'--clip {clip} --epochs {epochs} --batch_size {bs} --bptt {bptt} '
                        f'--dropout {dpt} --precision_schedule {ps} --cyclic_num_bits_schedule {nb} '
                        f'--cyclic_num_grad_bits_schedule {ngb} --num_cyclic_period {c} --use-wandb '
                        f'--tags {tags} '
                    )
                    os.system(command + ' > ptb_sched_output.txt')
                    with open('ptb_sched_output.txt', 'r') as f:
                        trn_output = f.readlines()
                    val_ppl = float(trn_output[-2].split(' ')[2])
                    test_ppl = float(trn_output[-1].split(' ')[2])
                    acc_list = [val_ppl, test_ppl]
                    add_accs_to_results(results, exp_name, acc_list) 
