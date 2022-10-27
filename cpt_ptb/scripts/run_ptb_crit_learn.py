import os
import json

def_iters = [25, 50, 75]
iters = [x + 100 for x in def_iters]

gpu = 0
data_path = '/home/exx/data/ptb'
emsize = 800
nhid = 800
lr = 20
clip = 0.25
bs = 20
seq_len = 35
dpt = 0.5
trials = 2

num_bits = '5 8'
num_grad_bits = '8 8'
max_bit = num_bits[-1]
min_bit = num_bits[0]
save_name = f'./results/ptb_crit_learn_{min_bit}_{max_bit}_02.json'

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
for it, dit in zip(iters, def_iters):
    exp_name = f'ptb_critlearn_{dit}'
    for t in range(trials):
        full_exp_name = exp_name + f'_{t}'
        command = (
            f'CUDA_VISIBLE_DEVICES={gpu} python ptb_crit_learn.py --exp-name {exp_name} '
            f'--data {data_path} --emsize {emsize} --nhid {nhid} --lr {lr} --clip {clip} '
            f'--epochs {it} --def-epochs {dit} --batch_size {bs} --bptt {seq_len} --dropout {dpt} '
            f'--cyclic_num_bits_schedule {num_bits} --cyclic_num_grad_bits_schedule {num_grad_bits} --verbose'
        )
        os.system(command + ' > ptb_cl_output.txt')
        with open('ptb_cl_output.txt', 'r') as f:
            trn_output = f.readlines()
        val_ppl = float(trn_output[-2].split(' ')[2])
        test_ppl = float(trn_output[-1].split(' ')[2])
        acc_list = [val_ppl, test_ppl]
        add_accs_to_results(results, exp_name, acc_list) 
