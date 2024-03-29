import os
import json

gpu = 3
start_defs = [100, 200, 300, 400]
end_defs = [x + 500 for x in start_defs]
iters = 1000
trials = 2

dpt = 0.5
lr = 1e-3
lrs = 'cosine'
layers = 2
hid = 512
wd = 5e-4
eval_len = 25

num_bits = '3 8'
num_grad_bits = '8 8'
max_bit = num_bits[-1]
min_bit = num_bits[0]
save_name = f'/data/crw13/gnn_quant/arxiv_crit_learn_probe_{min_bit}_{max_bit}_00.json'

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
for sd, ed in zip(start_defs, end_defs):
    exp_name = f'arxiv_critlearnprobe_{sd}_{ed}'
    for t in range(trials):
        full_exp_name = exp_name + f'_{t}'
        command = (
            f'CUDA_VISIBLE_DEVICES={gpu} python arxiv_crit_learn_probe.py --exp-name {full_exp_name} '
            f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} '
            f'--lr-schedule {lrs} --n-epochs {iters} --weight-decay {wd} --eval_every {eval_len} '
            f'--start-def {sd} --end-def {ed} --cyclic_num_bits_schedule {num_bits} '
            f'--cyclic_num_grad_bits_schedule {num_grad_bits} '
        )
        os.system(command + ' > arxiv_clp_output.txt')
        with open('arxiv_clp_output.txt', 'r') as f:
            trn_output = f.readlines()
        final_test_acc = float(trn_output[-3][-6:])
        best_val_acc = float(trn_output[-2][-6:])
        best_test_acc = float(trn_output[-1][-6:])
        acc_list = [final_test_acc, best_val_acc, best_test_acc]
        add_accs_to_results(results, exp_name, acc_list) 
