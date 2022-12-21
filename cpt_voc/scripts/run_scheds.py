import os
import json

# setup
gpu = 0
workers = 4
use_wandb = True
tags = 'voc_baseline'
save_model = False
eval_len = 5
dataset = 'csv'
csv_train = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/train.csv'
csv_classes = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/class_map.csv'
csv_val = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/val.csv'
trials = 1
save_name = './results/voc_sched_00.json'
tmp_fn = 'voc_sched_00.txt'

# hparams
depth = 18
epochs = 150
batch_size = 8
lr = 3e-5
use_lr_decay = False

# cpt params
n = 8
qscheds = [
    'demon_decay', 'demon_growth',
    'exp_decay', 'exp_growth',
    'linear_decay', 'linear_growth',
    'cos_decay', 'cos_growth'
]
nbs_list = ['4 8', '4 6']
ngbs_list = ['8 8', '6 6']

def add_to_results(results, name, accs):
    if name in results.keys():
        results[name]['final_mAP'].append(accs[0])
        results[name]['best_mAP'].append(accs[1])
    else:
        results[name] = {}
        results[name]['final_mAP'] = [accs[0]]
        results[name]['best_mAP'] = [accs[1]]

    # continually save results as they are updated
    with open(save_name, 'w') as f:
        json.dump(results, f)

results = {}
for nbs, ngbs in zip(nbs_list, ngbs_list):
    for ps in qscheds:
        bits = int(nbs[0])
        exp_name = f'voc_qbaseline_{lr}_decay{use_lr_decay}_{depth}_{epochs}_{batch_size}_{bits}'
        for t in range(trials):
            full_exp_name = f'{exp_name}_{t}'
            command = (
                f'CUDA_VISIBLE_DEVICES={gpu} python train.py --dataset {dataset} --csv_train {csv_train} '
                f'--csv_classes {csv_classes} --csv_val {csv_val} --depth {depth} --epochs {epochs} --lr {lr} '
                f'--batch_size {batch_size} --workers {workers} --exp_name {full_exp_name} --eval_len {eval_len} '
                f'--num_bits {bits} --num_grad_bits {bits} --num_cyclic_period {n} --precision_schedule {ps} '
                f'--cyclic_num_bits_schedule {nbs} --cyclic_num_grad_bits_schedule {ngbs} '
            )
            if use_wandb:
                command += f'--use_wandb --tags {tags} '
            if use_lr_decay:
                command += f'--use_lr_decay '
            if save_model:
                command += f'--save_model '
            os.system(command + f' > {tmp_fn}')
            with open(tmp_fn, 'r') as f:
                trn_output = f.readlines()
            final_mAP = float(trn_output[-2][-6:])
            best_mAP = float(trn_output[-1][-6:])
            mAP_list = [final_mAP, best_mAP]
            add_to_results(results, exp_name, mAP_list) 
            os.remove(tmp_fn)