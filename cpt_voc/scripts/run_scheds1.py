import os
import json

# setup
gpu = 1
workers = 4
use_wandb = True
tags = 'voc_scheds'
save_model = False
eval_len = 5
dataset = 'csv'
csv_train = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/train.csv'
csv_classes = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/class_map.csv'
csv_val = '/home/exx/data/VOCdevkit/VOC2012/CSVFormat/val.csv'
trials = 2
save_name = './results/voc_sched_rn18_6bit_01.json'
tmp_fn = 'voc_sched_00.txt'

# hparams
depth = 18
epochs = 120
batch_size = 8
lr = 3e-6
use_lr_decay = False

# cpt params
n = 8
qscheds = [
    # 'demon_decay', 'demon_growth',
    # 'exp_decay', 'exp_growth',
    # 'linear_decay', 'linear_growth',
    'cos_decay', 'cos_growth'
]
nbs_list = ['5 6']
ngbs_list = ['6 6']

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
        min_bits = int(nbs[0])
        max_bits = int(nbs[-1])
        
        if ps in ['demon_decay', 'exp_decay']:
            # vertical trials
            exp_name = f'voc_qsched_{depth}_{ps}_{min_bits}_{max_bits}_fv'
            for t in range(trials):
                full_exp_name = f'{exp_name}_{t}'
                command = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python train.py --dataset {dataset} --csv_train {csv_train} '
                    f'--csv_classes {csv_classes} --csv_val {csv_val} --depth {depth} --epochs {epochs} --lr {lr} '
                    f'--batch_size {batch_size} --workers {workers} --exp_name {full_exp_name} --eval_len {eval_len} '
                    f'--num_bits {max_bits} --num_grad_bits {max_bits} --num_cyclic_period {n} --precision_schedule {ps} '
                    f'--cyclic_num_bits_schedule {nbs} --cyclic_num_grad_bits_schedule {ngbs} --flip-vertically '
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


            # horizontal trials
            exp_name = f'voc_qsched_{depth}_{ps}_{min_bits}_{max_bits}_fh'
            for t in range(trials):
                full_exp_name = f'{exp_name}_{t}'
                command = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python train.py --dataset {dataset} --csv_train {csv_train} '
                    f'--csv_classes {csv_classes} --csv_val {csv_val} --depth {depth} --epochs {epochs} --lr {lr} '
                    f'--batch_size {batch_size} --workers {workers} --exp_name {full_exp_name} --eval_len {eval_len} '
                    f'--num_bits {max_bits} --num_grad_bits {max_bits} --num_cyclic_period {n} --precision_schedule {ps} '
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

        # other non-triangular schedules or when reflection is the same
        else:
            exp_name = f'voc_qsched_{depth}_{ps}_{min_bits}_{max_bits}'
            for t in range(trials):
                full_exp_name = f'{exp_name}_{t}'
                command = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python train.py --dataset {dataset} --csv_train {csv_train} '
                    f'--csv_classes {csv_classes} --csv_val {csv_val} --depth {depth} --epochs {epochs} --lr {lr} '
                    f'--batch_size {batch_size} --workers {workers} --exp_name {full_exp_name} --eval_len {eval_len} '
                    f'--num_bits {max_bits} --num_grad_bits {max_bits} --num_cyclic_period {n} --precision_schedule {ps} '
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