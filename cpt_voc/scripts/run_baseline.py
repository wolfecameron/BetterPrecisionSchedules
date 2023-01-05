import os

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
trials = 2

# hparams
depth = 101
epochs = 120
batch_size = 4
lr = 1e-5
use_lr_decay = False

# cpt params
nbs = '8 8'
ngbs = '8 8'
n = 8
ps = 'fixed'

for t in range(trials):
    bits = int(nbs[-1])
    exp_name = f'voc_qbaseline_{lr}_decay{use_lr_decay}_{depth}_{epochs}_{batch_size}_{bits}_{t}'
    command = (
        f'CUDA_VISIBLE_DEVICES={gpu} python train.py --dataset {dataset} --csv_train {csv_train} '
        f'--csv_classes {csv_classes} --csv_val {csv_val} --depth {depth} --epochs {epochs} --lr {lr} '
        f'--batch_size {batch_size} --workers {workers} --exp_name {exp_name} --eval_len {eval_len} '
        f'--num_bits {bits} --num_grad_bits {bits} --num_cyclic_period {n} --precision_schedule {ps} '
        f'--cyclic_num_bits_schedule {nbs} --cyclic_num_grad_bits_schedule {ngbs} '
    )
    if use_wandb:
        command += f'--use_wandb --tags {tags} '
    if use_lr_decay:
        command += f'--use_lr_decay '
    if save_model:
        command += f'--save_model '
    os.system(command)