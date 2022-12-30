import os

# test one
# setup
gpu = 0
datadir = '/home/exx/data/imagenet/'
base_save_dir = './quant_results'

# training
def_epoch = [0, 100, 25]
total_epoch = [x + 150 for x in def_epoch]
bs = 256
workers = 8
lr_sched = 'piecewise'
lr = 0.1
step_ratio = 0.1
wd = 0.
pretrained = False
use_wandb = False
tags = 'imagenet_crit_learn'

# quant
num_bits = '4 8'
num_grad_bit = '8 8'
min_bit = num_bits[0]
max_bit = num_bits[-1]

# architecture
ds = 'imagenet'
arch = f'resnet18'

# stuff that changes
for it, dit in zip(total_epoch, def_epoch):
    exp_name = f'critlearn_{ds}_{arch}_{lr_sched}_quant{min_bit}_{dit}_02'
    save_dir = os.path.join(base_save_dir, exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    command = (f'CUDA_VISIBLE_DEVICES={gpu} python crit_learn.py --exp-name {exp_name} '
            f'--arch {arch} --dataset {ds} --datadir {datadir} --epoch {it} --def-epochs {dit} '
            f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
            f'--step_ratio {step_ratio}  --save_folder {save_dir} --workers {workers} '
            f'--cyclic_num_bits_schedule {num_bits} --cyclic_num_grad_bits_schedule {num_grad_bit} '
    )
    if use_wandb:
        command += f'--use-wandb --tags {tags} '
    os.system(command)