import os

# test one
# setup
gpu = 0
datadir = './data/'
base_save_dir = './quant_results'
eval_every = 390
trials = 1

# training
def_iters = [0, 16000, 32000, 64000, 128000, 256000]
iters = [x + 256000 for x in def_iters]
bs = 128
lr_sched = 'piecewise-no-def-decay'
dataset = 'cifar10'
lr = 0.1
step_ratio = 0.1
wd = 0.
warm_up = False
pretrained = False

# quant
num_bits = '3 8'
num_grad_bit = '8 8'
tags = 'cifar_cl'

# stuff that changes
arch = f'{dataset}_resnet_74'
max_bit = num_bits[-1]
min_bit = num_bits[0]
for it, dit in zip(iters, def_iters):
    for t in range(trials):
        exp_name = f'critlearn_{dataset}_{arch}_{lr_sched}_{dit}_{t}/'
        save_dir = os.path.join(base_save_dir, exp_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        command = (f'CUDA_VISIBLE_DEVICES={gpu} python crit_learn.py --cmd train --exp-name {exp_name} '
                f'--arch {arch} --dataset {dataset} --datadir {datadir} --iters {it} --def-iters {dit} '
                f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                f'--cyclic_num_bits_schedule {num_bits} --cyclic_num_grad_bits_schedule {num_grad_bit} '
                f'--use-wandb --tags {tags} '
        )
        if warm_up:
            command += ' --warm_up'
        if pretrained:
            command += ' --pretrained'
        os.system(command)
