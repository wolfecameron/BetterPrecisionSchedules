import os

# test one
# setup
gpu = 1
arch = 'cifar10_resnet_74'
ds = 'cifar10'
datadir = '/home/exx/data/'
base_save_dir = './quant_results'
eval_every = 390
trials = 1

# training
iter_list = [64000, 128000, 256000]
bs = 128
lr_sched = 'piecewise'
lr = 0.1
step_ratio = 0.1
wd = 1e-4
warm_up = False
pretrained = False

# quant
fwbs = ['8 8']
bwbs = ['8 8']

# stuff that changes
ps = 'fixed'
tags = 'static_cifar'

for iters in iter_list:
    for fwb, bwb in zip(fwbs, bwbs):
        for t in range(trials):
            exp_name = f'fixlr_static_{ds}_{iters}iters_fw{fwb[0]}_bw{bwb[0]}_{t}/'
            save_dir = os.path.join(base_save_dir, exp_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            command = (
                    f'CUDA_VISIBLE_DEVICES={gpu} python train.py --cmd train --exp-name {exp_name} '
                    f'--arch {arch} --dataset {ds} --datadir {datadir} --iters {iters} '
                    f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                    f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                    f'--is_cyclic_precision --precision_schedule {ps} '
                    f'--cyclic_num_bits_schedule {fwb} --cyclic_num_grad_bits_schedule {bwb} '
                    f'--use-wandb --tags {tags} ')
            if warm_up:
                command += ' --warm_up'
            if pretrained:
                command += ' --pretrained'
            os.system(command)
