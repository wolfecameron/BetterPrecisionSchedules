# runs tests with fixed quantization level on the arxiv OGB dataset

import os

gpu = 0
epochs = 500
dpt = 0.5
lr_list = [1e-3]
#lrs_list = ['fixed', 'piecewise', 'anneal_cosine']
lrs_list = ['anneal_cosine']
layers = 2
hid = 512
wd = 5e-4
eval_len = 25
qsched = 'fixed'
nbs = '8 8'
ngbs = '8 8'
trials = 1
tags = 'axiv_static_quant1'

for lr in lr_list:
    for lrs in lrs_list:
        for trial in range(trials):
            exp_name = f'ogbarxiv_static_{lr}_{lrs}_{trial}_smallmod_highdpt'
            command = (
                f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --exp-name {exp_name} '
                f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
                f'--n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
                f'--precision_schedule {qsched} --cyclic_num_bits_schedule {nbs} '
                f'--cyclic_num_grad_bits_schedule {ngbs} --tags {tags} --use-wandb '
            )
            os.system(command)
