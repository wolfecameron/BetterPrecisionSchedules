# runs tests with fixed quantization level on the arxiv OGB dataset

import os

gpu = 0
arch = 'gat'
epochs = 800
dpt = 0.6
lr_list = [1e-3, 3e-4]
#lrs_list = ['fixed', 'piecewise', 'anneal_cosine']
lrs_list = ['anneal_cosine']
layers = 2
heads = 8
hid = 256
wd = 5e-4
eval_len = 25
qsched = 'fixed'
nbs = '8 8'
ngbs = '8 8'
trials = 1
qnorm = False
qagg = False
dpt_inp = True
dpt_attn = True

tags = 'gat_tuning'

#for qnorm, qagg in zip(qnorms, qaggs):
for lr in lr_list:
    for lrs in lrs_list:
        for trial in range(trials):
            exp_name = f'ogbarxiv_noquant_normwd_{arch}_{epochs}_{lr}_{lrs}_{trial}'
            command = (
                f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --arch {arch} --exp-name {exp_name} '
                f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
                f'--n-heads {heads} --n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
                f'--precision_schedule {qsched} --cyclic_num_bits_schedule {nbs} '
                f'--cyclic_num_grad_bits_schedule {ngbs} --tags {tags} --use-wandb '
            )
            if qnorm:
                command += f'--quant-norm '
            if qagg:
                command += f'--quant-agg '
            if dpt_inp:
                command += f'--dpt-inp '
            if dpt_attn:
                command += f'--dpt-attn '
            os.system(command)
