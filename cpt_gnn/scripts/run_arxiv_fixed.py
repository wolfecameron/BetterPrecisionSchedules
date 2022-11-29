# runs tests with fixed quantization level on the arxiv OGB dataset

import os

gpu = 0
arch = 'gnn'
epochs = 500
dpt = 0.2
lr = 1e-3
#lrs_list = ['fixed', 'piecewise', 'anneal_cosine']
lrs = 'anneal_cosine'
layers = 2
heads = 8
hid = 256
wd = 5e-4
eval_len = 25
qsched = 'fixed'
nbs = '8 8'
ngbs = '8 8'
trials = 2
qnorm = False
qagg = False
dpt_inp = True
dpt_attn = True
merge = 'cat'

tags = 'gnn_test'
for trial in range(trials):
    exp_name = f'ogbarxiv_test_{t}'
    command = (
        f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --arch {arch} --exp-name {exp_name} '
        f'--n-layers {layers} --n-hidden {hid} --dropout {dpt} --lr {lr} --lr-schedule {lrs} '
        f'--n-heads {heads} --n-epochs {epochs} --weight-decay {wd} --eval_every {eval_len} '
        f'--precision_schedule {qsched} --cyclic_num_bits_schedule {nbs} '
        f'--cyclic_num_grad_bits_schedule {ngbs} --tags {tags} --merge {merge} --use-wandb '
        f'--use-layer-norm --norm-attn '
    )
    if qnorm:
        command += f'--quant-norm '
    if qagg:
        command += f'--quant-agg '
    os.system(command)
