import os

gpu = 0
bs = 1024
epochs = 80
hidden = 512
neighbors = 32
dpt = 0.2
lr = 1e-3
lrr = 0.1
wd = 1e-4
trials = 2
lrs = 'anneal_cosine'
tag = 'amazon_fixed_tuning'

ps = 'fixed'
nbs_list = ['6 6', '8 8']
ngbs_list = ['6 6', '8 8']
cycles = 8

for nbs, ngbs in zip(nbs_list, ngbs_list):
    num_bits = nbs[-1]
    for t in range(trials):
        exp_name = f'amazon_fixed_fpagg_{num_bits}bits_{t}'
        command = (
            f'CUDA_VISIBLE_DEVICES={gpu} python train_amazon.py --exp-name {exp_name} '
            f'--n-hidden {hidden} --dropout {dpt} --num-neighbors {neighbors} '
            f'--lr {lr} --lr-schedule {lrs} --lr-ratio {lrr} --n-epochs {epochs} '
            f'--weight-decay {wd} --batch-size {bs} --tags {tag} --precision_schedule {ps} '
            f'--cyclic_num_bits_schedule {nbs} --cyclic_num_grad_bits_schedule {ngbs} '
            f'--num_cyclic_period {cycles} --num_bits 8 --num_grad_bits 8 '
            f'--use-wandb --tags {tag} '
        )
        os.system(command)