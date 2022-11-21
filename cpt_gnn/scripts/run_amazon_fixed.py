import os

gpu = 1
bs = 1024
epochs = 80
hidden = 512
neighbors_list = [64, 4]
dpt = 0.2
lr = 1e-3
lrr = 0.1
wd = 1e-4
trials = 1
lrs = 'anneal_cosine'
tag = 'amazon_baseline_tuning'

for neighbors in neighbors_list:
   for t in range(trials):
       exp_name = f'amazon_base_{lr}_{lrr}_{wd}_{dpt}_neigh{neighbors}_{t}'
       command = (
           f'CUDA_VISIBLE_DEVICES={gpu} python train_amazon.py --exp-name {exp_name} '
           f'--n-hidden {hidden} --dropout {dpt} --num-neighbors {neighbors} '
           f'--lr {lr} --lr-schedule {lrs} --lr-ratio {lrr} --n-epochs {epochs} '
           f'--weight-decay {wd} --batch-size {bs} --tags {tag} --use-wandb '
       )
       os.system(command)
