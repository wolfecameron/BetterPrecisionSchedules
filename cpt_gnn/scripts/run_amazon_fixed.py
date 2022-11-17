import os

gpu = 1
bs = 1024
epochs = 50
hidden = 512
neighbors = 4
dropouts = [0.2, 0.5]
lr_list = [1e-3, 1e-4]
lrr_list = [0.1, 0.001]
wds = [1e-4, 0.0]
trials = 1
lrs = 'anneal_cosine'
tag = 'amazon_baseline_tuning'

for lr in lr_list:
   for lrr in lrr_list:
       for wd in wds:
           for dpt in dropouts:
               for t in range(trials):
                   exp_name = f'amazon_base_{lr}_{lrr}_{wd}_{dpt}_{t}'
                   command = (
                       f'CUDA_VISIBLE_DEVICES={gpu} python train_amazon.py --exp-name {exp_name} '
                       f'--n-hidden {hidden} --dropout {dpt} --num-neighbors {neighbors} '
                       f'--lr {lr} --lr-schedule {lrs} --lr-ratio {lrr} --n-epochs {epochs} '
                       f'--weight-decay {wd} --batch-size {bs} --tags {tag} --use-wandb '
                   )
                   os.system(command)
