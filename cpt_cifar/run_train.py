import os

# test one
# setup
gpu = 0
arch = 'cifar10_resnet_74'
ds = 'cifar10'
datadir = '/home/exx/data/'
base_save_dir = './quant_results'
eval_every = 390
trials = 3

# training
iters = 64000
bs = 128
lr_sched = 'piecewise'
lr = 0.1
step_ratio = 0.1
wd = 1e-4
warm_up = False
pretrained = False

# quant
num_bit = '3 8'
num_grad_bit = '8 8'

# stuff that changes
prec_scheds = ['demon_decay', 'demon_growth', 'exp_growth', 'exp_decay', 'linear_growth', 'linear_decay', 'cos_growth', 'cos_decay']
cycles = [1]
exp_nums = ['06']

for ps in prec_scheds:
    for c, global_exp_num in zip(cycles, exp_nums):
        for t in range(trials):
            exp_name = f'{ds}_{ps}_quant_{c}_{global_exp_num}_{t}/'
            save_dir = os.path.join(base_save_dir, exp_name)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            command = (f'CUDA_VISIBLE_DEVICES={gpu} python train.py --cmd train '
                    f'--arch {arch} --dataset {ds} --datadir {datadir} --iters {iters} '
                    f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                    f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                    f'--is_cyclic_precision --num_cyclic_period {c} --precision_schedule {ps} '
                    f'--cyclic_num_bits_schedule {num_bit} --cyclic_num_grad_bits_schedule {num_grad_bit} ')
            if warm_up:
                command += ' --warm_up'
            if pretrained:
                command += ' --pretrained'
            os.system(command)
