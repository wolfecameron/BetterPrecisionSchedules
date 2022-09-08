import os

# test one
# setup
gpu = 1
datadir = '/home/exx/data/imagenet/'
base_save_dir = './quant_results'
eval_every = 5005
trials = 1

# training
epoch = 90
bs = 256
lr_sched = 'piecewise'
lr = 0.1
step_ratio = 0.1
wd = 1e-4
warm_up = False
pretrained = False

# quant
num_bits = ['4 6']
num_grad_bit = '8 8'

# stuff that changes
ds = 'imagenet'
prec_scheds = ['demon_decay', 'demon_growth', 'exp_decay', 'exp_growth', 'linear_decay', 'linear_growth', 'cos_decay', 'cos_growth']
cycles = [8]

arch = f'resnet34'
for nb in num_bits:
    max_bit = nb[-1]
    for ps in prec_scheds:
        if ps in ['demon_decay', 'exp_decay']:
            for c in cycles:
                for t in range(trials):
                    exp_name = f'{ds}_{arch}_{ps}_quant_max{max_bit}_{c}_{t}_vertical/'
                    save_dir = os.path.join(base_save_dir, exp_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    command = (f'CUDA_VISIBLE_DEVICES={gpu} python train.py --cmd train '
                            f'--arch {arch} --dataset {ds} --datadir {datadir} --epoch {epoch} '
                            f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                            f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                            f'--is_cyclic_precision --num_cyclic_period {c} --precision_schedule {ps} '
                            f'--cyclic_num_bits_schedule {nb} --cyclic_num_grad_bits_schedule {num_grad_bit} '
                            f'--flip-vertically ')
                    if warm_up:
                        command += ' --warm_up'
                    if pretrained:
                        command += ' --pretrained'
                    os.system(command)

                    exp_name = f'{ds}_{arch}_{ps}_quant_max{max_bit}_{c}_{t}_horizontal/'
                    save_dir = os.path.join(base_save_dir, exp_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    command = (f'CUDA_VISIBLE_DEVICES={gpu} python train.py --cmd train '
                            f'--arch {arch} --dataset {ds} --datadir {datadir} --epoch {epoch} '
                            f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                            f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                            f'--is_cyclic_precision --num_cyclic_period {c} --precision_schedule {ps} '
                            f'--cyclic_num_bits_schedule {nb} --cyclic_num_grad_bits_schedule {num_grad_bit} ')
                    if warm_up:
                        command += ' --warm_up'
                    if pretrained:
                        command += ' --pretrained'
                    os.system(command)
        else:
            for c in cycles:
                for t in range(trials):
                    exp_name = f'{ds}_{arch}_{ps}_quant_max{max_bit}_{c}_{t}/'
                    save_dir = os.path.join(base_save_dir, exp_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    command = (f'CUDA_VISIBLE_DEVICES={gpu} python train.py --cmd train '
                            f'--arch {arch} --dataset {ds} --datadir {datadir} --epoch {epoch} '
                            f'--batch_size {bs} --lr_schedule {lr_sched} --lr {lr} --weight_decay {wd} '
                            f'--step_ratio {step_ratio}  --save_folder {save_dir} --eval_every {eval_every} '
                            f'--is_cyclic_precision --num_cyclic_period {c} --precision_schedule {ps} '
                            f'--cyclic_num_bits_schedule {nb} --cyclic_num_grad_bits_schedule {num_grad_bit} ')
                    if warm_up:
                        command += ' --warm_up'
                    if pretrained:
                        command += ' --pretrained'
                    os.system(command)
