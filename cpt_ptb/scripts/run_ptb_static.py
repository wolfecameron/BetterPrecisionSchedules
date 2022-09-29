# run lstm training on ptb with static quantization

import os

gpu = 0 
data_path = '/home/exx/data/ptb'
emsize = 800
nhid = 800
lr = 20
clip = 0.25
epochs = 40
bs = 20
seq_len = 35
dpt = 0.5

exp_name = 'lstm_ptb_static_00'
tags = 'lstm_ptb_static0'
command = (
    f'CUDA_VISIBLE_DEVICES={gpu} python train_ptb.py --exp-name {exp_name} '
    f'--data {data_path} --emsize {emsize} --nhid {nhid} --lr {lr} --clip {clip} '
    f'--epochs {epochs} --batch_size {bs} --bptt {seq_len} --dropout {dpt} '
    f'--precision_schedule fixed --cyclic_num_bits_schedule 8 8 '
    f'--cyclic_num_grad_bits_schedule 8 8 --tags {tags} --use-wandb'
)
os.system(command)


