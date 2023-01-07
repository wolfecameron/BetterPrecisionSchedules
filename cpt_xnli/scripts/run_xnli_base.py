import os

# machine settings
gpu = 0
batch_size = 64
seq_len = 128

# model settings
model = 'bert-base-multilingual-cased'
lang = 'de'
train_lang = 'de'

# hparams
lr = 5e-5
epochs = 2.0

# exp settings
trials = 2

# cpt settings
ps = 'fixed'
nbs = '8 8'
ngbs = '8 8'
n = 2

# run trials
exp_name = 'xnli_ft_{nbs[0]}bits'
for t in range(trials):
    full_exp_name = f'{exp_name}_{t}'
    output_dir = f'/data/crw13_data/xnli_results/{full_exp_name}'
    command = (
        f'CUDA_VISIBLE_DEVICES={gpu} python train_xnli.py --language {lang} '
        f'--train_language {train_lang} --model_name_or_path {model} --do_train '
        f'--do_eval --per_device_train_batch_size {batch_size} --learning_rate {lr} '
        f'--num_train_epochs {epochs} --max_seq_length {seq_len} --save_steps -1 '
        f'--output_dir {output_dir} --overwrite_output_dir --precision_schedule {ps} '
        f'--num_bits_min {nbs[0]} --num_bits_max {nbs[-1]} --num_grad_bits_min {ngbs[0]} '
        f'--num_grad_bits_max {ngbs[-1]} --num_cyclic_period {n} --report_to wandb '
        f'--run_name {full_exp_name} ' 
    )
    os.system(command)
