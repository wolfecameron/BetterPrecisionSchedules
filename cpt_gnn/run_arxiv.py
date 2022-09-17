import os

gpu = 0
epochs = 1000
dpt = 0.2
lr = 1e-3
layers = 2
hid = 1024
wd = 5e-4

command = (
    f'CUDA_VISIBLE_DEVICES={gpu} python train_arxiv.py --n-layers {layers} '
    f'--n-hidden {hid} --dropout {dpt} --lr {lr} --n-epochs {epochs} '
    f'--weight-decay {wd} --verbose ')
os.system(command)
