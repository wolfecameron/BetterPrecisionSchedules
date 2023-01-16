import wandb
from transformers.integrations import TrainerCallback

class CustWandbCallback(TrainerCallback):
    """adds some custom metrics into wandb logger"""

    def on_train_begin(self, args, state, control, **kwargs):
        wandb.run.define_metric(name='train/train_step')
        wandb.run.define_metric(name='train/num_bits', step_metric='train/train_step')
        wandb.run.define_metric(name='train/num_grad_bits', step_metric='train/train_step')
