import wandb

from transformers.integrations import TrainerCallback
from modules.quant_scheds import cyclic_adjust_precision


class QuantCallback(TrainerCallback):

    def __init__(self):
        #self._iters = 0 # track number of training iters that have occurred
        self._cyclic_period = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._cyclic_period = (args.total_iters // args.num_cyclic_period) + 1

    def on_step_begin(self, args, state, control, **kwargs):
        cyclic_adjust_precision(args, state.global_step, self._cyclic_period)
        #self._iters += 1
