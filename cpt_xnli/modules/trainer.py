# subclass the Trainer module from hugging face to enable
# customization to the training loop

import wandb
from torch import nn
from transformers import Trainer
from transformers.trainer import *

from modules.quant_scheds import cyclic_adjust_precision

class QuantTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        
        # track the number of training iters for the quantization schedule
        self._curr_iter = 0

    def training_step(self, model, inputs):
        # set num bits and num grad bits based on current quantization level
        cyclic_adjust_precision(model.args, self._curr_iter, model.args.cyclic_period)
        if (self._curr_iter % 25) == 0:
            wandb.log({
                'train/train_step': self._curr_iter,
                'train/num_bits': model.args.num_bits,
                'train/num_grad_bits': model.args.num_grad_bits,
            })

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
        # keep a counter of iterations for adjusting precision
        self._curr_iter += 1

        return loss.detach()
