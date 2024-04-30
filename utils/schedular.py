import numpy as np
from torch.optim import lr_scheduler
from typing import Dict, Any
import torch

class ExponentialDecayScheduler():
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """
    def __init__(self, warmup_steps, max_steps, lr_final=None, lr_pre_warmup=0, ramp="linear"):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.lr_final = lr_final
        self.lr_pre_warmup = lr_pre_warmup
        self.ramp = ramp

    def get_scheduler(self, optimizer, lr_init: float):
        if self.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.lr_final

        def func(step):
            if step < self.warmup_steps:
                if self.ramp == "cosine":
                    lr = self.lr_pre_warmup + (lr_init - self.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.lr_pre_warmup
                        + (lr_init - self.lr_pre_warmup) * step / self.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
