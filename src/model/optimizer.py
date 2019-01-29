import numpy as np
import torch


class Tacotron2Optimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, max_lr=1e-3, min_lr=1e-5, warmup_steps=50000, k=0.01):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.k = k
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        if self.step_num > self.warmup_steps:
            lr = self.max_lr * np.exp(-1.0 * self.k * self.step_num)
            if lr >= self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()
