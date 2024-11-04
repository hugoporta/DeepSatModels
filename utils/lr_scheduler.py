# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
# from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def build_scheduler(config, optimizer, n_iter_per_epoch):
    t_initial = int(config['SOLVER']['num_epochs'] * n_iter_per_epoch / config['SOLVER']['num_cycles'])
    warmup_steps = int(config['SOLVER']['num_warmup_epochs'] * n_iter_per_epoch)
    # decay_steps = int(config['SOLVER']['decay_epochs'] * n_iter_per_epoch)

    lr_scheduler = None
    if config['SOLVER']['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            # t_mul=1.,
            lr_min=float(config['SOLVER']['lr_min']),
            warmup_lr_init=float(config['SOLVER']['lr_start']),
            warmup_t=warmup_steps,
            cycle_limit=int(config['SOLVER']['num_cycles']),
            t_in_epochs=False,
        )

    return lr_scheduler

"""
def build_scheduler_light(config, optimizer, n_iter_per_epoch, interval):

    if interval == "step":
        warmup_epochs = config['SOLVER']['num_warmup_epochs'] * n_iter_per_epoch
        max_epochs = config['SOLVER']['num_epochs'] * n_iter_per_epoch
    else:
        warmup_epochs = config['SOLVER']['num_warmup_epochs']
        max_epochs = config['SOLVER']['num_epochs']

    lr_scheduler = None
    if config['SOLVER']['lr_scheduler'] == 'cosine':
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=float(config['SOLVER']['lr_start']),
            max_epochs=max_epochs,
            eta_min=float(config['SOLVER']['lr_min']),
            last_epoch=-1,
        )

    return lr_scheduler"""

def build_scheduler_pytorch(config, optimizer, n_iter_per_epoch, interval):
    if interval == "step":
        warmup_epochs = config['SOLVER']['num_warmup_epochs'] * n_iter_per_epoch
        max_epochs = config['SOLVER']['num_epochs'] * n_iter_per_epoch
    else:
        warmup_epochs = config['SOLVER']['num_warmup_epochs']
        max_epochs = config['SOLVER']['num_epochs']

    start_factor = config['SOLVER']['lr_start'] / config["SOLVER"]["lr_base"]

    # Warm-up phase: linear increase in learning rate
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs)

    # Cosine annealing phase
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=config['SOLVER']['lr_min'])

    # Combine the two schedulers using SequentialLR
    lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None