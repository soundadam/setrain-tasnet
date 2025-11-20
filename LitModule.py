# -*- encoding: utf-8 -*-
'''
@Filename    :lightning.py
@Time        :2020/07/10 20:27:23
@Author      :Kai Li
@Version     :1.0
'''

import os
import torch
from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets_related.Datasets import Datasets
from pytorch_lightning import LightningModule
from model import ConvTasNet
from utils import save_validation_samples


class LitModule(LightningModule):
    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gLN",
                 num_spks=2,
                 activate="relu",
                 causal=False,
                 # optimizer
                 lr=1e-3,
                 # scheduler
                 scheduler_mode='min',
                 scheduler_factor=0.5,
                 patience=2,
                 # Dataset
                 train_mix_scp='/home/likai/data1/create_scp/tr_mix.scp',
                 train_ref_scp=[
                     '/home/likai/data1/create_scp/tr_s1.scp',
                     '/home/likai/data1/create_scp/tr_s2.scp'
                 ],
                 val_mix_scp='/home/likai/data1/create_scp/cv_mix.scp',
                 val_ref_scp=[
                     '/home/likai/data1/create_scp/cv_s1.scp',
                     '/home/likai/data1/create_scp/cv_s2.scp'
                 ],
                 sr=16000,
                 # DataLoader
                 batch_size=16,
                 num_workers=2,
                 chunk_size_in_seconds=10
                 ):
        super(LitModule, self).__init__()
        # ------------------Dataset&DataLoader Parameter-----------------
        self.train_mix_scp = train_mix_scp
        self.train_ref_scp = train_ref_scp
        self.val_mix_scp = val_mix_scp
        self.val_ref_scp = val_ref_scp
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size_in_seconds = chunk_size_in_seconds
        # ----------training&validation&testing Param---------
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        # -----------------------model-----------------------
        self.convtasnet = ConvTasNet(
            N, L, B, H, P, X, R, norm, num_spks, activate)
        self.sample_save_dir = None
        self.max_val_samples_to_save = 3
        self._saved_val_samples = 0
        self._last_sample_epoch = -1

    def forward(self, x):
        return self.convtasnet(x)

    # ---------------------
    # TRAINING STEP
    # ---------------------

    def training_step(self, batch, batch_idx):
        mix = batch['mix']
        refs = batch['ref']
        ests = self.forward(mix)
        ls_fn = Loss()
        loss = ls_fn.compute_loss(ests, refs)
        # log training loss for progress bar and epoch metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        opt = self.optimizers()           # returns the first optimizer if you have one
        lr = opt.param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True)
        
        # self.log('learning_rate', self.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        return loss

    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        mix = batch['mix']
        refs = batch['ref']
        ests = self.forward(mix)
        ls_fn = Loss()
        loss = ls_fn.compute_loss(ests, refs)
        # log validation loss aggregated over the epoch
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._maybe_save_samples(mix, refs, ests)
        return loss
    
    # in modern Lightning, per-epoch aggregation is handled via self.log(on_epoch=True)

    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.scheduler_mode,
            factor=self.scheduler_factor,
            patience=self.patience,
            # verbose=True,
            min_lr=1e-6,
        )
        # Use dict style to specify the monitored metric
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def train_dataloader(self):
        dataset = Datasets(self.train_mix_scp,
                           self.train_ref_scp, sr=self.sample_rate, chunk_size_in_seconds=self.chunk_size_in_seconds)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        dataset = Datasets(self.val_mix_scp,
                           self.val_ref_scp, sr=self.sample_rate, chunk_size_in_seconds=None)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)
    
    def _maybe_save_samples(self, noisy, refs, ests):
        if self.sample_save_dir is None or self.max_val_samples_to_save <= 0:
            return

        if self._last_sample_epoch != self.current_epoch:
            self._last_sample_epoch = self.current_epoch
            self._saved_val_samples = 0

        remaining = self.max_val_samples_to_save - self._saved_val_samples
        if remaining <= 0:
            return

        clean = refs[0] if isinstance(refs, (list, tuple)) else refs
        enhanced = ests[0] if isinstance(ests, (list, tuple)) else ests

        saved = save_validation_samples(
            sample_dir=self.sample_save_dir,
            noisy=noisy,
            clean=clean,
            enhanced=enhanced,
            sample_rate=self.sample_rate,
            epoch=self.current_epoch + 1,
            max_samples=remaining,
            start_index=self._saved_val_samples,
        )
        self._saved_val_samples += saved
