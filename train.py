# -*- encoding: utf-8 -*-
'''
@Filename    :train.py
@Time        :2020/07/10 23:23:18
@Author      :Kai Li
@Version     :1.0
'''

from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from LitModule import LitModule 
import torch
import argparse
import os
from pathlib import Path
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from utils import snapshot_experiment
def Train(opt):
    # init Lightning Model
    light = LitModule(**opt['light_conf'])

    # mkdir the file of Experiment path
    os.makedirs(os.path.join(opt['resume']['path'],
                             opt['resume']['checkpoint']), exist_ok=True)
    checkpoint_root = os.path.join(
        opt['resume']['path'], opt['resume']['checkpoint'])

    # Early Stopping
    early_stopping = False
    if opt['train']['early_stop']:
        early_stopping = EarlyStopping(monitor='val_loss', patience=opt['train']['patience'],
                                       mode='min', verbose=1)

    # Don't ask GPU if they are not available.
    if torch.cuda.is_available():
        gpus = len(opt['gpu_ids'])
    else:
        gpus = None
    
    wandb_logger = WandbLogger(
        project="setrain-tasnet",
        save_dir=opt['resume']['path'],
        log_model=False,
    )

    run = wandb_logger.experiment
    run_dir_attr = getattr(run, "dir", None)
    run_dir = run_dir_attr() if callable(run_dir_attr) else run_dir_attr
    if not run_dir:
        run_dir = os.path.join(opt['resume']['path'], f"wandb-run-{wandb_logger.version}")

    snapshot_experiment(run_dir, opt, Path(__file__).resolve().parent)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(run_dir, "val_samples")
    os.makedirs(sample_dir, exist_ok=True)
    light.sample_save_dir = sample_dir
    light.max_val_samples_to_save = opt['light_conf'].get('val_samples_to_save', 3)

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch{epoch:03d}-valloss{val_loss:.4f}",
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=1,
    )
    

    # Build callbacks list
    callbacks = [checkpoint]
    if early_stopping:
        callbacks.append(early_stopping)

    # Trainer setup (Lightning 1.9+/2.x style)
    if torch.cuda.is_available() and gpus and gpus > 0:
        accelerator = 'gpu'
        devices = gpus
    else:
        accelerator = 'cpu'
        devices = 1

    trainer = pl.Trainer(
        max_epochs=opt['train']['epochs'],
        default_root_dir=checkpoint_root,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=0.3,
        gradient_clip_val=5.0,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=1.0)

    trainer.fit(light, ckpt_path=opt['resume']['load_from'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)
