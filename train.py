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
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from utils import snapshot_experiment
def Train(opt):
    # init Lightning Model
    light = LitModule(**opt['light_conf'])

    experiments_root = os.path.join(opt['resume']['path'], 'experiments')
    os.makedirs(experiments_root, exist_ok=True)

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
        save_dir=experiments_root,
        log_model=False,
    )

    experiment_name = opt['resume'].get('experiment_name') or opt['resume'].get('checkpoint')
    if not experiment_name or str(experiment_name).lower() == 'auto':
        experiment_name = getattr(wandb_logger.experiment, 'name', None)
    if not experiment_name:
        experiment_name = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    experiment_dir = os.path.join(experiments_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    snapshot_experiment(experiment_dir, opt, Path(__file__).resolve().parent)

    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = os.path.join(experiment_dir, "val_samples")
    os.makedirs(sample_dir, exist_ok=True)
    light.sample_save_dir = sample_dir

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
        default_root_dir=experiment_dir,
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=0.3,
        gradient_clip_val=5.0,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=1.0)

    ckpt_path = opt['resume'].get('load_from')
    if isinstance(ckpt_path, str) and ckpt_path.lower() == 'none':
        ckpt_path = None
    trainer.fit(light, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)
