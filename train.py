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
import json
import shutil
from datetime import datetime
from pathlib import Path
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from utils import snapshot_experiment


def _prepare_experiment_dirs(opt):
    resume_cfg = opt.get('resume', {})
    base_root = Path(resume_cfg.get('path', './Conv-TasNet_lightning')).resolve()
    experiments_root = base_root / 'experiments'
    wandb_root = base_root / 'wandb'
    experiments_root.mkdir(parents=True, exist_ok=True)
    wandb_root.mkdir(parents=True, exist_ok=True)

    exp_name = resume_cfg.get('experiment_name') or resume_cfg.get('checkpoint')
    if not exp_name or str(exp_name).strip().lower() in ('', 'none'):
        exp_name = datetime.now().strftime('exp-%Y%m%d-%H%M%S')

    experiment_dir = experiments_root / exp_name
    snapshot_experiment(experiment_dir, opt, Path(__file__).resolve().parent)

    ckpt_dir = experiment_dir / 'checkpoints'
    sample_dir = experiment_dir / 'val_samples'
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir.mkdir(exist_ok=True)

    return base_root, wandb_root, experiment_dir, ckpt_dir, sample_dir


def _resolve_resume_checkpoint(load_from, ckpt_dir):
    if not load_from:
        return None
    load_from = str(load_from).strip()
    if load_from.lower() in ('', 'none'):
        return None

    ckpt_dir = Path(ckpt_dir)
    option = load_from.lower()
    if option in ('latest', 'last'):
        last_path = ckpt_dir / 'last.ckpt'
        if last_path.exists():
            return str(last_path)
        ckpts = sorted(ckpt_dir.glob('*.ckpt'))
        if ckpts:
            return str(ckpts[-1])
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} to resume from.")

    candidate = Path(load_from)
    if not candidate.is_absolute():
        candidate = ckpt_dir / candidate
    candidate = candidate.resolve()
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Checkpoint file not found: {candidate}")


def _record_checkpoint_metadata(checkpoint_cb, ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    meta = {}

    best_path = getattr(checkpoint_cb, 'best_model_path', None)
    if best_path:
        best_path = Path(best_path)
        best_dest = ckpt_dir / 'best.ckpt'
        if best_path.exists():
            if not best_dest.exists() or best_dest.resolve() != best_path.resolve():
                shutil.copy2(best_path, best_dest)
            meta['best'] = str(best_dest)
            score = getattr(checkpoint_cb, 'best_model_score', None)
            if score is not None:
                meta['best_score'] = float(score)
        else:
            meta['best'] = str(best_path)

    last_path = getattr(checkpoint_cb, 'last_model_path', None)
    if last_path:
        last_path = Path(last_path)
        last_dest = ckpt_dir / 'last.ckpt'
        if last_path.exists():
            if not last_dest.exists() or last_dest.resolve() != last_path.resolve():
                shutil.copy2(last_path, last_dest)
            meta['last'] = str(last_dest)
        else:
            meta['last'] = str(last_path)

    if meta:
        with open(ckpt_dir / 'checkpoint_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)


def Train(opt):
    # init Lightning Model
    light = LitModule(**opt['light_conf'])

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
    
    _, wandb_root, experiment_dir, ckpt_dir, sample_dir = _prepare_experiment_dirs(opt)

    wandb_logger = WandbLogger(
        project="setrain-tasnet",
        save_dir=str(wandb_root),
        log_model=False,
        save_code=True,
    )

    light.sample_save_dir = str(sample_dir)
    light.max_val_samples_to_save = opt['light_conf'].get('val_samples_to_save', light.max_val_samples_to_save)

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
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
        default_root_dir=str(experiment_dir),
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=0.3,
        gradient_clip_val=5.0,
        callbacks=callbacks,
        logger=wandb_logger,
        val_check_interval=1.0)

    ckpt_path = _resolve_resume_checkpoint(opt.get('resume', {}).get('load_from'), ckpt_dir)
    trainer.fit(light, ckpt_path=ckpt_path)

    _record_checkpoint_metadata(checkpoint, ckpt_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)
