import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path
import os
from option import parse
from models.LitGTCRN import LitModule 
from utils.utils import snapshot_experiment, _setup_directories, _is_global_zero, _get_experiment_name, _add_timestamp
import functools
def sweep_train_logic(opt):
    torch.set_float32_matmul_precision('medium')
    run = wandb.init(config=opt)
    lightconf = opt['light_conf']
    lightconf['model_arch'] = run.config.model_arch

    # lightconf['base_channels'] = run.config.base_channels
    # lightconf['kernel_size'] = run.config.kernel_size
    # lightconf['batch_size'] = run.config.batch_size
    # opt = parse(baseconfigpath, is_train=True)
    
    expname = 'gtcrnSweep' + f"_C{lightconf['base_channels']}_K{lightconf['kernel_size']}_BS{lightconf['batch_size']}"
    expname = _add_timestamp(expname)
    exp_dir, ckpt_dir, sample_dir = _setup_directories(expname)
    
    # 备份代码 (仅主进程)
    if _is_global_zero():
        snapshot_experiment(exp_dir, opt, Path(__file__).resolve().parent)

    # 6. 初始化模型
    light = LitModule(**lightconf)
    light.sample_save_dir = str(sample_dir)
    light.max_val_samples_to_save = opt['val'].get('val_samples_to_save', 3)

    wandb_logger = WandbLogger(
        experiment=run,  # <--- 关键：绑定到当前的 Sweep run
        name = expname,
        save_dir=str(exp_dir),
        log_model=False,
        save_code=True
    )
    # TODO artifact like opt not fully saved

    # 8. Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}_valloss{val_loss:.4f}",
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True,
        )
    ]
    
    if opt['train'].get('early_stop'):
        callbacks.append(EarlyStopping(
            monitor='val_loss', 
            patience=opt['train']['patience'],
            mode='min', 
            verbose=True
        ))

    # 9. Trainer 设置
    devices = len(opt['gpu_ids']) if torch.cuda.is_available() else 1
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        max_epochs=opt['train']['epochs'],
        default_root_dir=str(exp_dir),
        accelerator=accelerator,
        devices=devices,
        strategy='ddp' if devices > 1 else 'auto', # 显式指定 DDP 以利用多卡
        limit_train_batches=opt['train'].get('limit_train_batches', 1.0),
        gradient_clip_val=5.0,
        callbacks=callbacks,
        logger=wandb_logger,
        check_val_every_n_epoch=opt['val'].get('check_val_every_n_epoch', 1))
        # 动态梯度累积：保证在不同 batch_size 下优化方向的一致性
        # accumulate_grad_batches=accumulate_grad_batches
    # )

    # 10. 开始训练
    trainer.fit(light)
    
    # 11. 测试 (Sweep 中通常看 val_loss 就够了，但如果你想看最终指标也可以加)
    trainer.test(light, ckpt_path="best")

# ============================================================
# Sweep Configuration
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--configpath', type=str, default='utils/configs/train_gtcrn.yml', help='Path to option YAML file.')
    args = parser.parse_args()
    config = parse(args.configpath, is_train=True)

    sweep_configuration = {
        'method': 'grid',  # 网格搜索
        'name': 'GTCRN-deeper',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_loss'
        },
        'parameters': {
            'model_arch': {'values': ['v2', 'v3']}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project="GTCRN" 
    )
    
    print(f"Sweep ID: {sweep_id}")
    print("Starting Agent...")
    wandb.agent(sweep_id, function=functools.partial(sweep_train_logic, opt=config), count=3)