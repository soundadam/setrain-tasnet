from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from models.LitGTCRN import LitModule 
import torch
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.utils import snapshot_experiment, _get_experiment_name, _is_global_zero, _setup_directories



def Train(opt):
    # 1. 确定实验名称和目录
    # 如果是 Resume 模式，这里可能需要逻辑调整去指向旧目录，
    # 但通常 Resume 只是加载权重，产生新 Log，这里保持生成新目录的逻辑比较清晰。
    exp_name = _get_experiment_name(opt)
    exp_dir, ckpt_dir, sample_dir = _setup_directories(exp_name)

    # 2. 代码备份 (仅主进程)
    if _is_global_zero():
        # 假设 snapshot_experiment 接受 (目标目录, 配置, 代码根目录)
        snapshot_experiment(exp_dir, opt, Path(__file__).resolve().parent)

    # 3. 初始化 Lightning Model
    light = LitModule(**opt['light_conf'])
    light.sample_save_dir = str(sample_dir)
    # 可以在 yaml 中配置 val_samples_to_save
    light.max_val_samples_to_save = opt['val'].get('val_samples_to_save', 3)

    # 4. 配置 WandB
    # save_dir 指定为 exp_dir，这样 wandb 文件夹会生成在 exp_local/实验名/wandb 下，不污染根目录
    wandb_logger = WandbLogger(
        name=exp_name,
        project="GTCRN",
        save_dir=str(exp_dir), 
        log_model=True,    # 关键：不上传模型到云端
        save_code=True,    # 关键：不上传代码到云端 (本地已经 snapshot 了)
        offline=False       # 如果想完全断网跑，设为 True
    )

    # 5. 配置 Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}_valloss{val_loss:.4f}",
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True, # 自动保存 last.ckpt，替代了原代码中复杂的元数据逻辑
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

    # 6. Trainer 设置
    # 计算 GPU 数量
    devices = len(opt['gpu_ids']) if torch.cuda.is_available() else 1
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    trainer = pl.Trainer(
        max_epochs=opt['train']['epochs'],
        default_root_dir=str(exp_dir), # PL 的默认日志目录
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=opt['train'].get('limit_train_batches', 1.0), # 建议从 config 读，默认 1.0
        gradient_clip_val=5.0,
        callbacks=callbacks,
        logger=wandb_logger,
        check_val_every_n_epoch=opt['val'].get('check_val_every_n_epoch', 1)
        # strategy='ddp_find_unused_parameters_true' # 如果是多卡可能需要这个
    )

    # 7. 处理 Resume (如果有)
    ckpt_path = opt.get('resume', {}).get('load_from')
    if ckpt_path and str(ckpt_path).lower() != 'none':
        if not Path(ckpt_path).is_absolute():
            print(f"Warning: Resume path {ckpt_path} is relative, make sure it is correct.")
        print(f"Resuming from: {ckpt_path}")
    else:
        ckpt_path = None

    # 8. 开始训练
    trainer.fit(light, ckpt_path=ckpt_path)
    print("Training finished. Starting testing with the best checkpoint...")
    
    # 'best' 会自动加载 checkpoint_callback 保存的 val_loss 最小的那个模型
    # 这一步会自动调用 model.test_step
    trainer.test(light, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)