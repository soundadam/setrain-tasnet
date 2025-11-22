from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from LitModule import LitModule 
import torch
import argparse
import os
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils import snapshot_experiment

def get_experiment_name(opt):
    """生成或获取实验名称"""
    # 优先使用配置文件中的名称，如果没有则使用默认前缀+时间戳
    exp_name = opt.get('name', 'tasnet_experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{exp_name}_{timestamp}"

def setup_directories(exp_name, root_dir='exp_local'):
    """创建清晰的本地目录结构"""
    # 结构: exp_local/实验名/ {checkpoints, val_samples, src, wandb_logs}
    exp_dir = Path(root_dir) / exp_name
    ckpt_dir = exp_dir / 'checkpoints'
    sample_dir = exp_dir / 'val_samples'
    
    # 仅主进程创建目录
    if _is_global_zero():
        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        sample_dir.mkdir(exist_ok=True)
    
    return exp_dir, ckpt_dir, sample_dir

def _is_global_zero():
    # 简单的 rank check 逻辑
    rank = int(os.environ.get('RANK', 0))
    return rank == 0

def Train(opt):
    # 1. 确定实验名称和目录
    # 如果是 Resume 模式，这里可能需要逻辑调整去指向旧目录，
    # 但通常 Resume 只是加载权重，产生新 Log，这里保持生成新目录的逻辑比较清晰。
    exp_name = get_experiment_name(opt)
    exp_dir, ckpt_dir, sample_dir = setup_directories(exp_name)

    # 2. 代码备份 (仅主进程)
    if _is_global_zero():
        # 假设 snapshot_experiment 接受 (目标目录, 配置, 代码根目录)
        snapshot_experiment(exp_dir, opt, Path(__file__).resolve().parent)

    # 3. 初始化 Lightning Model
    light = LitModule(**opt['light_conf'])
    light.sample_save_dir = str(sample_dir)
    # 可以在 yaml 中配置 val_samples_to_save
    light.max_val_samples_to_save = opt['light_conf'].get('val_samples_to_save', 3)

    # 4. 配置 WandB
    # save_dir 指定为 exp_dir，这样 wandb 文件夹会生成在 exp_local/实验名/wandb 下，不污染根目录
    wandb_logger = WandbLogger(
        name=exp_name,
        project="setrain-tasnet",
        save_dir=str(exp_dir), 
        log_model=False,    # 关键：不上传模型到云端
        save_code=False,    # 关键：不上传代码到云端 (本地已经 snapshot 了)
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
        val_check_interval=1.0,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)