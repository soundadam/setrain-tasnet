import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path
import os
from option import parse
from models.LitGTCRN import LitModule 
from utils.utils import snapshot_experiment, _setup_directories, _is_global_zero

def sweep_train_logic():
    run = wandb.init()
    w_config = run.config
    opt = parse('utils/configs/train_gtcrn.yml', is_train=True)
    
    # 基础设置
    base_channels = 16
    batch_size = 24
    # accumulate_grad_batches = 1
    
    # if scale_mode == "small":
    #     base_channels = 16
    #     accumulate_grad_batches = 1  # 有效 Batch = 48
    # elif scale_mode == "medium":
    #     base_channels = 24
    #     batch_size = 32
    #     accumulate_grad_batches = 2  # 有效 Batch = 64 (或者设为1，有效32)
    # elif scale_mode == "large":
    #     base_channels = 32
    #     batch_size = 16 
    #     accumulate_grad_batches = 3  # 有效 Batch = 16*3 = 48 (保持梯度稳定性)
    
    # 获取其他 Sweep 参数
    kernel_size = w_config.kernel_size
    
    # 构造实验名称 (用于本地目录区分)
    exp_name = f"gtcrnSweep_C{base_channels}_K{kernel_size}_BS{batch_size}"
    
    # ============================================================
    # 4. 注入参数到 opt (覆盖 yaml 配置)
    # ============================================================
    
    # 修改 LightModule 的初始化参数
    # 你的 LitModule 需要在 __init__ 中接收 **kwargs 或者明确写出这些参数
    opt['light_conf']['base_channels'] = base_channels
    opt['light_conf']['kernel_size'] = kernel_size
    opt['light_conf']['use_grouped_rnn'] = False # 固定为 False
    opt['light_conf']['batch_size'] = batch_size
    
    # 5. 目录设置 (复用 train.py 的逻辑)
    exp_dir, ckpt_dir, sample_dir = _setup_directories(exp_name)
    
    # 备份代码 (仅主进程)
    if _is_global_zero():
        snapshot_experiment(exp_dir, opt, Path(__file__).resolve().parent)

    # 6. 初始化模型
    light = LitModule(**opt['light_conf'])
    light.sample_save_dir = str(sample_dir)
    light.max_val_samples_to_save = opt['val'].get('val_samples_to_save', 3)

    # 7. 配置 WandB Logger
    # 关键点：将当前的 run 对象传入，而不是让 Logger 新建一个
    wandb_logger = WandbLogger(
        experiment=run,  # <--- 关键：绑定到当前的 Sweep run
        save_dir=str(exp_dir),
        log_model=False
    )

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
    # 定义搜索空间
    sweep_configuration = {
        'method': 'grid',  # 网格搜索
        'name': 'GTCRN-explore',
        'metric': {
            'goal': 'minimize', 
            'name': 'val_loss'
        },
        'parameters': {
            'kernel_size': {'values': [3, 5, 7]},
            'base_channels': {'values': [16, 24, 32]}
        }
    }
    
    # scale_mode 映射表:
    # small:  Ch=16, BS=48, Acc=1 -> Eff_BS=48
    # medium: Ch=24, BS=32, Acc=2 -> Eff_BS=64
    # large:  Ch=32, BS=16, Acc=3 -> Eff_BS=48

    # 初始化 Sweep Controller
    # 注意：project 名称要和你想要记录的项目一致
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project="GTCRN" 
    )
    
    print(f"Sweep ID: {sweep_id}")
    print("Starting Agent...")
    
    # 启动 Agent
    # count=9 表示总共运行 3*3=9 次实验
    wandb.agent(sweep_id, function=sweep_train_logic, count=9)