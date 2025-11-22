# -*- encoding: utf-8 -*-
import torch
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from Loss import Loss
from datasets_related.Datasets import Datasets
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
                 lr=1e-3,
                 scheduler_mode='min',
                 scheduler_factor=0.5,
                 patience=2,
                 train_mix_scp=None,
                 train_ref_scp=None,
                 val_mix_scp=None,
                 val_ref_scp=None,
                 sr=16000,
                 batch_size=16,
                 num_workers=2,
                 chunk_size_in_seconds=10,
                 # 新增参数用于控制保存频率
                 save_interval_epochs=5 
                 ):
        super(LitModule, self).__init__()
        self.save_hyperparameters()

        # ... (参数初始化保持不变) ...
        self.train_mix_scp = train_mix_scp
        self.train_ref_scp = train_ref_scp
        self.val_mix_scp = val_mix_scp
        self.val_ref_scp = val_ref_scp
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size_in_seconds = chunk_size_in_seconds
        
        self.learning_rate = lr
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.patience = patience
        
        self.convtasnet = ConvTasNet(N, L, B, H, P, X, R, norm, num_spks, activate)
        self.loss_fn = Loss()

        # --- 音频保存相关配置 ---
        self.sample_save_dir = None
        self.max_val_samples_to_save = 3
        self.save_interval_epochs = save_interval_epochs # 每多少个 epoch 保存一次
        
        # 用于在 validation_step 和 on_validation_epoch_end 之间传递数据
        self.validation_vis_batch = None 

    def forward(self, x):
        return self.convtasnet(x)

    def on_train_start(self):
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                self.logger.experiment.define_metric("train_loss", step_metric="epoch")
                self.logger.experiment.define_metric("val_loss", step_metric="epoch")
                self.logger.experiment.define_metric("SI-SNR", step_metric="epoch")
            except AttributeError:
                pass

    def training_step(self, batch, batch_idx):
        mix = batch['mix']
        refs = batch['ref']
        ests = self.forward(mix)
        loss = self.loss_fn.compute_loss(ests, refs)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mix = batch['mix']
        refs = batch['ref']
        ests = self.forward(mix)
        
        loss = self.loss_fn.compute_loss(ests, refs)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('SI-SNR', -loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # --- 逻辑修改部分 ---
        # 判断当前 epoch 是否需要保存，且只缓存第一个 batch 的数据
        # (current_epoch 是从 0 开始的，所以 +1)
        should_save = (self.current_epoch + 1) % self.save_interval_epochs == 0
        
        if should_save and batch_idx == 0:
            # 将 tensor 转移到 CPU 并 detach，避免占用显存
            # 处理 ref/est 可能是 list 的情况 (多说话人)
            refs_cpu = [r.detach().cpu() for r in refs] if isinstance(refs, (list, tuple)) else refs.detach().cpu()
            ests_cpu = [e.detach().cpu() for e in ests] if isinstance(ests, (list, tuple)) else ests.detach().cpu()
            
            self.validation_vis_batch = {
                'mix': mix.detach().cpu(),
                'refs': refs_cpu,
                'ests': ests_cpu
            }
            
        return loss

    def on_validation_epoch_end(self):
        # 1. 检查是否有缓存的数据
        if self.validation_vis_batch is None:
            return

        # 2. 检查是否是主进程 (防止多卡重复写入)
        if self.trainer.world_size > 1 and not self.trainer.is_global_zero:
            self.validation_vis_batch = None # 清空
            return
            
        # 3. 检查保存路径
        if self.sample_save_dir is None:
            self.validation_vis_batch = None
            return

        # 4. 提取数据
        data = self.validation_vis_batch
        noisy = data['mix']
        refs = data['refs']
        ests = data['ests']

        # 取出第一个说话人用于可视化 (根据你的 utils 逻辑调整)
        clean = refs[0] if isinstance(refs, (list, tuple)) else refs
        enhanced = ests[0] if isinstance(ests, (list, tuple)) else ests
        
        # 5. 调用保存函数
        # 这里 max_samples 直接用你设定的全局数量
        try:
            save_validation_samples(
                sample_dir=self.sample_save_dir,
                noisy=noisy,
                clean=clean,
                enhanced=enhanced,
                sample_rate=self.sample_rate,
                epoch=self.current_epoch + 1,
                max_samples=self.max_val_samples_to_save,
                start_index=0 # 每次都从0开始命名，或者加上 epoch 后缀区分
            )
            print(f"Saved validation samples at epoch {self.current_epoch + 1}")
        except Exception as e:
            print(f"Failed to save validation samples: {e}")
        
        # 6. 重要：清空缓存，释放内存，并防止下个 epoch 误保存
        self.validation_vis_batch = None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=self.scheduler_mode, factor=self.scheduler_factor,
            patience=self.patience, min_lr=1e-6
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

    def train_dataloader(self):
        if not self.train_mix_scp: raise ValueError("train_mix_scp missing")
        dataset = Datasets(self.train_mix_scp, self.train_ref_scp, sr=self.sample_rate, 
                           chunk_size_in_seconds=self.chunk_size_in_seconds)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        if not self.val_mix_scp: raise ValueError("val_mix_scp missing")
        dataset = Datasets(self.val_mix_scp, self.val_ref_scp, sr=self.sample_rate, chunk_size_in_seconds=None)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=False, drop_last=True, pin_memory=True)