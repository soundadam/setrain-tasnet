# -*- encoding: utf-8 -*-
import torch
from torch import optim
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import multiprocessing
# from Loss import Loss
from utils.loss_factory import HybridLoss as Loss
from datasets_related.Datasets import Datasets
from models.gtcrn import GTCRN
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore
from utils.utils import save_validation_samples
from utils.scheduler import LinearWarmupCosineAnnealingLR

class LitModule(LightningModule):
    def __init__(self,
                 lr=1e-3,
                 train_mix_scp=None,
                 train_ref_scp=None,
                 val_mix_scp=None,
                 val_ref_scp=None,
                 test_mix_scp=None,
                 test_ref_scp=None,
                 sr=16000,
                 batch_size=16,
                 num_workers=2,
                 chunk_size_in_seconds=10,
                 save_interval_epochs=25,
                 # --- Model Hyperparameters ---
                 base_channels=16,
                 kernel_size=3,  # Passed as int, converted to tuple later
                 use_grouped_rnn=False
                 ):
        super(LitModule, self).__init__()
        self.save_hyperparameters()

        # Paths
        self.train_mix_scp = train_mix_scp
        self.train_ref_scp = train_ref_scp
        self.val_mix_scp = val_mix_scp
        self.val_ref_scp = val_ref_scp
        self.test_mix_scp = test_mix_scp if test_mix_scp else val_mix_scp
        self.test_ref_scp = test_ref_scp if test_ref_scp else val_ref_scp
        
        self.sample_rate = sr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size_in_seconds = chunk_size_in_seconds
        self.learning_rate = lr
        
        # --- Model Initialization with Config ---
        # Convert kernel_size int to tuple if necessary, e.g. 3 -> (3,3)
        k_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        
        self.model = GTCRN(
            base_channels=base_channels,
            kernel_size=k_size,
            use_grouped_rnn=use_grouped_rnn
        )
        
        self.loss_fn = Loss()
        
        self.test_pesq = None 
        self.test_dnsmos = None 
        # Helpers
        self.max_val_samples_to_save = 3
        self.save_interval_epochs = save_interval_epochs
        self.validation_vis_batch = None 

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self.logger and hasattr(self.logger, 'experiment'):
            try:
                self.logger.experiment.define_metric("train_loss", step_metric="epoch")
                self.logger.experiment.define_metric("val_loss", step_metric="epoch")
                self.logger.experiment.define_metric("SI-SNR", step_metric="epoch")
                self.logger.experiment.define_metric("lr-Adam", step_metric="epoch") 
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
        loss, sisnr = self.loss_fn.compute_loss(ests, refs, need_sisnr=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('SI-SNR', sisnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # --- 逻辑修改部分 ---
        # 判断当前 epoch 是否需要保存，且只缓存第一个 batch 的数据
        # (current_epoch 是从 0 开始的，所以 +1)
        should_save = (self.current_epoch + 1) % self.save_interval_epochs == 0
        if should_save and batch_idx == 0:
            # 将 tensor 转移到 CPU 并 detach，避免占用显存
            # 处理 ref/est 可能是 list 的情况 (多说话人)
            # TODO 可以直接默认单个人
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
            self.validation_vis_batch = None
            return
            
        # 3. 检查保存路径
        if self.sample_save_dir is None:
            self.validation_vis_batch = None
            return

        data = self.validation_vis_batch
        try:
            save_validation_samples(
                sample_dir=self.sample_save_dir,
                noisy=data['mix'],
                clean=data['refs'][0] if isinstance(data['refs'], list) else data['refs'],
                enhanced=data['ests'][0] if isinstance(data['ests'], list) else data['ests'],
                sample_rate=self.sample_rate,
                epoch=self.current_epoch + 1,
                max_samples=self.max_val_samples_to_save
            )
        except Exception as e:
            print(f"Failed save samples: {e}")
        self.validation_vis_batch = None
    def on_test_start(self):
        print("Initializing Test Metrics (DNSMOS & PESQ)...")
        
        # 1. PESQ (CPU bound, 轻量)
        if self.test_pesq is None:
            self.test_pesq = PerceptualEvaluationSpeechQuality(
                fs=self.sample_rate, 
                mode='wb'
            )
            # PESQ 不需要 .to(device)，因为它只能在 CPU 跑

        # 2. DNSMOS (GPU bound, 重量级)
        if self.test_dnsmos is None:
            self.test_dnsmos = DeepNoiseSuppressionMeanOpinionScore(
                self.sample_rate, 
                personalized=False
            ).to(self.device)
            #以此确保它被移动到了当前的 GPU 上
            # self.test_dnsmos = self.test_dnsmos.to(self.device)
    # --- Optimized Test Step ---
    def test_step(self, batch, batch_idx):
        mix = batch['mix']
        refs = batch['ref']
        
        # 1. Inference (GPU)
        ests = self.forward(mix)
        
        # 2. DNSMOS (GPU if available)
        # self.test_dnsmos is already initialized on the correct device
        try:
            dnsmos_res = self.test_dnsmos(ests)
            self.log('test_DNSMOS_OVRL', dnsmos_res['dnsmos_ovrl'], on_step=False, on_epoch=True)
            self.log('test_DNSMOS_SIG', dnsmos_res['dnsmos_sig'], on_step=False, on_epoch=True)
            self.log('test_DNSMOS_BAK', dnsmos_res['dnsmos_bak'], on_step=False, on_epoch=True)
            self.log('test_DNSMOS_P808', dnsmos_res['dnsmos_p808'], on_step=False, on_epoch=True)
        except Exception as e:
            print(f"DNSMOS Error: {e}")

        # 3. PESQ (CPU)
        # Move tensors to CPU explicitly for PESQ
        try:
            # torchmetrics PESQ expects inputs on CPU usually, or handles transfer internally but it's safer to detach/cpu
            pesq_score = self.test_pesq(ests.detach().cpu(), refs.detach().cpu())
            self.log('test_PESQ', pesq_score, on_step=False, on_epoch=True)
        except Exception as e:
            print(f"PESQ Error: {e}")

    def configure_optimizers(self):
        # ... (Same as before)
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.15) 
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_steps=warmup_steps, decay_until_step=total_steps,
            max_lr=self.learning_rate, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        }
        
    def train_dataloader(self):
        # ... (Same as before)
        return DataLoader(Datasets(self.train_mix_scp, self.train_ref_scp, sr=self.sample_rate, 
                           chunk_size_in_seconds=self.chunk_size_in_seconds), 
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)
    
    def val_dataloader(self):
        # ... (Same as before)
        return DataLoader(Datasets(self.val_mix_scp, self.val_ref_scp, sr=self.sample_rate, chunk_size_in_seconds=None), 
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        # Ensure drop_last=False for testing to evaluate all samples
        dataset = Datasets(self.test_mix_scp, self.test_ref_scp, sr=self.sample_rate, chunk_size_in_seconds=None)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
                          shuffle=False, drop_last=False, pin_memory=True)