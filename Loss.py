# -*- encoding: utf-8 -*-
'''
@Filename    :Loss.py
@Time        :2020/07/09 22:11:13
@Author      :Kai Li
@Version     :1.0
@Modified    :Fixed for Single-Channel Speech Enhancement
'''

import torch
from itertools import permutations

class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, B x T tensor
        s: reference signal, B x T tensor
        Return:
        sisnr: B tensor
        """

        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        
        # 减去均值 (Zero-mean)
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        
        # 计算目标信号分量
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        
        # 计算 SI-SNR
        return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, ests, refs):
        # -----------------------------------------------------------
        # 修改 1: 语音增强模式 (Speech Enhancement / Single Source)
        # -----------------------------------------------------------
        # 如果输入是 Tensor，说明是单源，直接计算，跳过 PIT
        if torch.is_tensor(ests):
            # 兼容性处理：如果 refs 被包在 list 里，取出来
            if isinstance(refs, (list, tuple)):
                refs = refs[0]
            
            # 计算 SI-SNR (返回 Batch 大小的向量)
            sisnr_val = self.sisnr(ests, refs)
            # 返回负均值 (最大化 SI-SNR = 最小化 Loss)
            return -torch.mean(sisnr_val)

        # -----------------------------------------------------------
        # 修改 2: 原始 PIT 逻辑 (Source Separation / Multi Source)
        # -----------------------------------------------------------
        # 如果 ests 是列表，说明有多个说话人，执行 PIT 算法
        
        # 兼容性处理：如果 refs 是 Tensor，但在多源模式下，尝试将其转为列表（视具体 dataloader 而定）
        # 通常 Separation 任务中 refs 也是列表
        
        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # 这里的 len(ests) 是说话人数量 (Sources)，不是 Batch Size
        num_speakers = len(ests)
        batch_size = ests[0].size(0)

        # 计算所有排列组合的 SI-SNR
        # sisnr_mat shape: (Num_Permutations, Batch)
        sisnr_mat = torch.stack(
            [sisnr_loss(p) for p in permutations(range(num_speakers))])
        
        # 为每个样本选择最好的排列
        max_perutt, _ = torch.max(sisnr_mat, dim=0)
        
        # 返回负均值
        return -torch.mean(max_perutt)


if __name__ == "__main__":
    # Test 1: Speech Enhancement (Tensor input)
    print("Testing SE Mode...")
    ests = torch.randn(24, 160000) # Batch=24
    egs = torch.randn(24, 160000)
    loss = Loss()
    print("SE Loss:", loss.compute_loss(ests, egs))

    # Test 2: Source Separation (List input)
    print("Testing SS/PIT Mode...")
    ests_list = [torch.randn(4, 320), torch.randn(4, 320)] # 2 Speakers, Batch=4
    egs_list = [torch.randn(4, 320), torch.randn(4, 320)]
    print("SS Loss:", loss.compute_loss(ests_list, egs_list))