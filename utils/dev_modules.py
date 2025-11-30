import torch.nn as nn

# replace SFE with SFE_Conv2d
class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        # TODO: High-Performance Tip 2: Improve Frequency Modeling
        # Instead of Unfold+Reshape (which is hard-coded mixing), 
        # consider using a Dilated Conv2d along the frequency axis or a large kernel (1, 7) Conv2d.
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs
    
class SFE_Conv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1):
        super().__init__()
        # 目标: 输出形状要和 SFE_Original 一致 -> (B, C*k, T, F)
        # 所以 out_channels = in_channels * kernel_size
        
        # padding 计算: 为了保持 F 维度不变 (Same Padding)
        # formula: padding = (kernel_size - 1) * dilation // 2
        pad_f = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * kernel_size, # 扩充通道数以匹配 Unfold
            kernel_size=(1, kernel_size),           # 只在 Freq 方向卷积
            stride=(1, 1),
            padding=(0, pad_f),                     # Time轴不padding，Freq轴padding
            dilation=(1, dilation),                 # 支持膨胀卷积 (TODO 中提到的)
            groups=in_channels,                     # 关键点！
            bias=False                              # Unfold 没有 bias，这里也可以选 False
        )
        
        # 初始化技巧 (可选):
        # 如果你想让初始状态接近 Unfold (即中心为1，两侧为0，不做加权混合)，
        # 可以手动初始化权重。但通常随机初始化训练效果更好。

    def forward(self, x):
        # x: (B, C, T, F)
        # Conv2d 天然支持这种操作，且无需 Reshape
        return self.conv(x)