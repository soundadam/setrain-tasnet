import torch
import torch.nn as nn

# ==========================================
# 1. 原始实现: Unfold + Reshape (硬编码提取)
# ==========================================
class SFE_Original(nn.Module):
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        # Unfold 提取滑动窗口
        # kernel_size=(1, k) 表示只在 Freq 轴上滑动，Time 轴保持原样
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), 
                                stride=(1, stride), 
                                padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        # x: (B, C, T, F)
        B, C, T, F = x.shape
        # Unfold 输出: (B, C * 1 * k, L), 其中 L = T * F
        # Reshape 目的: 恢复 T, F 维度，并将 k 堆叠到 Channel 维度
        # 输出形状: (B, C * k, T, F)
        xs = self.unfold(x).reshape(B, C * self.kernel_size, T, F)
        return xs

# ==========================================
# 2. 推荐替代方案: Grouped Conv2d (可学习提取)
# ==========================================
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

# ==========================================
# 3. 验证与对比环境
# ==========================================
def run_playground():
    # --- 模拟输入数据 ---
    # Batch=2, Channels=16, Time=100, Freq=64
    B, C, T, F = 2, 16, 100, 64
    x = torch.randn(B, C, T, F)
    
    k_size = 3
    
    print(f"Input Shape: {x.shape}")
    print(f"Kernel Size: {k_size}")
    print("-" * 40)

    # --- 1. 运行原始 Unfold ---
    model_unfold = SFE_Original(kernel_size=k_size)
    y_unfold = model_unfold(x)
    print(f"[Original Unfold] Output Shape: {y_unfold.shape}")
    # 参数量: 0 (Unfold 只是复制像素，没有权重)
    print(f"[Original Unfold] Params: 0 (Hard-coded mixing)")
    
    print("-" * 40)

    # --- 2. 运行 Conv2d 替代版 ---
    model_conv = SFE_Conv2d(in_channels=C, kernel_size=k_size, dilation=1)
    y_conv = model_conv(x)
    print(f"[Conv2d Replace ] Output Shape: {y_conv.shape}")
    # 参数量计算: (k * 1 * 1) * out_channels / groups 
    # = 3 * (16*3) / 16 = 9 个参数 per group * 16 groups = 144
    num_params = sum(p.numel() for p in model_conv.parameters())
    print(f"[Conv2d Replace ] Params: {num_params} (Learnable mixing)")

    # --- 3. 运行 Conv2d 膨胀版 (TODO 建议的) ---
    dilation_rate = 2
    model_dilated = SFE_Conv2d(in_channels=C, kernel_size=k_size, dilation=dilation_rate)
    y_dilated = model_dilated(x)
    print(f"[Dilated Conv2d ] Output Shape: {y_dilated.shape} (Dilation={dilation_rate})")
    
    print("-" * 40)

    # --- 4. 维度一致性检查 ---
    assert y_unfold.shape == y_conv.shape, "Error: Shapes do not match!"
    print("✅ SUCCESS: Output shapes match perfectly.")
    
    print("\n[分析]")
    print("1. Unfold+Reshape 实际上是把每个频点 f 的邻居 [f-1, f, f+1] 搬运到了通道维度。")
    print("2. Grouped Conv2d 做的事情是：对 [f-1, f, f+1] 进行加权求和。")
    print("3. 如果 Conv2d 输出通道数 = 输入通道数 * Kernel，它就有能力保留所有信息（不仅是求和）。")
    print("4. 使用 Conv2d 的好处是：网络可以学会'如何'利用邻居频率信息，而不是简单粗暴地堆叠。")

if __name__ == "__main__":
    run_playground()