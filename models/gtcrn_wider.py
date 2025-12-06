"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Modified for configurability and potential performance scaling.
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
# from utils.dev_modules import SFE

class LayerNorm2d(nn.Module):
    """
    Apply LayerNorm on the Channel dimension for (B, C, T, F) input.
    Equivalent to permuting to (B, T, F, C), applying LN, and permuting back.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # x: (B, C, T, F)
        x = x.permute(0, 2, 3, 1)  # -> (B, T, F, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, T, F)
        return x
    
class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        # TODO: High-Performance Tip 1: Relax Band Merging
        # The current compression (192 high-freq bins -> 64 ERB bins) is lossy.
        # Try increasing erb_subband_2 (e.g., to 96 or 128) to preserve more high-freq details.
        # Alternatively, replace fixed ERB filters with a learnable 1D Conv layer.
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        # TODO: High-Performance Tip 3: Add Frequency Attention
        # TRA only models time. Adding a lightweight SE-Block (Squeeze-and-Excitation) 
        # for Frequency or Channels here can boost performance significantly.
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        #TODO 流式处理中，att_gru只会处理一小段，并没有存储ht_final的信息，会不会失效？
        # 换言之，长序列建模会失效？
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False, use_layer_norm=False):
        super().__init__()
        self.use_deconv = use_deconv
        # Expand depthwise kernel for larger receptive fields while keeping odd sizes.
        # base_kt, base_kf = kernel_size
        # make_odd = lambda v: v if v % 2 else v + 1
        # large_kernel = (
        #     make_odd(base_kt if base_kt >= 5 else 5),
        #     make_odd(base_kf if base_kf >= 5 else 5),
        # )
        # self.pad_size = (large_kernel[0]-1) * dilation[0]
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        # extra_freq_pad = ((large_kernel[1] - base_kf) * dilation[1]) // 2
        # pad_t_arg, pad_f_arg = padding
        # if use_deconv:
        #     depth_padding = (
        #         pad_t_arg + (large_kernel[0] - base_kt) * dilation[0],
        #         pad_f_arg + extra_freq_pad,
        #     )
        # else:
        #     depth_padding = (0, pad_f_arg + extra_freq_pad)
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        if use_layer_norm == True:
            self.point_norm1 = nn.BatchNorm2d(hidden_channels)
            self.point_norm2 = nn.BatchNorm2d(in_channels//2)
        else:
            self.point_norm1 = LayerNorm2d(hidden_channels)
            self.point_norm2 = LayerNorm2d(in_channels//2)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        # 1. Stack
        x = torch.stack([x1, x2], dim=2) # (B, C, 2, T, F)
        B, C, G, T, F = x.shape
        x = x.view(B, C * G, T, F) # (B, 2C, T, F)
        # through einops.rearrange, easier to read
        # x = torch.stack([x1, x2], dim=1)
        # x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        # x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_norm1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_norm2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # TODO: High-Performance Tip 5: Unlock Grouping
        # Splitting RNN into groups saves paramters but hurts information flow.
        # If computation allows, remove the split and use a single large GRU.
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, use_grouped=True, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size
        
        # Intra RNN
        
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter RNN
        # Configurable Grouped RNN
        if use_grouped:
            self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        else:
            # Use standard GRU if grouping is disabled (Better performance)
            self.inter_rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
            self.intra_rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size//2, batch_first=True, bidirectional=True)
            
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out


class Encoder(nn.Module):
    def __init__(self, base_channels=16, kernel_size=(3,3), use_layer_norm=False):
        super().__init__()
        # 3 (input chan) * 3 (SFE kernel) = 9
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, base_channels, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            # Parametric Kernel Size
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), padding=(0, (kernel_size[1]-1)//2), dilation=(1,1), use_deconv=False, use_layer_norm=use_layer_norm),
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), padding=(0, (kernel_size[1]-1)//2), dilation=(2,1), use_deconv=False, use_layer_norm=use_layer_norm),
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), padding=(0, (kernel_size[1]-1)//2), dilation=(5,1), use_deconv=False, use_layer_norm=use_layer_norm)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self, base_channels=16, kernel_size=(3,3)):
        super().__init__()
        # 辅助函数：计算 padding
        # pad_t: 时间轴 padding。对于 Decoder，必须严格等于 (K-1)*D，才能将 F.pad 增加的长度以及 Deconv 自身的扩展全部抵消。
        pad_t = lambda k, d: (k - 1) * d
        # pad_f: 频率轴 padding。保持为 "Same Padding"，即 (K-1)//2。
        pad_f = lambda k: (k - 1) // 2
        
        k_t, k_f = kernel_size
        
        self.de_convs = nn.ModuleList([
            # 修正后的 padding 计算：
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), 
                        padding=(pad_t(k_t, 5), pad_f(k_f)), dilation=(5,1), use_deconv=True, use_layer_norm=False),
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), 
                        padding=(pad_t(k_t, 2), pad_f(k_f)), dilation=(2,1), use_deconv=True, use_layer_norm=False),
            GTConvBlock(base_channels, base_channels, kernel_size, stride=(1,1), 
                        padding=(pad_t(k_t, 1), pad_f(k_f)), dilation=(1,1), use_deconv=True, use_layer_norm=False),
            ConvBlock(base_channels, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        # --- Configurable Parameters ---
        base_channels=16,      # Increase to 32 or 64 for performance
        kernel_size=(3,3),     # Increase to (5,5) or (7,7)
        use_grouped_rnn=True   # Set to False to disable grouping (increase params/performance)
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder(base_channels=base_channels, kernel_size=kernel_size)
        
        width = 65 # calconv1d(129, ker=5, stride=2, padding=2, dialation=1, groups=1)
        
        self.dpgrnn1 = DPGRNN(base_channels, width, base_channels, use_grouped=use_grouped_rnn)
        self.dpgrnn2 = DPGRNN(base_channels, width, base_channels, use_grouped=use_grouped_rnn)
        
        self.decoder = Decoder(base_channels=base_channels, kernel_size=kernel_size)

        self.mask = Mask()

    def forward(self, x):
        """
        x: (B, L)
        """
        device = x.device
        n_samples = x.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
        
        spec = torch.stft(x,  **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        
        spec = spec.permute(0,3,2,1)  # (B,2,T,F)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


if __name__ == "__main__":
    model = GTCRN(base_channels=24, use_grouped_rnn=True).eval()

    from ptflops import get_model_complexity_info
    """complexity count"""
    dummy = torch.randn(1, 16000)   
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                            print_per_layer_stat=False, verbose=True)
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(flops, params/1e3)
    

    """causality check"""
    a = torch.randn(2, 16000)
    b = torch.randn(2, 16000)
    c = 1e12*torch.randn(2, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    y1 = model(x1)[0]
    y2 = model(x2)[0]

    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
