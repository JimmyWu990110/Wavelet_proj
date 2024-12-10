import torch
import math
from pytorch_wavelets import DWTForward, DWTInverse

class ImageAttentionBlock(torch.nn.Module):
    def __init__(self, hidden_dim, image_size, num_head=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.layer_norm = torch.nn.LayerNorm([hidden_dim])
        self.att = torch.nn.MultiheadAttention(hidden_dim, num_head, batch_first=True)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.LayerNorm([hidden_dim]),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):  # x: [B, C, H, W]
        x = x.reshape(-1, self.hidden_dim, self.image_size * self.image_size).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.layer_norm(x)
        attention_value, _ = self.att(x_norm, x_norm, x_norm)
        x = x + attention_value
        x = x + self.feed_forward(x)
        x = x.transpose(1, 2).reshape(-1, self.hidden_dim, self.image_size, self.image_size)
        return x

class BottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class BaseConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups)
        self.norm = torch.nn.GroupNorm(groups, out_channels)
        self.act = torch.nn.GELU()
    
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = BaseConvBlock(in_channels, in_channels)
        self.l2 = BaseConvBlock(in_channels, out_channels)

    def forward(self, x):
        res = x
        x = self.l1(x)
        x = self.l2(x)
        return (x + res) / math.sqrt(2)

class WaveShort(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='symmetric')
        self.conv = torch.nn.Conv2d(4 * in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=4)

    def forward(self, x):
        xl, xh = self.dwt(x)
        b, c, _, h, w = xh[0].shape
        xh = xh[0].reshape(b, 3 * c, h, w)
        x = torch.cat([xl, xh], dim=1)
        x = self.conv(x)
        return x

class WaveBottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, model='symmetric')
        self.idwt = DWTInverse(wave=wave, model='symmetric')
        self.res = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        xl, xh = self.dwt(x)
        xl = self.res(xl / 2) * 2
        x = self.idwt((xl, xh))
        return x

class WaveDownSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='symmetric')
        self.conv1 = BaseConvBlock(in_channels, out_channels)
        self.conv2 = BaseConvBlock(out_channels, out_channels)
        self.short = BottleNeckBlock(in_channels, out_channels // 4)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        x = self.short(x)
        hl, hh = self.dwt(h)
        b, c, _, w, h = hh[0].shape
        hh = hh[0].reshape(b, 3 * c, w, h)
        xl, xh = self.dwt(h)
        hl = hl / 2.
        x = xl / 2.
        h = hl + t_emb
        h = self.conv2(hl)
        return (x + h) / math.sqrt(2), hh
    

class WaveUpSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.idwt = DWTInverse(wave=wave, mode='symmetric')
        self.conv1 = BaseConvBlock(in_channels, out_channels)
        self.conv2 = BaseConvBlock(out_channels, out_channels)
        self.short = BottleNeckBlock(in_channels, out_channels)
        self.skip_conv = BaseConvBlock(in_channels * 3, out_channels * 3, groups=3)

    def forward(self, x, skip, t_emb):  # Skip Connection
        h = self.conv1(x)
        x = self.short(x)
        b, c, w, h = x.shape
        skip = self.skip_conv(skip / 2.) * 2.
        skip = skip.reshape(b, c, 3, w, h)
        h = self.idwt((2. * h, [skip]))
        x = self.idwt((2. * x, [skip]))
        h = h + t_emb
        h = self.conv2(h)
        return (x + h) / math.sqrt(2)
    

class FreqAwareDiffusion(torch.nn.Module):
    def __init__(self, image_size, image_channels, time_range, wave='haar', beta_range=(1e-4, 0.02), wavespace=False, device="cuda", role="denoise"):
        super().__init__()
        self.beta_range = beta_range
        self.time_range = time_range
        self.image_size = image_size
        self.role = "denoise"
        self.device = device
        self.wavespace = wavespace
        if wavespace:
            image_channels *= 4
            image_size = image_size // 2
            self.in_dwt = DWTForward(J=1, wave=wave, mode='symmetric')
            self.out_idwt = DWTInverse(wave=wave, mode='symmetric')
        
        self.short1 = WaveBottleBlock(image_channels, 128, wave=wave)
        self.short2 = WaveBottleBlock(128, 256, wave=wave)
        
        self.in_conv = torch.nn.Conv2d(image_channels, 64, kernel_size=3, padding=1, bias=False, groups=groups)
        self.res1 = ResidualBlock(64, 64)
        self.down1 = WaveDownSampleBlock(64, 128, wave=wave)
        self.res2 = ResidualBlock(128, 128)
        self.down2 = WaveDownSampleBlock(128, 256, wave=wave)
        self.bottle1 = WaveBottleNeckBlock(256, 128, wave=wave)
        self.att = ImageAttentionBlock(128, image_size // 4)
        self.bottle2 = WaveBottleNeckBlock(128, 256, wave=wave)
        self.res3 = ResidualBlock(256, 256)
        self.up1 = WaveUpSampleBlock(256, 128, wave=wave)
        self.res4 = ResidualBlock(128, 128)
        self.up2 = WaveUpSampleBlock(128, 64, wave=wave)
        self.out_conv = BaseConvBlock(64, image_channels)

    @torch.no_grad()
    def pos_encoding(self, t, channels, embed_size):
        if self.wavespace:
            embed_size = embed_size // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        enc1 = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        enc2 = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        enc = torch.cat([enc1, enc2], dim=-1)
        return enc.reshape(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
    
    def forward(self, x, t):
        if self.wavespace:
            xl, xh = self.in_dwt(x)
            b, c, _, h, w = xh[0].shape
            xh = xh[0].reshape(b, 3 * c, h, w)
            x = torch.cat([xl, xh], dim=1)
        s1 = self.short1(x)
        s2 = self.short2(s1)
        
        h = self.in_conv(x)
        h = self.res1(h)
        h, skip1 = self.down1(h, self.pos_encoding(t, 128, self.image_size // 2)
        h = (h + s1) / math.sqrt(2)
        h = self.res2(h)
        h, skip2 = self.down2(h, self.pos_encoding(t, 256, self.image_size // 4)
        h = (h + s2) / math.sqrt(2)
        h = self.bottle1(h)
        h = self.att(h)
        h = self.bottle2(h)
        h = self.res3(h)
        h = self.up1(h, skip2, self.pos_encoding(t, 128, self.image_size // 2)
        h = self.res4(h)
        h = self.up2(h, skip1, self.pos_encoding(t, 64, self.image_size)
        x = self.out_conv(h)
        if self.wavespace:
            c = x.shape[1] // 4
            xl = x[:, :c]
            xh = x[:, c:].reshape(b, c, 3, h, w)
            x = self.out_idwt((xl, [xh]))
        return x

    def get_beta(self, t):
        return self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) * t / self.time_range
    
    def get_alpha(self, t):
        return 1 - self.get_beta(t)
    
    def get_alpha_bar(self, t):
        return math.prod([self.get_alpha(i) for i in range(t)])
    
    def get_noisy_img(self, img, noise, t):
        alpha_bar = self.get_alpha_bar(t)
        return math.sqrt(alpha_bar) * img + math.sqrt(1 - alpha_bar) * noise

    def get_denoised_img(self, img, noise, z, t):
        alpha = self.get_alpha(t)
        beta = self.get_beta(t)
        alpha_bar = self.get_alpha_bar(t)
        pre_scale = 1 / math.sqrt(alpha)
        noise_scale = beta / math.sqrt(1 - alpha_bar)
        post_sigma = math.sqrt(beta) * z
        x = pre_scale * (img - noise_scale * noise) + post_sigma
        return x

    def loss(self, batch):
        batch = batch.to(self.device)
        times = torch.randint(1, self.time_range, (batch.shape[0],), device=self.device)
        noises = torch.randn_like(batch, device=self.device)
        noisy_imgs = torch.stack([self.get_noisy_img(img, noise, t) for img, noise, t in zip(batch, noises, times)], dim=0)
        pred_noise = self(noisy_imgs, times.unsqueeze(-1).type(torch.float))
        loss = torch.nn.functional.mse_loss(pred_noise.reshape(-1, self.image_size * self.image_size), noises.reshape(-1, self.image_size * self.image_size))
        return loss
    
    @torch.no_grad()
    def denoise(self, x, t):
        if t > 1 and self.role != "denoise":
            z = torch.randn_like(x, device=self.device)
        else:
            z = 0
        pred_noise = self(x, t.reshape(1, 1).repeat(x.shape[0], 1))
        return self.get_denoised_img(x, pred_noise, z, t)

    @torch.no_grad()
    def generate(self, n):
        x = torch.randn(n, 3, self.image_size, self.image_size, device=self.device)
        for t in range(self.time_range, 0, -1):
            x = self.denoise(x, torch.tensor(t, device=self.device))
        return x
        
    @torch.no_grad()
    def generational_denoise(self, x, t):  # x: [B, C, H, W]
        for i in range(t, 0, -1):
            x = self.denoise(x, torch.tensor(i, device=self.device))
        return x
