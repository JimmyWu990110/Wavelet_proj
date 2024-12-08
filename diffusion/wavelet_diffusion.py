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
        self.norm = torch.nn.GroupNorm(1, out_channels)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class BaseConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.residual = residual
        self.conv1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = torch.nn.GroupNorm(1, mid_channels)
        self.act1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = torch.nn.GroupNorm(1, out_channels)
    
    def forward(self, x):
        if self.residual:
            residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.residual:
            x = torch.nn.functional.gelu(x + residual)
        return x


class WaveDownSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='symmetric')
        self.conv = torch.nn.Sequential(
            BottleNeckBlock(in_channels * 4, in_channels),  # Adjust Channels
            BaseConvBlock(in_channels, in_channels, residual=True),
            BaseConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        xl, xh = self.dwt(x)
        b, c, _, h, w = xh[0].shape
        xh = xh[0].reshape(b, 3 * c, h, w)
        x = torch.cat([xl, xh], dim=1)
        x = self.conv(x)
        return x
    

class WaveUpSampleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, wave='haar'):
        super().__init__()
        self.bottle = BottleNeckBlock(in_channels // 2, 2 * in_channels)
        self.idwt = DWTInverse(wave=wave, mode='symmetric')
        self.conv = torch.nn.Sequential(
            BaseConvBlock(in_channels, in_channels, residual=True), 
            BaseConvBlock(in_channels, out_channels, in_channels // 2)
        )

    def forward(self, x, skip):  # Skip Connection
        b, c, h, w = x.shape
        x = self.bottle(x)
        xl = x[:, :c]
        xh = x[:, c:].reshape(b, c, 3, h, w)
        x = self.idwt((xl, [xh]))
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    

class WaveDiffusion(torch.nn.Module):
    def __init__(self, image_size, image_channels, time_range, beta_range=(1e-4, 0.02), device="cuda"):
        super().__init__()
        self.beta_range = beta_range
        self.time_range = time_range
        self.image_size = image_size
        self.device = device
        
        self.in_conv = BaseConvBlock(image_channels, 64)
        self.down1 = WaveDownSampleBlock(64, 128)
        self.down2 = WaveDownSampleBlock(128, 256)
        self.att1 = ImageAttentionBlock(256, image_size // 4)
        self.down3 = WaveDownSampleBlock(256, 256)
        self.att2 = ImageAttentionBlock(256, image_size // 8)
        self.up1 = WaveUpSampleBlock(512, 128)
        self.att3 = ImageAttentionBlock(128, image_size // 4)
        self.up2 = WaveUpSampleBlock(256, 64)
        self.up3 = WaveUpSampleBlock(128, 64)
        self.out_conv = torch.nn.Conv2d(64, image_channels, kernel_size=1)

    @torch.no_grad()
    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        enc1 = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        enc2 = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        enc = torch.cat([enc1, enc2], dim=-1)
        return enc.reshape(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)
    
    def forward(self, x, t):
        x1 = self.in_conv(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, self.image_size // 2)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, self.image_size // 4)
        x3 = self.att1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, self.image_size // 8)
        x4 = self.att2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, self.image_size // 4)
        x = self.att3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, self.image_size // 2)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, self.image_size)
        x = self.out_conv(x)
        return x

    def get_beta(self, t):
        return self.beta_range[0] + (self.beta_range[1] - self.beta_range[0]) * t / self.time_range
    
    def get_alpha(self, t):
        return 1 - self.get_beta(t)
    
    def get_alpha_bar(self, t):
        return math.prod([self.get_alpha(i) for i in range(t)])
    
    def get_noisy_img(self, img, noise, t):
        alpha_bar = self.get_alpha_bar(t)
        return math.sqrt(alpha_bar) * img + math.sqrt(1-alpha_bar) * noise

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
        times = torch.randint(0, self.time_range, (batch.shape[0],), device=self.device)
        noises = torch.randn_like(batch, device=self.device)
        noisy_imgs = torch.stack([self.get_noisy_img(img, noise, t) for img, noise, t in zip(batch, noises, times)], dim=0)
        pred_noise = self(noisy_imgs, times.unsqueeze(-1).type(torch.float))
        loss = torch.nn.functional.mse_loss(pred_noise.reshape(-1, self.image_size * self.image_size), noises.reshape(-1, self.image_size * self.image_size))
        return loss
    
    @torch.no_grad()
    def denoise(self, x, t):
        if t > 1:
            z = torch.randn_like(x, device=self.device)
        else:
            z = 0
        pred_noise = self(x, t.reshape(1, 1).repeat(x.shape[0], 1))
        return self.get_denoised_img(x, pred_noise, z, t)

    @torch.no_grad()
    def generate(self, n):
        x = torch.randn(n, 3, self.image_size, self.image_size, device=self.device)
        for t in reversed(range(self.time_range)):
            x = self.denoise(x, torch.tensor(t, device=self.device))
        return x
