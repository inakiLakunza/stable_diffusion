import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        # (1, 1280)
        return x
    

class Upsample(nn.Moduel):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
       # (batch_size, features, height, width) -> (batch_size, features, height * 2, width * 2)
       x = F.interpolate(x, scale_factor=2, mode="nearest")
       return self.conv(x) 


class SwitchSequential(nn.Sequential):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.Module([
            # Switch Sequential: Given a list of layers apply them one by one

            # (batch_size, 4, height / 8, width / 8) # 4 -> 3 image channels + 1 noise
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_residualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (batch_size, 320, height / 8, width / 8) -> (batch_size, 320, height / 16, width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, pàdding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (batch_size, 640, height / 16, width / 16) -> (batch_size, 640, height / 32, width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, pàdding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (batch_size, 1280, height / 32, width / 32) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, pàdding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height / 64, width / 64) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])


        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        )


        self.decoders = nn.ModuleList([
            # We have double than earlier because we have skip connections

            # (batch_size, 2560, height / 64, width / 64) -> (batch_size, 1280, height / 64, width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embeddings = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, height / 8, width / 8)
        # content: (batch_size, seq_len, dim)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embeddings(time)

        # (batch, 4, height / 8, width / 8) -> (batch, 320, height / 8, width / 8)
        outpùt = self.unet(latent, context, time)

        # (batch, 4, height / 8, width / 8)
        return output
    

