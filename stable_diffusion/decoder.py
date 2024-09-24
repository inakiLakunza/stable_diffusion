
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        # Closer features to each other will have a kind of the same distribution
        # and things that are far from each other will not. The whole idea behing
        # group normalization is not to make these oscillate too much, so that the training is faster
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)

        residue = x

        n, c, h, w = x.shape

        # We need to do a reshaping and a transposition to apply attention

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(n, c, h * w)

        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        x = x.transpose(-1, -2)

        # (batch_size, height * width, features) -> (batch_size, height * width, features)
        x = self.attention(x)

        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, features, height * width) -> (batch_size, features, height, width)
        x = x.view(n, c, h, w)

        x += residue
        
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            # In order to do the adding we need the same dimensionality
            # so we use a convoluational layer when needed
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)

        residue = x
        
        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        # Add output with `'input' , if needed the dimensionality is changed using a 2d conv
        return x + self.residual_layer(residue)