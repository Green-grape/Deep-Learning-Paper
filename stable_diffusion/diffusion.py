import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, 4*dim)
        self.linear_2 = nn.Linear(4*dim, 4*dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x


# noise + context를 같이 학습
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, 1, padding=0)

    def forward(self, feature, time):
        # feature: (Batch_Size, in_channels, height, width)
        # time: (1, 1280)
        residue = feature
        # (Batch_Size, in_channels, height, width) -> (Batch_Size, in_channels, height, width)
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # (Batch_Size, in_channels, height, width) -> (Batch_Size, out_channels, height, width)
        feature = self.conv_feature(feature)

        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        # (1, 1280) -> (1, out_channels)
        time = self.linear_time(time)

        # (Batch_Size, out_channelsm, Height, Width) + (1, out_channels, 1, 1) -> (Batch_Size, out_channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        # (Batch_Size, out_channels, Height, Width) + (Batch_Size, out_channels, Height, Width) -> (Batch_Size, out_channels, Height, Width)
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head*n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, 1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(
            n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4*channels*2)
        self.linear_geglu_2 = nn.Linear(channels*4, channels)

        self.conv_output = nn.Conv2d(channels, channels, 1, padding=0)

    def forward(self, x, context):
        residue_1 = x

        # Transform x
        x = self.groupnorm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (Batch_Size, channels, Height, Width) -> (Batch_Size, channels, Height*Width) -> (Batch_Size, Height*Width, channels)
        x = x.view((n, c, h*w)).transpose(-1, -2)

        # Normalization + Self Attention + Residual Connection
        residue_2 = x

        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x = x + residue_2

        residue_2 = x

        # Normalization + Cross Attention + Residual Connection
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residue_2
        residue_2 = x

        # Normalization + Feed Forward Network(GeGlu) + Residual Connection
        x = self.layernorm_3(x)
        # (Batch_Size, Height*Width, channels) -> (Batch_Size, Height*Width, channels*4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.silu(gate)
        # (Batch_Size, Height*Width, channels*4) -> (Batch_Size, Height*Width, channels)
        x = self.linear_geglu_2(x)
        x = x + residue_2

        # (Batch_Size, Height*Width, channels) -> (Batch_Size, channels, Height, Width)
        x = x.transpose(-1, -2).view((n, c, h, w))
        return self.conv_output(x) + residue_1


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch_Size, dim, height, width) -> (Batch_Size, dim, height*2, width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

# Switch Sequential: layer를 하나하나씩 적용하는 것으로, UNet의 AttentionBlock과 ResidualBlock을 구분하기 위해 사용한다.


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
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
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, height/8, width/8) -> (Batch_Size, 320, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, 3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(
                8, 40)),  # 8: attention head, 40: dim
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(
                8, 40)),  # 8: attention head, 40: dim

            # (Batch_Size, 320, height/8, width/8) -> (Batch_Size, 640, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, 3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(
                8, 80)),  # 8: attention head, 40: dim
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(
                8, 80)),  # 8: attention head, 80: dim

            # (Batch_Size, 640, height/16, width/16) -> (Batch_Size, 1280, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, 3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(
                8, 160)),  # 8: attention head, 160: dim
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(
                8, 160)),  # 8: attention head, 160: dim

            # (Batch_Size, 1280, height/32, width/32) -> (Batch_Size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, 3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, height/64, width/64) -> (Batch_Size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)
                             ),  # 2560-> skip connection
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            # (Batch_Size, 2560, height/32, width/32) -> (Batch_Size, 1280, height/32, width/32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280),
                             UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280),
                             UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1920, height/32, width/32) -> (Batch_Size, 1280, height/32, width/32)
            SwitchSequential(UNET_ResidualBlock(1920, 1280),
                             UNET_AttentionBlock(8, 160), Upsample(1280)),

            # (Batch_Size, 1920, height/16, width/16) -> (Batch_Size, 640, height/16, width/16)
            SwitchSequential(UNET_ResidualBlock(1920, 640),
                             UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 1280, height/16, width/16) -> (Batch_Size, 640, height/16, width/16)
            SwitchSequential(UNET_ResidualBlock(1280, 640),
                             UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640),
                             UNET_AttentionBlock(8, 80), Upsample(640)),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320),
                             UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320),
                             UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320),
                             UNET_AttentionBlock(8, 40))
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size,4, Height/8, Width/8)
        # context: (Batch_Size, Seq_Len, Dim) -> (Batch_Size, 77, 768)
        # time: (1, 1280)

        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for decoder in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = decoder(x, context, time)
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # x:(Batch_Size, 320, Height/8, width/8)

        x = self.groupnorm(x)
        x = F.silu(x)
        # (Batch_Size, 320, Height/8, width/8) -> (Batch_Size, 4, Height/8, width/8)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)  # noise time step
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # latent -> VAE의 result: (Batch_Size, 4, height/8, width/8)
        # context -> CLIP의 prompt: (Batch_size, Seq_Len, Dim)
        # time -> noise time step (1, 320), 320: vector size

        # Transformer의 positional encoding과 비슷하게 time에 대한 정보를 전달한다, 모델에게 denoification의 어느 단계인지 알려준다.
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch_Size, 4, height/8, width/8) -> (Batch_Size, 320, height, width)
        # 왜 320인가? -> UNet에서 output channel이 320이기 때문에, output layer를 지나가야 channel 수가 맞아진다.
        output = self.unet(latent, context, time)

        # (Batch_Size, 320, height, width) -> (Batch_Size, 4, height/8, width/8)
        output = self.final(output)
        return output
