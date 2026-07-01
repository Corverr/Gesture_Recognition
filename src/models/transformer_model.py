import torch
import torch.nn as nn
from src.models.base_model import BaseVideoModel


class TimeSformerPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=384,
                 patch_size=(8, 8), frame_size=112, num_frames=8):
        super(TimeSformerPatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.num_patches_h = frame_size // patch_size[0]
        self.num_patches_w = frame_size // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_tokens = self.num_frames * self.num_patches

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_frames, embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

    def forward(self, x):
        batch_size, channels, frames, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)

        x = self.projection(x)

        x = x.view(batch_size * frames, self.embed_dim, -1)

        x = x.permute(0, 2, 1)

        x = x.view(batch_size, frames, self.num_patches, self.embed_dim)

        x = x + self.pos_embed_spatial.unsqueeze(0)

        x = x + self.pos_embed_temporal.unsqueeze(2)

        x = x.view(batch_size, self.num_frames * self.num_patches, self.embed_dim)

        return x


class TimeSformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TimeSformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_out)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class TimeSformer(BaseVideoModel):
    def __init__(self, num_classes, num_frames=8, frame_size=112,
                 embed_dim=384, num_heads=6, num_layers=4,
                 patch_size=(8, 8)):
        super(TimeSformer, self).__init__(num_classes, num_frames)
        self.name = "TimeSformer"
        self.embed_dim = embed_dim

        self.patch_embed = TimeSformerPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size,
            num_frames=num_frames
        )

        self.blocks = nn.ModuleList([
            TimeSformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


def create_transformer(num_classes, num_frames=8):
    return TimeSformer(
        num_classes=num_classes,
        num_frames=num_frames,
        embed_dim=384,
        num_heads=6,
        num_layers=4
    )