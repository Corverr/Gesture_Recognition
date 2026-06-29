import torch
import torch.nn as nn
from src.models.base_model import BaseVideoModel


class MViTPatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, embed_dim=768,
                 patch_size=(2, 16, 16), frame_size=224):
        super(MViTPatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches_t = frame_size // patch_size[0]
        self.num_patches_h = frame_size // patch_size[1]
        self.num_patches_w = frame_size // patch_size[2]
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        x = self.projection(x)
        batch_size, embed_dim, t, h, w = x.shape
        x = x.view(batch_size, embed_dim, -1)
        x = x.permute(0, 2, 1)
        x = x + self.positional_encoding
        return x


class MViTBlock(nn.Module):

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(MViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
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


class MViT(BaseVideoModel):

    def __init__(self, num_classes, num_frames=16, frame_size=224,
                 embed_dim=384, num_heads=6, num_layers=6,
                 patch_size=(2, 16, 16)):
        super(MViT, self).__init__(num_classes, num_frames)
        self.name = "MViT"
        self.embed_dim = embed_dim

        self.patch_embed = MViTPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size
        )

        self.blocks = nn.ModuleList([
            MViTBlock(
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


def create_mvit(num_classes, num_frames=16):
    return MViT(
        num_classes=num_classes,
        num_frames=num_frames,
        embed_dim=384,
        num_heads=6,
        num_layers=6
    )