import torch
import torch.nn as nn
from src.models.base_model import BaseVideoModel


class MViTPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128,
                 patch_size_time=2, patch_size_hw=4,
                 num_frames=4, frame_size=64):
        super(MViTPatchEmbedding, self).__init__()

        self.embed_dim = embed_dim

        # Количество патчей по каждому измерению
        self.num_patches_t = num_frames // patch_size_time
        self.num_patches_h = frame_size // patch_size_hw
        self.num_patches_w = frame_size // patch_size_hw
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        print(f"MViT: num_patches_t={self.num_patches_t}, num_patches_h={self.num_patches_h}, "
              f"num_patches_w={self.num_patches_w}, total={self.num_patches}")

        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size_time, patch_size_hw, patch_size_hw),
            stride=(patch_size_time, patch_size_hw, patch_size_hw)
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        # x: [batch, channels, frames, height, width]
        x = self.projection(x)  # [batch, embed_dim, t, h, w]
        batch_size, embed_dim, t, h, w = x.shape

        # Проверяем, что количество патчей совпадает
        expected_patches = t * h * w
        if expected_patches != self.num_patches:
            print(f"Warning: expected {self.num_patches} patches, got {expected_patches}")

        x = x.view(batch_size, embed_dim, -1)  # [batch, embed_dim, patches]
        x = x.permute(0, 2, 1)  # [batch, patches, embed_dim]

        # Обрезаем или дополняем positional_encoding при необходимости
        if x.shape[1] != self.positional_encoding.shape[1]:
            if x.shape[1] < self.positional_encoding.shape[1]:
                pos_enc = self.positional_encoding[:, :x.shape[1], :]
            else:
                pos_enc = torch.cat([
                    self.positional_encoding,
                    torch.zeros(1, x.shape[1] - self.positional_encoding.shape[1], self.embed_dim).to(x.device)
                ], dim=1)
        else:
            pos_enc = self.positional_encoding

        x = x + pos_enc
        return x


class MViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super(MViTBlock, self).__init__()

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


class MViT(BaseVideoModel):
    def __init__(self, num_classes, num_frames=4, frame_size=64,
                 embed_dim=128, num_heads=4, num_layers=2,
                 patch_size_time=2, patch_size_hw=4):
        super(MViT, self).__init__(num_classes, num_frames)
        self.name = "MViT"
        self.embed_dim = embed_dim

        self.patch_embed = MViTPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size_time=patch_size_time,
            patch_size_hw=patch_size_hw,
            num_frames=num_frames,
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
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits


def create_mvit(num_classes, num_frames=4):
    return MViT(
        num_classes=num_classes,
        num_frames=num_frames,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        patch_size_time=2,
        patch_size_hw=4
    )