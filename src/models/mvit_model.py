"""
Модель MViT (Multiscale Vision Transformer) для распознавания жестов.
Современная архитектура на основе трансформеров для видео.
"""

import torch
import torch.nn as nn

from src.models.base_model import BaseVideoModel


class MViTPatchEmbedding(nn.Module):
    """
    Разбивает видео на патчи и преобразует их в эмбеддинги.

    Как работает:
        1. Видео размером [B, C, T, H, W]
        2. Разбивается на патчи размером [P_T, P_H, P_W]
        3. Каждый патч превращается в вектор размером embed_dim
        4. Добавляется позиционное кодирование
    """

    def __init__(self, in_channels=3, embed_dim=768,
                 patch_size=(2, 16, 16), frame_size=224):
        """
        Параметры:
            in_channels (int): Количество каналов (3 для RGB)
            embed_dim (int): Размер эмбеддинга
            patch_size (tuple): Размер патча (время, высота, ширина)
            frame_size (int): Размер кадра
        """
        super(MViTPatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Вычисляем количество патчей
        self.num_patches_t = frame_size // patch_size[0]
        self.num_patches_h = frame_size // patch_size[1]
        self.num_patches_w = frame_size // patch_size[2]
        self.num_patches = self.num_patches_t * self.num_patches_h * self.num_patches_w

        # 3D свертка для создания патчей
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Позиционное кодирование (для каждого патча)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        """
        Параметры:
            x (Tensor): [batch_size, channels, frames, height, width]

        Возвращает:
            patches (Tensor): [batch_size, num_patches, embed_dim]
        """
        # Проецируем через 3D свертку
        x = self.projection(x)  # [B, embed_dim, num_patches_t, num_patches_h, num_patches_w]

        # Распрямляем
        batch_size, embed_dim, t, h, w = x.shape
        x = x.view(batch_size, embed_dim, -1)  # [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, embed_dim]

        # Добавляем позиционное кодирование
        x = x + self.positional_encoding

        return x


class MViTBlock(nn.Module):
    """
    Один блок MViT трансформера с многомасштабным вниманием.
    """

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
        """
        Параметры:
            x (Tensor): [batch_size, num_patches, embed_dim]

        Возвращает:
            x (Tensor): [batch_size, num_patches, embed_dim]
        """
        # Self-attention с residual связью
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = residual + self.dropout(attn_out)

        # MLP с residual связью
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class MViT(BaseVideoModel):
    """
    Multiscale Vision Transformer для распознавания жестов.

    Преимущества:
        - Механизм внимания позволяет видеть глобальные взаимосвязи
        - Многомасштабная обработка
        - Современное SOTA качество

    Недостатки:
        - Требует много памяти
        - Медленнее CNN на маленьких данных
    """

    def __init__(self, num_classes, num_frames=16, frame_size=224,
                 embed_dim=384, num_heads=6, num_layers=6,
                 patch_size=(2, 16, 16)):
        """
        Конструктор.

        Параметры:
            num_classes (int): Количество классов
            num_frames (int): Количество кадров
            frame_size (int): Размер кадра
            embed_dim (int): Размер эмбеддинга
            num_heads (int): Количество голов в attention
            num_layers (int): Количество блоков трансформера
            patch_size (tuple): Размер патча
        """
        super(MViT, self).__init__(num_classes, num_frames)
        self.name = "MViT"
        self.embed_dim = embed_dim

        # ===== 1. PATCH EMBEDDING =====
        self.patch_embed = MViTPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size
        )

        # ===== 2. ТРАНСФОРМЕРНЫЕ БЛОКИ =====
        self.blocks = nn.ModuleList([
            MViTBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        # ===== 3. КЛАССИФИКАЦИЯ =====
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Прямой проход.

        Параметры:
            x (Tensor): [batch_size, 3, frames, 224, 224]

        Возвращает:
            logits (Tensor): [batch_size, num_classes]
        """
        # ===== 1. Создаем патчи =====
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # ===== 2. Проходим через трансформер =====
        for block in self.blocks:
            x = block(x)

        # ===== 3. Усредняем по патчам =====
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, embed_dim]

        # ===== 4. Классификация =====
        logits = self.classifier(x)  # [B, num_classes]

        return logits


def create_mvit(num_classes, num_frames=16):
    """Фабричная функция для создания MViT модели."""
    return MViT(
        num_classes=num_classes,
        num_frames=num_frames,
        embed_dim=384,
        num_heads=6,
        num_layers=6
    )