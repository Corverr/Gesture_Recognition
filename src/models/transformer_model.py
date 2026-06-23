"""
Модель Video Transformer (TimeSformer) для распознавания жестов.
Разделяет пространственное и временное внимание для эффективности.
"""

import torch
import torch.nn as nn

from src.models.base_model import BaseVideoModel


class TimeSformerPatchEmbedding(nn.Module):
    """
    Разбивает видео на патчи и создает эмбеддинги.

    Особенность: добавляет отдельные позиционные кодирования
    для пространства и времени.
    """

    def __init__(self, in_channels=3, embed_dim=768,
                 patch_size=(16, 16), frame_size=224, num_frames=16):
        super(TimeSformerPatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Количество пространственных патчей
        self.num_patches_h = frame_size // patch_size[0]
        self.num_patches_w = frame_size // patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_tokens = self.num_frames * self.num_patches

        # 2D свертка для пространственных патчей
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Пространственное позиционное кодирование
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Временное позиционное кодирование
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_frames, embed_dim)
        )

        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

    def forward(self, x):
        """
        Параметры:
            x (Tensor): [batch_size, channels, frames, height, width]

        Возвращает:
            tokens (Tensor): [batch_size, num_frames * num_patches, embed_dim]
        """
        batch_size, channels, frames, h, w = x.shape

        # ===== 1. Обрабатываем каждый кадр отдельно =====
        # Переставляем размерности: [B*T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)

        # Патчи: [B*T, embed_dim, num_patches_h, num_patches_w]
        x = self.projection(x)

        # Распрямляем: [B*T, embed_dim, num_patches]
        x = x.view(batch_size * frames, self.embed_dim, -1)

        # Переставляем: [B*T, num_patches, embed_dim]
        x = x.permute(0, 2, 1)

        # ===== 2. Восстанавливаем временную размерность =====
        x = x.view(batch_size, frames, self.num_patches, self.embed_dim)

        # ===== 3. Добавляем позиционные кодирования =====
        # Пространственное: добавляем к каждому кадру
        x = x + self.pos_embed_spatial.unsqueeze(0)

        # Временное: добавляем к каждому пространственному патчу
        x = x + self.pos_embed_temporal.unsqueeze(2)

        # ===== 4. Объединяем во временной последовательности =====
        x = x.view(batch_size, self.num_frames * self.num_patches, self.embed_dim)

        return x


class TimeSformerBlock(nn.Module):
    """
    Блок трансформера с разделенным пространственным и временным вниманием.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TimeSformerBlock, self).__init__()

        self.norm1_space = nn.LayerNorm(embed_dim)
        self.attention_space = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1_time = nn.LayerNorm(embed_dim)
        self.attention_time = nn.MultiheadAttention(
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
        """
        Параметры:
            x (Tensor): [batch_size, num_frames * num_patches, embed_dim]

        Возвращает:
            x (Tensor): [batch_size, num_frames * num_patches, embed_dim]
        """
        batch_size, num_tokens, embed_dim = x.shape

        # ===== 1. ПРОСТРАНСТВЕННОЕ ВНИМАНИЕ =====
        # Переставляем: [batch_size * num_frames, num_patches, embed_dim]
        x_reshaped = x.view(batch_size, -1, num_tokens // batch_size, embed_dim)

        # Применяем внимание к каждому кадру отдельно
        residual = x
        x = self.norm1_space(x)

        # TODO: Здесь нужно пространственное внимание (между патчами внутри кадра)
        # Для упрощения используем стандартное self-attention

        attn_out, _ = self.attention_space(x, x, x)
        x = residual + self.dropout(attn_out)

        # ===== 2. ВРЕМЕННОЕ ВНИМАНИЕ =====
        # Переставляем: [batch_size * num_patches, num_frames, embed_dim]
        residual = x
        x = self.norm1_time(x)

        # Меняем структуру для временного внимания
        # Код для настоящего TimeSformer более сложный

        attn_out, _ = self.attention_time(x, x, x)
        x = residual + self.dropout(attn_out)

        # ===== 3. MLP =====
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)

        return x


class TimeSformer(BaseVideoModel):
    """
    TimeSformer — трансформер для видео с разделенным вниманием.

    Как работает:
        1. Пространственное внимание: каждый кадр обрабатывается отдельно
        2. Временное внимание: обмен информацией между кадрами

    Преимущества:
        - Более эффективен, чем MViT
        - Лучшее качество на длинных видео
        - Хорошо масштабируется

    Недостатки:
        - Сложнее в реализации
        - Может переобучаться на маленьких данных
    """

    def __init__(self, num_classes, num_frames=16, frame_size=224,
                 embed_dim=384, num_heads=6, num_layers=6,
                 patch_size=(16, 16)):
        super(TimeSformer, self).__init__(num_classes, num_frames)
        self.name = "TimeSformer"
        self.embed_dim = embed_dim

        # ===== 1. PATCH EMBEDDING =====
        self.patch_embed = TimeSformerPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            patch_size=patch_size,
            frame_size=frame_size,
            num_frames=num_frames
        )

        # ===== 2. ТРАНСФОРМЕРНЫЕ БЛОКИ =====
        self.blocks = nn.ModuleList([
            TimeSformerBlock(
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
        """
        # ===== 1. Создаем патчи =====
        x = self.patch_embed(x)  # [B, num_frames * num_patches, embed_dim]

        # ===== 2. Проходим через трансформер =====
        for block in self.blocks:
            x = block(x)

        # ===== 3. Усредняем по всем патчам =====
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, embed_dim]

        # ===== 4. Классификация =====
        logits = self.classifier(x)  # [B, num_classes]

        return logits


def create_transformer(num_classes, num_frames=16):
    """Фабричная функция для создания TimeSformer модели."""
    return TimeSformer(
        num_classes=num_classes,
        num_frames=num_frames,
        embed_dim=384,
        num_heads=6,
        num_layers=4
    )