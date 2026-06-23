"""
Модель I3D (Inflated 3D ConvNet) для распознавания жестов.
Использует 3D-свертки для одновременной обработки пространства и времени.
"""

import torch.nn as nn
from src.models.base_model import BaseVideoModel


class I3DModel(BaseVideoModel):
    """
    I3D — одна из самых известных моделей для видео.

    Как работает:
        1. Использует 3D-свертки (ядро двигается по времени)
        2. Захватывает движение прямо на уровне пикселей
        3. Не требует отдельного LSTM для анализа времени

    Преимущества:
        - Отличное качество распознавания
        - Единая архитектура для пространства и времени
        - Можно использовать предобученные веса

    Недостатки:
        - Требует много памяти (GPU)
        - Медленнее в обучении
    """

    def __init__(self, num_classes, num_frames=16):
        super(I3DModel, self).__init__(num_classes, num_frames)
        self.name = "I3D"

        # ===== СОЗДАЕМ УПРОЩЕННУЮ 3D CNN =====
        # В реальном проекте лучше использовать готовую реализацию:
        # from pytorch_i3d import InceptionI3d
        # self.model = InceptionI3d(num_classes=num_classes)

        # Упрощенная версия для понимания:
        self.model = nn.Sequential(
            # 3D свертка: обрабатывает [каналы, время, высоту, ширину]
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Сжимаем до [256, 1, 1, 1]
            nn.Flatten(),  # Распрямляем в вектор
            nn.Linear(256, num_classes)  # Классификация
        )

    def forward(self, x):
        """
        Прямой проход.

        Параметры:
            x (Tensor): [batch_size, 3, frames, 224, 224]

        Возвращает:
            logits (Tensor): [batch_size, num_classes]
        """
        return self.model(x)