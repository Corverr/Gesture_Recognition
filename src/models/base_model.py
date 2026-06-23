"""
Базовый класс для всех моделей компьютерного зрения для видео.
"""

import torch.nn as nn


class BaseVideoModel(nn.Module):
    """
    Базовый класс для всех моделей.

    Все модели в нашем проекте наследуются от этого класса.
    Это гарантирует, что у всех моделей будет одинаковый интерфейс.
    """

    def __init__(self, num_classes, num_frames=16):
        """
        Конструктор базового класса.

        Параметры:
            num_classes (int): Количество классов для классификации (1000 в Slovo)
            num_frames (int): Количество кадров, которое получает модель
        """
        super(BaseVideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.name = "BaseModel"  # Будет переопределено в дочерних классах

    def forward(self, x):
        """
        Прямой проход через модель.

        Параметры:
            x (Tensor): Входные данные [batch_size, channels, frames, height, width]

        Возвращает:
            logits (Tensor): Выход модели [batch_size, num_classes]
        """
        raise NotImplementedError("Каждая модель должна реализовать метод forward()")

    def get_name(self):
        """Возвращает название модели."""
        return self.name

    def get_num_parameters(self):
        """Возвращает количество параметров модели."""
        return sum(p.numel() for p in self.parameters())