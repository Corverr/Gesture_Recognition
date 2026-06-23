"""
Модель CNN + LSTM для распознавания жестов.
"""

import torch.nn as nn
from torchvision.models import resnet18
from src.models.base_model import BaseVideoModel


class CNNLSTM(BaseVideoModel):
    """
    Гибридная модель: CNN (ResNet18) + LSTM.

    Архитектура:
        1. ResNet18 обрабатывает каждый кадр → признаки (512 чисел)
        2. LSTM анализирует последовательность признаков
        3. Классификатор выдает вероятности классов

    Преимущества:
        - Простая и понятная архитектура
        - Хорошо работает для видео средней длины
        - Можно использовать предобученный ResNet18

    Недостатки:
        - Не улавливает движение на уровне пикселей
        - LSTM может забывать длинные последовательности
    """

    def __init__(self, num_classes, num_frames=16, hidden_size=256, num_layers=2):
        """
        Конструктор модели.

        Параметры:
            num_classes (int): Количество классов
            num_frames (int): Количество кадров
            hidden_size (int): Размер скрытого состояния LSTM
            num_layers (int): Количество слоев LSTM
        """
        super(CNNLSTM, self).__init__(num_classes, num_frames)
        self.name = "CNN+LSTM"

        # ===== 1. CNN ДЛЯ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ =====
        # Используем предобученный ResNet18 (хорошо распознает объекты)
        self.cnn = resnet18(pretrained=True)
        # Убираем последний слой классификации (fc) — оставляем только признаки
        # Вместо 1000 классов мы получим 512 признаков
        self.cnn.fc = nn.Identity()

        # Замораживаем веса CNN (не обучаем их сначала)
        # Это ускоряет обучение, так как мы не переобучаем всю сеть
        for param in self.cnn.parameters():
            param.requires_grad = False

        # ===== 2. LSTM ДЛЯ АНАЛИЗА ПОСЛЕДОВАТЕЛЬНОСТИ =====
        # Входной размер: 512 признаков из CNN
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Вход: [batch, seq_len, features]
            bidirectional=True  # Двунаправленный LSTM (смотрит вперед и назад)
        )

        # ===== 3. КЛАССИФИКАТОР =====
        # Вход: hidden_size * 2 (так как bidirectional)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Отключаем 30% нейронов для регуляризации
            nn.Linear(hidden_size * 2, 256),  # Полносвязный слой
            nn.ReLU(),  # Функция активации
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # Выход: количество классов
        )

    def forward(self, x):
        """
        Прямой проход через модель.

        Параметры:
            x (Tensor): [batch_size, channels, frames, height, width]

        Возвращает:
            logits (Tensor): [batch_size, num_classes]
        """
        batch_size, channels, frames, h, w = x.shape

        # ===== 1. ПОДГОТАВЛИВАЕМ ДАННЫЕ ДЛЯ CNN =====
        # Исходный размер: [B, C, T, H, W]
        # Нужно переставить в [B*T, C, H, W], чтобы CNN обрабатывал каждый кадр
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(batch_size * frames, channels, h, w)  # [B*T, C, H, W]

        # ===== 2. ИЗВЛЕКАЕМ ПРИЗНАКИ ЧЕРЕЗ CNN =====
        features = self.cnn(x)  # [B*T, 512]

        # ===== 3. ВОССТАНАВЛИВАЕМ ВРЕМЕННУЮ РАЗМЕРНОСТЬ =====
        features = features.view(batch_size, frames, -1)  # [B, T, 512]

        # ===== 4. АНАЛИЗИРУЕМ ПОСЛЕДОВАТЕЛЬНОСТЬ ЧЕРЕЗ LSTM =====
        lstm_out, _ = self.lstm(features)  # [B, T, hidden*2]

        # Берем только последний выход LSTM
        # Это самый "умный" момент, когда LSTM уже "понял" весь жест
        last_out = lstm_out[:, -1, :]  # [B, hidden*2]

        # ===== 5. КЛАССИФИКАЦИЯ =====
        logits = self.classifier(last_out)  # [B, num_classes]

        return logits