"""
Модель TSM (Temporal Shift Module) для распознавания жестов.
Эффективная альтернатива 3D CNN с меньшими вычислительными затратами.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

from src.models.base_model import BaseVideoModel


class TemporalShiftModule(nn.Module):
    """
    Модуль временного сдвига.

    Как работает:
        1. Принимает тензор [B, C, T, H, W]
        2. Разделяет каналы на три группы:
           - Часть каналов сдвигается назад во времени (←)
           - Часть каналов остается без изменений
           - Часть каналов сдвигается вперед во времени (→)
        3. Это позволяет сети улавливать движение

    Параметры:
        n (int): Количество каналов для сдвига (по умолчанию 1/8 от всех)
    """

    def __init__(self, n=8):
        super(TemporalShiftModule, self).__init__()
        self.n = n

    def forward(self, x):
        """
        Параметры:
            x (Tensor): [batch_size, channels, frames, height, width]

        Возвращает:
            shifted (Tensor): [batch_size, channels, frames, height, width]
        """
        batch_size, channels, frames, h, w = x.shape

        # Сколько каналов сдвигаем в каждую сторону
        # Сдвигаем 1/8 каналов назад и 1/8 вперед
        fold = max(1, channels // self.n)

        # Разделяем каналы на 3 группы
        # Группа 1: сдвигаем назад (первые fold каналов)
        out = torch.zeros_like(x)

        # Сдвиг назад: берем кадры 1..T-1 и добавляем нулевой кадр в начало
        out[:, :fold, 1:, :, :] = x[:, :fold, :-1, :, :]
        out[:, :fold, 0, :, :] = 0  # Заполняем первый кадр нулями

        # Группа 2: без изменений (средние каналы)
        out[:, fold:channels - fold, :, :, :] = x[:, fold:channels - fold, :, :, :]

        # Сдвиг вперед: берем кадры 0..T-2 и добавляем нулевой кадр в конец
        out[:, channels - fold:, :-1, :, :] = x[:, channels - fold:, 1:, :, :]
        out[:, channels - fold:, -1, :, :] = 0  # Заполняем последний кадр нулями

        return out


class TSMResNet(BaseVideoModel):
    """
    ResNet с модулями временного сдвига.

    Архитектура:
        1. ResNet18/50 обрабатывает видео
        2. В середине сети вставлены модули TemporalShift
        3. Классификатор на выходе

    Преимущества:
        - В 2-3 раза быстрее I3D
        - Почти такое же качество
        - Можно использовать предобученные веса ImageNet
    """

    def __init__(self, num_classes, num_frames=16, model_name='resnet18',
                 shift_div=8, shift_place='block'):
        """
        Конструктор.

        Параметры:
            num_classes (int): Количество классов
            num_frames (int): Количество кадров
            model_name (str): 'resnet18' или 'resnet50'
            shift_div (int): Делитель для количества сдвигаемых каналов
            shift_place (str): Где вставлять сдвиг ('block' или 'residual')
        """
        super(TSMResNet, self).__init__(num_classes, num_frames)
        self.name = f"TSM_{model_name}"
        self.shift_div = shift_div
        self.shift_place = shift_place

        # Выбираем архитектуру ResNet
        if model_name == 'resnet18':
            self.base_model = resnet18(pretrained=True)
            self.feat_dim = 512
        elif model_name == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            self.feat_dim = 2048
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        # Заменяем первый слой для работы с видео
        # Исходный conv1: [3, 7, 7] -> меняем на 3D: [3, 1, 7, 7] (не сжимаем время)
        # Но для TSM мы оставляем 2D свертку, а сдвиг делаем внутри блоков

        # Сохраняем все слои ResNet
        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool

        # Получаем блоки ResNet
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4

        # Вставляем модули сдвига в каждый блок
        self._insert_shift_modules()

        # Классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _insert_shift_modules(self):
        """Вставляет модули временного сдвига в блоки ResNet."""
        # Вставляем сдвиг в каждый блок layer1-layer4
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                # Добавляем сдвиг перед первым сверточным слоем блока
                if self.shift_place == 'block':
                    # Вставляем как отдельный модуль
                    block.conv1 = nn.Sequential(
                        TemporalShiftModule(self.shift_div),
                        block.conv1
                    )
                elif self.shift_place == 'residual':
                    # Вставляем в shortcut (для остаточных связей)
                    if block.downsample is not None:
                        block.downsample = nn.Sequential(
                            TemporalShiftModule(self.shift_div),
                            block.downsample
                        )

    @staticmethod
    def _prepare_temporal_data(x):
        """
        Преобразует данные для обработки 2D CNN.

        Вход: [B, C, T, H, W]
        Выход: [B*T, C, H, W]
        """
        batch_size, channels, frames, h, w = x.shape

        # Переставляем размерности
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        x = x.view(batch_size * frames, channels, h, w)  # [B*T, C, H, W]

        return x, batch_size, frames

    def forward(self, x):
        """
        Прямой проход.

        Параметры:
            x (Tensor): [batch_size, channels, frames, height, width]

        Возвращает:
            logits (Tensor): [batch_size, num_classes]
        """
        batch_size, channels, frames, h, w = x.shape

        # ===== 1. ПЕРВЫЙ СЛОЙ (CONV1) =====
        # Преобразуем в [B*T, C, H, W]
        x, _, _ = self._prepare_temporal_data(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Восстанавливаем временную размерность
        x = x.view(batch_size, frames, -1, h // 4, w // 4)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

        # ===== 2. RESNET БЛОКИ (со встроенными сдвигами) =====
        # Проходим по блокам, сохраняя временную размерность
        x = self._forward_block_with_time(self.layer1, x)
        x = self._forward_block_with_time(self.layer2, x)
        x = self._forward_block_with_time(self.layer3, x)
        x = self._forward_block_with_time(self.layer4, x)

        # ===== 3. КЛАССИФИКАЦИЯ =====
        # Усредняем по времени и пространству
        x = x.mean(dim=2)  # Усредняем по кадрам: [B, C, H, W]
        x = self.avgpool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]

        logits = self.classifier(x)  # [B, num_classes]

        return logits

    @staticmethod
    def _forward_block_with_time(block, x):
        """
        Пропускает данные через блок ResNet с сохранением временной размерности.
        """
        batch_size, channels, frames, h, w = x.shape

        # Преобразуем в [B*T, C, H, W] для обработки 2D свертками
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * frames, channels, h, w)

        # Проходим через блок
        x = block(x)

        # Восстанавливаем размерность
        _, channels, h, w = x.shape
        x = x.view(batch_size, frames, channels, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x


# Для совместимости с main.py
def TSM(num_classes, num_frames=16):
    """Фабричная функция для создания TSM модели."""
    return TSMResNet(num_classes, num_frames, model_name='resnet18')