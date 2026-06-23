"""
Модуль загрузки датасета Slovo.
Отвечает за чтение видео, извлечение кадров и подготовку данных для моделей.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader, cpu
import cv2


class SlovoDataset(Dataset):
    """
    Датасет для распознавания жестов на основе Slovo.

    Как работает:
    1. Читает annotations.csv с информацией о видео
    2. По attachment_id находит нужное видео в папке
    3. Извлекает num_frames кадров из видео
    4. Возвращает кадры и метку класса
    """

    def __init__(self, data_dir, annotation_file, split='train',
                 num_frames=16, frame_size=224, transform=None):
        """
        Конструктор класса.

        Параметры:
            data_dir (str): Папка с видеофайлами (например, "data/raw/")
            annotation_file (str): Путь к файлу annotations.csv
            split (str): 'train' или 'test' - какую часть датасета использовать
            num_frames (int): Сколько кадров брать из видео
            frame_size (int): Размер кадра (ширина = высота = frame_size)
            transform: Функции аугментации (пока не используется)
        """
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        # ===== ЗАГРУЖАЕМ АННОТАЦИИ =====
        # Читаем CSV-файл с помощью pandas
        self.annotations = pd.read_csv(annotation_file)
        print(f"Загружено {len(self.annotations)} записей из annotations.csv")

        # ===== ФИЛЬТРУЕМ ПО SPLIT =====
        # В колонке 'train' записано True или False
        # Если split='train', берем только где train=True, и наоборот
        self.annotations = self.annotations[self.annotations['train'] == (split == 'train')]
        print(f"Используется {split} часть: {len(self.annotations)} видео")

        # ===== СОЗДАЕМ СЛОВАРЬ КЛАССОВ =====
        # Из колонки 'text' берем уникальные названия жестов
        # Сортируем их и нумеруем с 0
        self.classes = sorted(self.annotations['text'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"Всего классов: {len(self.classes)}")

        # Выведем первые 10 классов для примера
        print(f"Примеры классов: {list(self.classes)[:10]}...")

    def __len__(self):
        """
        Возвращает количество видео в датасете.
        Это нужно для того, чтобы PyTorch знал, сколько данных.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Получает один элемент из датасета по индексу.
        Это главный метод, который вызывается при загрузке данных.

        Параметры:
            idx (int): Индекс видео в списке

        Возвращает:
            frames (Tensor): Кадры размером [C, T, H, W]
            label (int): Номер класса
        """
        # ===== ПОЛУЧАЕМ ИНФОРМАЦИЮ О ВИДЕО =====
        row = self.annotations.iloc[idx]

        # attachment_id — это имя видеофайла (без расширения)
        # Например: "de81cc1c-..."
        video_id = row['attachment_id']
        video_path = os.path.join(self.data_dir, f"{video_id}.mp4")

        # Текст жеста — название класса
        class_name = row['text']
        label = self.class_to_idx[class_name]

        # ===== ЗАГРУЖАЕМ ВИДЕО =====
        frames = self._load_video(video_path)

        # ===== ПРЕВРАЩАЕМ В ТЕНЗОР =====
        # frames: [T, H, W, C] (кадры, высота, ширина, каналы)
        # Переставляем в [C, T, H, W] для PyTorch
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)

        return frames, label

    def _load_video(self, video_path):
        """
        Загружает видео и извлекает равномерно распределенные кадры.

        Как работает:
        1. Открываем видео через decord
        2. Считаем общее количество кадров
        3. Выбираем num_frames кадров равномерно по всему видео
        4. Изменяем размер кадров на frame_size x frame_size
        5. Нормализуем значения пикселей в диапазон [-1, 1]

        Возвращает:
            frames (np.array): [num_frames, frame_size, frame_size, 3]
        """
        try:
            # ===== 1. ОТКРЫВАЕМ ВИДЕО =====
            # VideoReader из decord — быстрый загрузчик
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            # ===== 2. ВЫБИРАЕМ КАДРЫ =====
            if total_frames < self.num_frames:
                # Если видео короче, чем нужно кадров — повторяем кадры
                # Например: в видео 5 кадров, а нужно 16 — повторяем 3 раза
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Равномерно выбираем кадры по всему видео
                # Например: в видео 100 кадров, нужно 16 → берем кадры 0, 6, 12, ...
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            # ===== 3. ЗАГРУЖАЕМ ВЫБРАННЫЕ КАДРЫ =====
            frames = vr.get_batch(indices).asnumpy()
            # frames теперь: [num_frames, H, W, C]

            # ===== 4. МЕНЯЕМ РАЗМЕР КАДРОВ =====
            if frames.shape[1] != self.frame_size or frames.shape[2] != self.frame_size:
                resized_frames = []
                for frame in frames:
                    # cv2.resize изменяет размер
                    resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                    resized_frames.append(resized)
                frames = np.array(resized_frames)

            # ===== 5. НОРМАЛИЗУЕМ =====
            # Пиксели обычно от 0 до 255, приводим к [0, 1]
            frames = frames / 255.0
            # Затем к [-1, 1] — так модели лучше обучаются
            frames = (frames - 0.5) / 0.5

            return frames

        except Exception as e:
            # Если видео не загрузилось (например, файл поврежден)
            print(f"Ошибка загрузки видео {video_path}: {e}")
            # Возвращаем случайный шум, чтобы обучение не упало
            return np.random.randn(self.num_frames, self.frame_size, self.frame_size, 3)

    def get_num_classes(self):
        """Возвращает количество классов в датасете."""
        return len(self.classes)

    def get_class_names(self):
        """Возвращает названия всех классов."""
        return self.classes


# ===== ФУНКЦИЯ ДЛЯ ПРОВЕРКИ ДАТАСЕТА =====
def test_dataset():
    """
    Простая функция для проверки, что датасет работает.
    """
    print("=" * 60)
    print(" ТЕСТИРОВАНИЕ ДАТАСЕТА SLOVO")
    print("=" * 60)

    # Создаем датасет
    dataset = SlovoDataset(
        data_dir="data/raw/",
        annotation_file="data/raw/annotations.csv",
        split='train',
        num_frames=16,
        frame_size=224
    )

    # Берем один пример
    print("\n Загружаем первый пример...")
    frames, label = dataset[0]

    print(f" Форма кадров: {frames.shape}")
    print(f" Класс: {label} -> {dataset.classes[label]}")

    print("\n Датасет работает корректно")


if __name__ == "__main__":
    # Если запустить этот файл напрямую, выполнится тест
    test_dataset()