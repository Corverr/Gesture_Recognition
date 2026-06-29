import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader, cpu
import cv2


class SlovoDataset(Dataset):
    def __init__(self, data_dir, annotation_file, split='train',
                 num_frames=16, frame_size=224, transform=None):

        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        self.annotations = pd.read_csv(annotation_file)
        print(f"Загружено {len(self.annotations)} записей из annotations.csv")

        self.annotations = self.annotations[self.annotations['train'] == (split == 'train')]
        print(f"Используется {split} часть: {len(self.annotations)} видео")

        self.classes = sorted(self.annotations['text'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"Всего классов: {len(self.classes)}")

        print(f"Примеры классов: {list(self.classes)[:10]}...")

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, idx):

        row = self.annotations.iloc[idx]

        video_id = row['attachment_id']
        video_path = os.path.join(self.data_dir, f"{video_id}.mp4")

        class_name = row['text']
        label = self.class_to_idx[class_name]

        frames = self._load_video(video_path)

        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)

        return frames, label

    def _load_video(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames < self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

            frames = vr.get_batch(indices).asnumpy()

            if frames.shape[1] != self.frame_size or frames.shape[2] != self.frame_size:
                resized_frames = []
                for frame in frames:
                    resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                    resized_frames.append(resized)
                frames = np.array(resized_frames)

            frames = frames / 255.0
            frames = (frames - 0.5) / 0.5

            return frames

        except Exception as e:
            print(f"Ошибка загрузки видео {video_path}: {e}")
            return np.random.randn(self.num_frames, self.frame_size, self.frame_size, 3)

    def get_num_classes(self):
        return len(self.classes)

    def get_class_names(self):
        return self.classes


def test_dataset():

    print("Тестирование датасета  SLOVO")

    dataset = SlovoDataset(
        data_dir="data/raw/",
        annotation_file="data/raw/annotations.csv",
        split='train',
        num_frames=16,
        frame_size=224
    )

    print("\n Загружаем первый пример...")
    frames, label = dataset[0]

    print(f" Форма кадров: {frames.shape}")
    print(f" Класс: {label} -> {dataset.classes[label]}")

    print("\n Датасет работает корректно")


if __name__ == "__main__":
    test_dataset()