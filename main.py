"""
Главный файл проекта "Распознавание жестов русского жестового языка".
Здесь запускается весь пайплайн: загрузка данных, обучение, оценка.
"""

import os
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# Импортируем наши модули
from src.dataset.slovo_dataset import SlovoDataset
from src.models.cnn_lstm import CNNLSTM
from src.models.i3d_model import I3DModel
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics


def set_seed(seed):
    """
    Фиксируем seed для воспроизводимости.
    Если seed зафиксирован, результаты будут одинаковыми при каждом запуске.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Загружает YAML-файл с настройками."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    """
    Создает загрузчики данных для обучения и валидации.

    Возвращает:
        train_loader, val_loader, num_classes, class_names
    """
    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    # Создаем тренировочный датасет
    train_dataset = SlovoDataset(
        data_dir=config['dataset']['data_dir'],
        annotation_file=config['dataset']['annotation_file'],
        split='train',
        num_frames=config['dataset']['num_frames'],
        frame_size=config['dataset']['frame_size']
    )

    # Создаем валидационный датасет
    val_dataset = SlovoDataset(
        data_dir=config['dataset']['data_dir'],
        annotation_file=config['dataset']['annotation_file'],
        split='test',
        num_frames=config['dataset']['num_frames'],
        frame_size=config['dataset']['frame_size']
    )

    # Создаем загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True  # Ускоряет передачу данных на GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

    print(f"\n Загрузчики созданы:")
    print(f"   Train: {len(train_loader)} батчей")
    print(f"   Val:   {len(val_loader)} батчей")

    return train_loader, val_loader, train_dataset.get_num_classes(), train_dataset.get_class_names()


def get_model(model_name, num_classes, num_frames):
    """
    Создает модель по имени.

    Параметры:
        model_name (str): Название модели
        num_classes (int): Количество классов
        num_frames (int): Количество кадров

    Возвращает:
        model: Модель для обучения
    """
    # Импортируем все модели
    from src.models.cnn_lstm import CNNLSTM
    from src.models.i3d_model import I3DModel
    from src.models.tsm_model import TSMResNet
    from src.models.mvit_model import MViT
    from src.models.transformer_model import TimeSformer

    models = {
        'cnn_lstm': CNNLSTM,
        'i3d': I3DModel,
        'tsm': TSMResNet,
        'mvit': MViT,
        'transformer': TimeSformer,
    }

    if model_name not in models:
        raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(models.keys())}")

    return models[model_name](num_classes, num_frames)


def main():
    """
    Главная функция, которая запускает весь пайплайн.

    Шаги:
        1. Загружаем конфиг
        2. Фиксируем seed для воспроизводимости
        3. Загружаем данные
        4. Для каждой модели:
            a. Создаем модель
            b. Обучаем
            c. Оцениваем
        5. Выводим итоговое сравнение
    """
    # ===== 1. ЗАГРУЖАЕМ КОНФИГ =====
    config = load_config('configs/default.yaml')
    print("=" * 60)
    print("РАСПОЗНАВАНИЕ ЖЕСТОВ РУССКОГО ЖЕСТОВОГО ЯЗЫКА")
    print("=" * 60)
    print(f"Конфиг загружен: {config}")

    # ===== 2. FIX SEED =====
    set_seed(config['training']['seed'])

    # ===== 3. ОПРЕДЕЛЯЕМ УСТРОЙСТВО =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ===== 4. ЗАГРУЖАЕМ ДАННЫЕ =====
    train_loader, val_loader, num_classes, class_names = create_dataloaders(config)

    # ===== 5. ОБУЧАЕМ МОДЕЛИ =====
    results = {}
    models_to_train = config['models']

    # Если в конфиге не указаны модели, используем все доступные
    if not models_to_train:
        models_to_train = ['cnn_lstm']
        print("Модели не указаны в конфиге. Используем CNN+LSTM.")

    for model_name in models_to_train:
        print("\n" + "=" * 60)
        print(f"ОБУЧЕНИЕ МОДЕЛИ: {model_name.upper()}")
        print("=" * 60)

        # Создаем модель
        model = get_model(model_name, num_classes, config['dataset']['num_frames'])
        model = model.to(device)

        # Создаем тренера
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=device
        )

        # Обучаем
        history = trainer.train(epochs=config['training']['epochs'])

        # Оцениваем
        print("\nОценка модели на тестовых данных...")
        metrics = calculate_metrics(model, val_loader, device)
        results[model_name] = metrics

        print(f"\nРезультаты для {model_name}:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")

    # ===== 6. ВЫВОДИМ ИТОГОВОЕ СРАВНЕНИЕ =====
    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    print("\n┌────────────┬──────────┬──────────┬──────────┬──────────┬────────────┐")
    print("│ Модель    │ Accuracy │ Precision│ Recall   │ F1       │ Top-5 Acc  │")
    print("├────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤")

    for model_name, metrics in results.items():
        print(f"│ {model_name:10} │ {metrics['accuracy']:.4f} │ {metrics['precision']:.4f} │ "
              f"{metrics['recall']:.4f} │ {metrics['f1']:.4f} │ {metrics['top5_accuracy']:.4f}  │")

    print("└────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘")


if __name__ == "__main__":
    main()