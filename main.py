import random

import numpy as np
import torch
import yaml
from src.models.simple_cnn import SimpleCNN
from torch.utils.data import DataLoader

from src.dataset.slovo_dataset import SlovoDataset
from src.evaluation.metrics import calculate_metrics
from src.models.cnn_lstm import CNNLSTM
from src.models.i3d_model import I3DModel
from src.models.mvit_model import MViT
from src.models.transformer_model import TimeSformer
from src.training.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_dataloaders(config):
    print("ЗАГРУЗКА ДАННЫХ")

    train_dataset = SlovoDataset(
        data_dir=config['dataset']['data_dir'],
        annotation_file=config['dataset']['annotation_file'],
        split='train',
        num_frames=config['dataset']['num_frames'],
        frame_size=config['dataset']['frame_size']
    )

    val_dataset = SlovoDataset(
        data_dir=config['dataset']['data_dir'],
        annotation_file=config['dataset']['annotation_file'],
        split='test',
        num_frames=config['dataset']['num_frames'],
        frame_size=config['dataset']['frame_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=config['dataset']['num_workers'],
        pin_memory=True
    )

    print(f"\nЗагрузчики созданы:")
    print(f"   Train: {len(train_loader)} батчей")
    print(f"   Val:   {len(val_loader)} батчей")

    return train_loader, val_loader, train_dataset.get_num_classes(), train_dataset.get_class_names()


def get_model(model_name, num_classes, num_frames):
    models = {
        'cnn_lstm': CNNLSTM,
        'i3d': I3DModel,
        'simple_cnn': SimpleCNN,
        'mvit': MViT,
        'transformer': TimeSformer,
    }

    if model_name not in models:
        raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(models.keys())}")

    return models[model_name](num_classes, num_frames)


def main():
    config = load_config('configs/default.yaml')

    print("РАСПОЗНАВАНИЕ ЖЕСТОВ РУССКОГО ЖЕСТОВОГО ЯЗЫКА")

    set_seed(config['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуется устройство: {device}")

    train_loader, val_loader, num_classes, class_names = create_dataloaders(config)

    models_to_train = config.get('models', ['cnn_lstm'])

    results = {}

    for model_name in models_to_train:
        print(f"ОБУЧЕНИЕ МОДЕЛИ: {model_name.upper()}")

        model = get_model(model_name, num_classes, config['dataset']['num_frames'])
        model = model.to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config['training'],
            device=device
        )

        trainer.train(epochs=config['training']['epochs'])

        print("\nОценка модели на тестовых данных...")
        metrics = calculate_metrics(model, val_loader, device)
        results[model_name] = metrics

        print(f"\nРезультаты для {model_name}:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")

    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")

    print("\n┌────────────┬──────────┬──────────┬──────────┬──────────┬────────────┐")
    print("│ Модель    │ Accuracy │ Precision│ Recall   │ F1       │ Top-5 Acc  │")
    print("├────────────┼──────────┼──────────┼──────────┼──────────┼────────────┤")

    for model_name, metrics in results.items():
        print(f"│ {model_name:10} │ {metrics['accuracy']:.4f} │ {metrics['precision']:.4f} │ "
              f"{metrics['recall']:.4f} │ {metrics['f1']:.4f} │ {metrics['top5_accuracy']:.4f}  │")

    print("└────────────┴──────────┴──────────┴──────────┴──────────┴────────────┘")


if __name__ == "__main__":
    main()