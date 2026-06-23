"""
Класс для обучения моделей.
Содержит логику тренировки, валидации и сохранения моделей.
"""

import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Тренер для обучения моделей.

    Что делает:
        1. Принимает модель, данные и настройки
        2. Запускает цикл обучения
        3. Каждую эпоху проверяет качество на валидации
        4. Сохраняет лучшую модель
        5. Логирует результаты
    """

    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Конструктор.

        Параметры:
            model: Модель для обучения
            train_loader (DataLoader): Загрузчик тренировочных данных
            val_loader (DataLoader): Загрузчик валидационных данных
            config (dict): Настройки обучения
            device: cuda или cpu
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Функция потерь — CrossEntropyLoss для классификации
        # Она сравнивает предсказанные вероятности с правильным классом
        self.criterion = nn.CrossEntropyLoss()

        # Оптимизатор — Adam (адаптивный метод)
        # Он обновляет веса модели, чтобы уменьшить ошибку
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5  # L2-регуляризация
        )

        # Планировщик скорости — уменьшает learning_rate
        # Это помогает модели точнее "приземлиться" в минимум
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )

        # Лучшая точность на валидации
        self.best_val_acc = 0

        # История обучения (для графиков)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        """
        Обучение одной эпохи.

        Что происходит:
            1. Проходим по всем батчам в train_loader
            2. Для каждого батча:
                a. Подаем видео в модель → получаем предсказания
                b. Считаем ошибку между предсказаниями и правильными ответами
                c. Вычисляем градиенты (обратное распространение)
                d. Обновляем веса модели
            3. Возвращаем среднюю ошибку за эпоху

        Возвращает:
            avg_loss (float): Средняя ошибка за эпоху
        """
        self.model.train()
        total_loss = 0

        # tqdm — прогресс-бар
        pbar = tqdm(self.train_loader, desc="Обучение")
        for videos, labels in pbar:
            # Отправляем данные на GPU (если есть)
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Обнуляем градиенты (важно!)
            self.optimizer.zero_grad()

            # Прямой проход: подаем видео → получаем предсказания
            outputs = self.model(videos)

            # Считаем ошибку
            loss = self.criterion(outputs, labels)

            # Обратный проход: вычисляем градиенты
            loss.backward()

            # Обновляем веса
            self.optimizer.step()

            # Запоминаем ошибку
            total_loss += loss.item()

            # Обновляем прогресс-бар
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Валидация модели на отложенных данных.

        Что происходит:
            1. Проходим по всем батчам в val_loader
            2. Для каждого батча получаем предсказания (без вычисления градиентов!)
            3. Считаем ошибку и точность
            4. Возвращаем среднюю ошибку и точность

        Возвращает:
            avg_loss (float): Средняя ошибка на валидации
            accuracy (float): Точность в процентах
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Отключаем вычисление градиентов (экономит память)
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Валидация")
            for videos, labels in pbar:
                videos = videos.to(self.device)
                labels = labels.to(self.device)

                # Прямой проход
                outputs = self.model(videos)

                # Ошибка
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Точность: выбираем класс с наибольшей вероятностью
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return total_loss / len(self.val_loader), accuracy

    def train(self, epochs):
        """
        Полный цикл обучения.

        Параметры:
            epochs (int): Количество эпох

        Возвращает:
            history (dict): История обучения
        """
        print(f"\nНачинаем обучение модели {self.model.get_name()}")
        print(f"Количество параметров: {self.model.get_num_parameters():,}")

        for epoch in range(epochs):
            print(f"\n{'=' * 50}")
            print(f"Эпоха {epoch + 1}/{epochs}")
            print(f"{'=' * 50}")

            # 1. Обучение
            train_loss = self.train_epoch()

            # 2. Валидация
            val_loss, val_acc = self.validate()

            # 3. Запоминаем историю
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 4. Обновляем learning_rate
            self.scheduler.step()

            # 5. Выводим результаты
            print(f"\nРезультаты эпохи {epoch + 1}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Val Acc:    {val_acc:.2f}%")

            # 6. Сохраняем лучшую модель
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"best_{self.model.get_name()}.pth")
                print(f"   Сохранена лучшая модель (точность: {val_acc:.2f}%)")

        # Сохраняем историю в файл
        self.save_history()

        print(f"\nОбучение завершено! Лучшая точность: {self.best_val_acc:.2f}%")
        return self.history

    def save_model(self, filename):
        """
        Сохраняет модель в файл.

        Параметры:
            filename (str): Имя файла
        """
        os.makedirs("results/models", exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_name': self.model.get_name(),
            'config': self.config
        }, os.path.join("results/models", filename))
        print(f"Модель сохранена: {filename}")

    def save_history(self):
        """
        Сохраняет историю обучения в JSON-файл.
        Это нужно для построения графиков.
        """
        os.makedirs("results/logs", exist_ok=True)
        filename = f"history_{self.model.get_name()}.json"
        with open(os.path.join("results/logs", filename), 'w') as f:
            json.dump(self.history, f)
        print(f"История сохранена: {filename}")