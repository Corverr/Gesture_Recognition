"""
Модуль для подсчета метрик качества моделей.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(model, dataloader, device):
    """
    Рассчитывает все основные метрики для модели.

    Какие метрики:
        1. Accuracy (Точность): доля правильных ответов
        2. Precision (Полнота): из всех предсказанных — сколько правильных
        3. Recall: из всех правильных — сколько найдено
        4. F1: гармоническое среднее Precision и Recall
        5. Top-5 Accuracy: правильный ответ в топ-5 предсказаний

    Параметры:
        model: Обученная модель
        dataloader: Загрузчик с тестовыми данными
        device: cuda или cpu

    Возвращает:
        dict: Словарь с метриками
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Вычисление метрик...")

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)

            # Получаем предсказания
            outputs = model(videos)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Сохраняем для подсчета метрик
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Превращаем в массивы numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ===== 1. ТОЧНОСТЬ (Accuracy) =====
    # Процент правильных ответов
    accuracy = accuracy_score(all_labels, all_preds)

    # ===== 2. ПОЛНОТА (Precision) и ТОЧНОСТЬ (Recall) =====
    # weighted — усредняем по всем классам с учетом их размера
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    # ===== 3. F1-MEASURE =====
    # Гармоническое среднее Precision и Recall
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # ===== 4. TOP-5 ACCURACY =====
    # Правильный ответ есть в 5 самых вероятных классах?
    top5 = top_k_accuracy(all_probs, all_labels, k=5)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top5_accuracy': top5
    }


def top_k_accuracy(probabilities, labels, k=5):
    """
    Вычисляет Top-K Accuracy.

    Параметры:
        probabilities (np.array): Вероятности для каждого класса [N, num_classes]
        labels (np.array): Правильные ответы [N]
        k (int): Сколько топ-классов проверять

    Возвращает:
        float: Top-K Accuracy
    """
    # Берем индексы k самых вероятных классов
    top_k_preds = np.argsort(-probabilities, axis=1)[:, :k]

    # Проверяем, есть ли правильный ответ среди top-k
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1

    return correct / len(labels)