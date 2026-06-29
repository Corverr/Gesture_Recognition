import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(model, dataloader, device):

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print("Вычисление метрик...")

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)

            outputs = model(videos)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    top5 = top_k_accuracy(all_probs, all_labels, k=5)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top5_accuracy': top5
    }


def top_k_accuracy(probabilities, labels, k=5):

    top_k_preds = np.argsort(-probabilities, axis=1)[:, :k]

    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1

    return correct / len(labels)