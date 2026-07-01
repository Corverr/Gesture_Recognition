import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

LOG_DIR = 'results/logs/'

if not os.path.exists(LOG_DIR):
    print("Папка results/logs/ не найдена.")
    print("Укажите правильный путь к папке с history_*.json файлами")
    exit()

history_files = [f for f in os.listdir(LOG_DIR) if f.startswith('history_') and f.endswith('.json')]

if not history_files:
    print("Файлы history_*.json не найдены в папке results/logs/")
    exit()

print("Найдено файлов:", len(history_files))

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 5)

all_train_loss = {}
all_val_loss = {}
all_val_acc = {}

for file in history_files:
    file_path = os.path.join(LOG_DIR, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        model_name = file.replace('history_', '').replace('.json', '')
        all_train_loss[model_name] = data['train_loss']
        all_val_loss[model_name] = data['val_loss']
        all_val_acc[model_name] = data['val_acc']
        print("Загружены данные для", model_name)

os.makedirs('results/plots', exist_ok=True)

plt.figure(figsize=(12, 5))
for name, losses in all_train_loss.items():
    plt.plot(losses, marker='o', label=name)
plt.title('Train Loss по эпохам', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('results/plots/train_loss.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
for name, losses in all_val_loss.items():
    plt.plot(losses, marker='s', label=name)
plt.title('Validation Loss по эпохам', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('results/plots/val_loss.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
for name, acc in all_val_acc.items():
    plt.plot(acc, marker='^', label=name)
plt.title('Validation Accuracy по эпохам', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend()
plt.grid(True)
plt.savefig('results/plots/val_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Графики построены и сохранены в папку results/plots/")