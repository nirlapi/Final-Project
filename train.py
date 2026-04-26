import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================================================
# 1. Paths
# =========================================================
DATASET_ROOT = "/home/linuxu/Nir/exported_dataset_with_augmentation"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# =========================================================
# 2. Config
# =========================================================
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 2

# =========================================================
# 3. Transforms
# =========================================================
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =========================================================
# 4. Datasets
# =========================================================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transform)

# =========================================================
# 5. DataLoaders
# =========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =========================================================
# 6. Info
# =========================================================
class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))

# Optional: inspect one batch
images, labels = next(iter(train_loader))
print("Batch images shape:", images.shape)   # expected: [B, 1, H, W]
print("Batch labels shape:", labels.shape)
from collections import Counter

train_labels = [label for _, label in train_dataset.samples]
val_labels = [label for _, label in val_dataset.samples]
test_labels = [label for _, label in test_dataset.samples]

print("Train:", Counter(train_labels))
print("Val:", Counter(val_labels))
print("Test:", Counter(test_labels))

from collections import Counter
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_label_counts = Counter(train_labels)
print("Train label counts:", train_label_counts)

total_samples = sum(train_label_counts.values())
num_classes = len(train_label_counts)

class_weights = []
for class_idx in range(num_classes):
    class_count = train_label_counts[class_idx]
    weight = total_samples / (num_classes * class_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights:", class_weights)

import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = SimpleCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(model)


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall, all_labels, all_preds
    
NUM_EPOCHS = 5000
best_val_f1 = -1
best_model_path = "best_simple_cnn.pth"

history = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall
    })

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(
        f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
        f"F1: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}"
    )
    print(
        f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
        f"F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
    )
    print("-" * 80)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)

print(f"Best model saved to: {best_model_path}")
print(f"Best validation F1: {best_val_f1:.4f}")

history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False, encoding="utf-8-sig")
history_df.head()

from sklearn.metrics import confusion_matrix, classification_report

model.load_state_dict(torch.load(best_model_path, map_location=device))

val_loss, val_acc, val_f1, val_precision, val_recall, val_true, val_pred = evaluate(
    model, val_loader, criterion, device
)

print("Validation Confusion Matrix:")
print(confusion_matrix(val_true, val_pred))
print()

print("Validation Classification Report:")
print(classification_report(val_true, val_pred, digits=4))

test_loss, test_acc, test_f1, test_precision, test_recall, test_true, test_pred = evaluate(
    model, test_loader, criterion, device
)

print("Test Confusion Matrix:")
print(confusion_matrix(test_true, test_pred))
print()

print("Test Classification Report:")
print(classification_report(test_true, test_pred, digits=4))


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_f1"], label="Train F1")
plt.plot(history_df["epoch"], history_df["val_f1"], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Train vs Validation F1")
plt.legend()
plt.show()    
 
