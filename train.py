import os
from pathlib import Path
from collections import Counter
from PIL import Image

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# =========================================================
# 1. Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_CSV = DATA_DIR / "train_split.csv"
TRAIN_AUGMENTED_CSV = DATA_DIR / "c" / "New_Augmentation" / "train_manifest_augmented.csv"
VAL_CSV = DATA_DIR / "val_split.csv"
TEST_CSV = DATA_DIR / "test_split.csv"
NEW_AUGMENTATION_DIR = DATA_DIR / "c" / "New_Augmentation"

# =========================================================
# 2. Config
# =========================================================
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
PRED_THRESHOLD = 0.6

# =========================================================
# 3. Transforms
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =========================================================
# 4. Dataset
# =========================================================
class ImageCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None, clean_conflicts=False):
        csv_paths = [csv_path] if isinstance(csv_path, (str, Path)) else list(csv_path)
        self.csv_paths = [Path(path) for path in csv_paths]
        self.transform = transform
        self.clean_conflicts = clean_conflicts
        image_column_candidates = ["resized_file_path", "file_path"]
        frames = []

        for csv_file in self.csv_paths:
            data = pd.read_csv(csv_file)

            if "pain_label" not in data.columns:
                raise KeyError(f"{csv_file} must contain a 'pain_label' column.")

            image_columns = [col for col in image_column_candidates if col in data.columns]
            if not image_columns:
                raise KeyError(
                    f"{csv_file} must contain one of {image_column_candidates} columns with image paths."
                )

            data = data[data["pain_label"].notna()].copy()

            # Keep only generated augmented rows from the augmented manifest.
            if csv_file.name == TRAIN_AUGMENTED_CSV.name and "is_augmented" in data.columns:
                data = data[pd.to_numeric(data["is_augmented"], errors="coerce") == 1].copy()

            for column in image_columns:
                data[column] = data[column].fillna("").astype(str)

            frames.append(data)

        self.data = pd.concat(frames, ignore_index=True)
        self.image_columns = [col for col in image_column_candidates if col in self.data.columns]

        self.data["_resolved_image_path"] = self.data.apply(self._resolve_row_image_path, axis=1)
        self.data = self.data[self.data["_resolved_image_path"].apply(Path.exists)].copy()

        raw_labels = sorted(pd.to_numeric(self.data["pain_label"], errors="coerce").dropna().unique().tolist())
        if not raw_labels:
            joined_paths = ", ".join(str(path) for path in self.csv_paths)
            raise ValueError(f"{joined_paths} does not contain any valid pain_label values.")

        self.label_to_index = {label: index for index, label in enumerate(raw_labels)}
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}
        self.data["pain_label"] = pd.to_numeric(self.data["pain_label"], errors="coerce").map(self.label_to_index)
        self.data = self.data[self.data["pain_label"].notna()].copy()
        self.data["pain_label"] = self.data["pain_label"].astype(int)

        if self.clean_conflicts:
            # Keep one label per image path; if duplicates conflict, use the modal label for that path.
            path_mode_label = self.data.groupby("_resolved_image_path")["pain_label"].agg(
                lambda values: values.mode().iloc[0]
            )
            self.data["pain_label"] = self.data["_resolved_image_path"].map(path_mode_label).astype(int)

            # Keep one row per resolved image path so augmented images are added once each.
            self.data = self.data.drop_duplicates(subset=["_resolved_image_path"], keep="first").copy()

        self.samples = [
            (row["_resolved_image_path"], int(row["pain_label"]))
            for _, row in self.data.iterrows()
        ]
        self.classes = [str(label) for label in raw_labels]

    def _resolve_image_path(self, image_path):
        image_path_str = str(image_path).strip()
        path = Path(image_path_str)
        if path.is_absolute():
            return path

        # Support both .../train/... and .../Train/... paths on case-sensitive filesystems.
        # Also support mapping Data/New_Augmentation to Data/c/New_Augmentation
        path_variants = [image_path_str]
        if "exported_dataset_with_augmentation/train/" in image_path_str:
            path_variants.append(
                image_path_str.replace("exported_dataset_with_augmentation/train/", "exported_dataset_with_augmentation/Train/")
            )
        if "Data/New_Augmentation/" in image_path_str:
            path_variants.append(
                image_path_str.replace("Data/New_Augmentation/", "Data/c/New_Augmentation/")
            )

        candidates = []
        for path_variant in path_variants:
            variant_path = Path(path_variant)
            candidates.extend([
                PROJECT_ROOT / variant_path,
                PROJECT_ROOT.parent / variant_path,
                DATA_DIR / variant_path,
                NEW_AUGMENTATION_DIR / variant_path,
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        return (PROJECT_ROOT / path).resolve()

    def _resolve_row_image_path(self, row):
        for column in self.image_columns:
            image_path = str(row.get(column, "")).strip()
            if not image_path:
                continue

            resolved_path = self._resolve_image_path(image_path)
            if resolved_path.exists():
                return resolved_path

        fallback_value = str(row.get(self.image_columns[0], "")).strip()
        return self._resolve_image_path(fallback_value)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

# =========================================================
# 4. Datasets
# =========================================================
train_dataset = ImageCSVDataset([TRAIN_CSV, TRAIN_AUGMENTED_CSV], transform=train_transform, clean_conflicts=True)
val_dataset = ImageCSVDataset(VAL_CSV, transform=eval_transform, clean_conflicts=False)
test_dataset = ImageCSVDataset(TEST_CSV, transform=eval_transform, clean_conflicts=False)

# =========================================================
# 5. Labels and sampling
# =========================================================
train_labels = [label for _, label in train_dataset.samples]
val_labels = [label for _, label in val_dataset.samples]
test_labels = [label for _, label in test_dataset.samples]

train_label_counts = Counter(train_labels)
val_label_counts = Counter(val_labels)
test_label_counts = Counter(test_labels)

num_present_classes = len(train_label_counts)
num_train_samples = len(train_labels)
sample_weights = [
    num_train_samples / (num_present_classes * train_label_counts[label])
    for label in train_labels
]
train_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

# =========================================================
# 6. DataLoaders
# =========================================================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
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
# 7. Info
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
print("Batch images shape:", images.shape)   # expected: [B, 3, H, W]
print("Batch labels shape:", labels.shape)
print("Train:", train_label_counts)
print("Val:", val_label_counts)
print("Test:", test_label_counts)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("Train label counts:", train_label_counts)

total_samples = sum(train_label_counts.values())
num_classes = len(class_names)

class_weights = []
for class_idx in range(num_classes):
    class_count = train_label_counts.get(class_idx, 0)
    weight = 0.0 if class_count == 0 else total_samples / (num_classes * class_count)
    class_weights.append(weight)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Class weights:", class_weights)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=9, padding=4),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 256 -> 128
            nn.Dropout2d(p=0.28),

            nn.Conv2d(24, 48, kernel_size=7, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64
            nn.Dropout2d(p=0.28),

            nn.Conv2d(48, 96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32
            nn.Dropout2d(p=0.28),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 32 -> 16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128 * (img_size // 16) * (img_size // 16), 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = SimpleCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.7,
    patience=12,
    verbose=True
)

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

        probs = torch.softmax(outputs, dim=1)
        preds = (probs[:, 1] >= PRED_THRESHOLD).long()

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

            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= PRED_THRESHOLD).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall, all_labels, all_preds
    
NUM_EPOCHS = 1000
best_val_acc = -1
best_val_f1 = -1
best_model_path = "best_simple_cnn.pth"
best_model_weights = None
early_stop_patience = 150
epochs_without_improvement = 0

history = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_acc, val_f1, val_precision, val_recall, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    scheduler.step(val_loss)

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
    print(f"LR    | {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 80)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_model_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(best_model_weights, best_model_path)
        print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stop_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs without improvement in validation accuracy.")
        break

print(f"Best model saved to: {best_model_path}")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"Best validation F1: {best_val_f1:.4f}")

history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False, encoding="utf-8-sig")
history_df.head()

from sklearn.metrics import confusion_matrix, classification_report

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
else:
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
 
