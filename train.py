import os
import random
from pathlib import Path
import sys
from collections import Counter
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
NEW_AUGMENTATION_DIR = DATA_DIR / "c" / "New_Augmentation_2"
TRAIN_CSV = DATA_DIR / "train_split.csv"
TRAIN_AUGMENTED_CSV = NEW_AUGMENTATION_DIR / "train_manifest_augmented.csv"
VAL_CSV = DATA_DIR / "val_split.csv"
TEST_CSV = DATA_DIR / "test_split.csv"

IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

DROPOUT_SPATIAL = 0.08
DROPOUT_CLASSIFIER = 0.08
ACTIVATION = "ReLU"
LABEL_SMOOTHING = 0.0
# Increased base LR to speed up initial convergence; can be tuned later
LR = 2e-3
# Disable weight decay initially to avoid over-regularizing while using sampler
WEIGHT_DECAY = 0.0

SINGLE_BATCH_DEBUG = False
PRED_THRESHOLD = 0.5

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Also support mapping augmentation paths to the current augmentation directory.
        path_variants = [image_path_str]
        if "exported_dataset_with_augmentation/train/" in image_path_str:
            path_variants.append(
                image_path_str.replace("exported_dataset_with_augmentation/train/", "exported_dataset_with_augmentation/Train/")
            )
        if "Data/New_Augmentation_2/" in image_path_str:
            path_variants.append(
                image_path_str.replace("Data/New_Augmentation_2/", "Data/c/New_Augmentation_2/")
            )
        elif "Data/New_Augmentation/" in image_path_str:
            path_variants.append(
                image_path_str.replace("Data/New_Augmentation/", "Data/c/New_Augmentation_2/")
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

# Ensure validation and test datasets use the same label mapping as the training dataset
try:
    val_index_to_label = val_dataset.index_to_label
    test_index_to_label = test_dataset.index_to_label
    train_label_to_index = train_dataset.label_to_index

    new_val_samples = []
    for path, lbl in val_dataset.samples:
        raw_label = val_index_to_label.get(lbl, None)
        if raw_label in train_label_to_index:
            new_val_samples.append((path, train_label_to_index[raw_label]))
    val_dataset.samples = new_val_samples

    new_test_samples = []
    for path, lbl in test_dataset.samples:
        raw_label = test_index_to_label.get(lbl, None)
        if raw_label in train_label_to_index:
            new_test_samples.append((path, train_label_to_index[raw_label]))
    test_dataset.samples = new_test_samples
except Exception:
    # If the dataset implementation does not expose index_to_label/label_to_index,
    # skip remapping and assume labels already align.
    pass

# =========================================================
# 5. Labels and sampling
# =========================================================
train_labels = [label for _, label in train_dataset.samples]
val_labels = [label for _, label in val_dataset.samples]
test_labels = [label for _, label in test_dataset.samples]

train_label_counts = Counter(train_labels)
val_label_counts = Counter(val_labels)
test_label_counts = Counter(test_labels)

# Create a WeightedRandomSampler to balance classes during training
num_present_classes = len(train_label_counts)
num_train_samples = len(train_labels)
sample_weights = [
    num_train_samples / (num_present_classes * train_label_counts[label])
    for label in train_labels
]
train_sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
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

if SINGLE_BATCH_DEBUG:
    device = torch.device("cpu")
else:
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

class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("Class weights:", class_weights)

# =========================================================
# 8. Model
# =========================================================
class LabelSmoothingLoss(nn.Module):
    """Custom loss with label smoothing and class weights (matches hyperparameter_search.py)"""
    def __init__(self, num_classes, smoothing=0.05, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, logits, targets):
        if self.smoothing == 0:
            return self.criterion(logits, targets).mean()
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        loss = torch.sum(-true_dist * torch.nn.functional.log_softmax(logits, dim=1), dim=1)
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_spatial=DROPOUT_SPATIAL, dropout_classifier=DROPOUT_CLASSIFIER, activation="ReLU"):
        super().__init__()

        if activation == "ReLU":
            act_fn = nn.ReLU
        elif activation == "GELU":
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        # Deeper architecture with smaller kernels to capture finer thermal patterns
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            act_fn(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(512, 512),
            act_fn(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(512, 256),
            act_fn(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(256, 128),
            act_fn(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        
model = SimpleCNN(
    num_classes=num_classes,
    dropout_spatial=DROPOUT_SPATIAL,
    dropout_classifier=DROPOUT_CLASSIFIER,
    activation=ACTIVATION
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# When using a sampler we do not pass class weights to the loss (sampler balances classes)
criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=LABEL_SMOOTHING, weight=None)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.7,
    patience=10,
    verbose=True
)

print(model)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

# =========================================================
# Quick single-batch debug to inspect logits/probs/labels
# Enable only when SINGLE_BATCH_DEBUG=1
# =========================================================
if SINGLE_BATCH_DEBUG:
    try:
        print("Running single-batch debug...")
        batch_images, batch_labels = next(iter(train_loader))
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        model.eval()
        with torch.no_grad():
            batch_outputs = model(batch_images)
            batch_probs = torch.nn.functional.softmax(batch_outputs, dim=1)
            batch_preds = (batch_probs[:, 1] >= PRED_THRESHOLD).long()

        print("Logits mean/std/min/max:", batch_outputs.mean().item(), batch_outputs.std().item(), batch_outputs.min().item(), batch_outputs.max().item())
        print("Probs class0 mean/std:", batch_probs[:,0].mean().item(), batch_probs[:,0].std().item())
        print("Probs class1 mean/std:", batch_probs[:,1].mean().item(), batch_probs[:,1].std().item())
        unique_preds, counts = torch.unique(batch_preds, return_counts=True)
        print("Pred counts:", dict(zip(unique_preds.tolist(), counts.tolist())))
        unique_labels, lcounts = torch.unique(batch_labels, return_counts=True)
        print("Label counts:", dict(zip(unique_labels.tolist(), lcounts.tolist())))
        print("Sample probs (first 8):")
        for i in range(min(8, batch_probs.size(0))):
            print(f"i={i} prob0={batch_probs[i,0].item():.4f} prob1={batch_probs[i,1].item():.4f} pred={batch_preds[i].item()} label={batch_labels[i].item()}")
        print("End single-batch debug")
        sys.exit(0)
    except Exception as e:
        print("Single-batch debug failed:", e)

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
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            pos_probs = probs[:, 1]

            all_probs.extend(pos_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    # If there are no samples, return defaults
    if probs_arr.size == 0:
        return epoch_loss, 0.0, 0.0, 0.0, 0.0, labels_arr.tolist(), [], 0.5

    # Threshold sweep on validation to choose operating point maximizing balanced accuracy
    best_thresh = 0.5
    best_bal = -1.0
    best_metrics = None
    thresh_candidates = np.linspace(0.1, 0.9, 33)
    for t in thresh_candidates:
        preds_t = (probs_arr >= t).astype(int)
        bal = balanced_accuracy_score(labels_arr, preds_t)
        if bal > best_bal:
            best_bal = bal
            best_thresh = float(t)
            best_metrics = {
                "f1": float(f1_score(labels_arr, preds_t, average="binary", zero_division=0)),
                "precision": float(precision_score(labels_arr, preds_t, average="binary", zero_division=0)),
                "recall": float(recall_score(labels_arr, preds_t, average="binary", zero_division=0)),
            }

    best_preds = (probs_arr >= best_thresh).astype(int).tolist()

    return epoch_loss, float(best_bal), best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], labels_arr.tolist(), best_preds, best_thresh
    if probs_arr.size == 0:
        return epoch_loss, 0.0, 0.0, 0.0, 0.0, labels_arr.tolist(), [], best_thresh

    thresh_candidates = np.linspace(0.1, 0.9, 33)
    for t in thresh_candidates:
        preds_t = (probs_arr >= t).astype(int)
        bal = balanced_accuracy_score(labels_arr, preds_t)
        if bal > best_bal:
            best_bal = bal
            best_thresh = float(t)
            best_metrics = {
                "f1": float(f1_score(labels_arr, preds_t, average="binary", zero_division=0)),
                "precision": float(precision_score(labels_arr, preds_t, average="binary", zero_division=0)),
                "recall": float(recall_score(labels_arr, preds_t, average="binary", zero_division=0)),
            }

    best_preds = (probs_arr >= best_thresh).astype(int).tolist()

    return epoch_loss, float(best_bal), best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], labels_arr.tolist(), best_preds, best_thresh
    
NUM_EPOCHS = 120
best_val_balanced = -1
best_val_f1 = -1
best_model_path = "best_simple_cnn.pth"
best_model_weights = None
# Longer training window so the wider custom CNN can keep improving
early_stop_patience = 30
epochs_without_improvement = 0

history = []

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_balanced, val_f1, val_precision, val_recall, val_true_all, val_pred_all, val_chosen_threshold = evaluate(
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
        "val_balanced": val_balanced,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_chosen_threshold": val_chosen_threshold
    })

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(
        f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
        f"F1: {train_f1:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}"
    )
    print(
        f"Val   | Loss: {val_loss:.4f} | Balanced: {val_balanced:.4f} | "
        f"F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}"
    )
    print(f"Val chosen threshold: {val_chosen_threshold:.3f}")
    print(f"LR    | {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 80)

    if val_balanced > best_val_balanced:
        best_val_balanced = val_balanced
        best_val_f1 = val_f1
        best_model_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(best_model_weights, best_model_path)
        print(f"Saved best model with validation balanced accuracy: {val_balanced:.4f} (thr={val_chosen_threshold:.3f})")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stop_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs without improvement in validation balanced accuracy.")
        break

print(f"Best model saved to: {best_model_path}")
print(f"Best validation balanced accuracy: {best_val_balanced:.4f}")
print(f"Best validation F1: {best_val_f1:.4f}")

history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False, encoding="utf-8-sig")
history_df.head()

from sklearn.metrics import confusion_matrix, classification_report

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
else:
    model.load_state_dict(torch.load(best_model_path, map_location=device))

val_loss, val_balanced, val_f1, val_precision, val_recall, val_true, val_pred, val_thresh = evaluate(
    model, val_loader, criterion, device
)

print("Validation Confusion Matrix:")
print(confusion_matrix(val_true, val_pred))
print()

print("Validation Classification Report:")
print(classification_report(val_true, val_pred, digits=4))

test_loss, test_balanced, test_f1, test_precision, test_recall, test_true, test_pred, test_thresh = evaluate(
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
 
