import os
import json
import copy
import random
from pathlib import Path
from collections import Counter
from PIL import Image
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix
)

# =========================================================
# Reproducibility and deterministic execution
# =========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# File paths and global configuration
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"

TRAIN_CSV = DATA_DIR / "train_split.csv"
VAL_CSV = DATA_DIR / "val_split.csv"

OUTPUT_DIR = PROJECT_ROOT / "hyperparameter_search_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = OUTPUT_DIR / "hyperparameter_search_results.csv"
EPOCH_HISTORY_CSV = OUTPUT_DIR / "hyperparameter_epoch_history.csv"
BEST_MODEL_PATH = OUTPUT_DIR / "best_hyperparameter_model.pt"
BEST_CONFIG_PATH = OUTPUT_DIR / "best_hyperparameter_config.json"

IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

EPOCHS = 75
PATIENCE = 10
MIN_DELTA = 1e-4

# =========================================================
# Search strategy
# =========================================================
# Supported options:
# "random"  - Recommended default for practical runs
# "grid"    - Evaluates every combination in the search space
# "optuna"  - Requires installation: pip install optuna
SEARCH_METHOD = "optuna"

# Number of random combinations to sample from the full grid.
# 50-60 usually gives a good balance between runtime and search quality.
N_RANDOM_RUNS = 50

# Number of Optuna trials when SEARCH_METHOD = "optuna"
N_OPTUNA_TRIALS = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Hyperparameter grid
# =========================================================
HYPERPARAMETER_GRID = {
    "dropout_spatial": [0.2, 0.3, 0.4],
    "dropout_classifier": [0.25, 0.35, 0.5],
    "label_smoothing": [0.0, 0.05, 0.1],
    "activation": ["ReLU", "GELU"],
    "weight_decay": [1e-4, 5e-4, 1e-3],
    "lr": [1e-4, 5e-4, 1e-3],
    "threshold": [0.5, 0.6, 0.7],
}

# =========================================================
# Dataset loading and indexing logic
# =========================================================
class ImageCSVDataset(Dataset):
    def __init__(
        self,
        csv_path,
        transform=None,
        clean_conflicts=False,
        label_to_index=None
    ):
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
                raise KeyError(f"{csv_file} must contain one of {image_column_candidates}")

            data = data[data["pain_label"].notna()].copy()

            for column in image_columns:
                data[column] = data[column].fillna("").astype(str)

            frames.append(data)

        self.data = pd.concat(frames, ignore_index=True)

        self.image_columns = [
            col for col in image_column_candidates
            if col in self.data.columns
        ]

        self.data["_resolved_image_path"] = self.data.apply(
            self._resolve_row_image_path,
            axis=1
        )

        self.data = self.data[
            self.data["_resolved_image_path"].apply(lambda p: Path(p).exists())
        ].copy()

        self.data["pain_label"] = pd.to_numeric(
            self.data["pain_label"],
            errors="coerce"
        )

        self.data = self.data[self.data["pain_label"].notna()].copy()

        raw_labels = sorted(self.data["pain_label"].unique().tolist())

        if not raw_labels:
            raise ValueError("No valid pain_label values found.")

        if label_to_index is None:
            self.label_to_index = {
                label: index
                for index, label in enumerate(raw_labels)
            }
        else:
            self.label_to_index = label_to_index

            unknown_labels = set(raw_labels) - set(self.label_to_index.keys())

            if unknown_labels:
                raise ValueError(
                    f"Validation/Test contains labels not found in train mapping: {unknown_labels}"
                )

        self.index_to_label = {
            index: label
            for label, index in self.label_to_index.items()
        }

        self.data["pain_label"] = self.data["pain_label"].map(self.label_to_index)
        self.data = self.data[self.data["pain_label"].notna()].copy()
        self.data["pain_label"] = self.data["pain_label"].astype(int)

        if self.clean_conflicts:
            path_mode_label = self.data.groupby("_resolved_image_path")["pain_label"].agg(
                lambda values: values.mode().iloc[0]
            )

            self.data["pain_label"] = self.data["_resolved_image_path"].map(
                path_mode_label
            ).astype(int)

            self.data = self.data.drop_duplicates(
                subset=["_resolved_image_path"],
                keep="first"
            ).copy()

        self.samples = [
            (row["_resolved_image_path"], int(row["pain_label"]))
            for _, row in self.data.iterrows()
        ]

        self.classes = [
            str(self.index_to_label[index])
            for index in sorted(self.index_to_label.keys())
        ]

    def _resolve_image_path(self, image_path):
        image_path_str = str(image_path).strip()

        if not image_path_str:
            raise FileNotFoundError("Empty image path in CSV row")

        raw_path = Path(image_path_str)
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([
                PROJECT_ROOT / raw_path,
                DATA_DIR / raw_path,
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Could not resolve image path '{image_path_str}' as absolute, relative to PROJECT_ROOT, or relative to DATA_DIR"
        )

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
# Image preprocessing transforms
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =========================================================
# CNN model definition
# =========================================================
class ConfigurableCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout_spatial=0.3,
        dropout_classifier=0.25,
        activation="ReLU"
    ):
        super().__init__()

        if activation == "ReLU":
            act_fn = nn.ReLU
        elif activation == "GELU":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=9, padding=4),
            nn.BatchNorm2d(24),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(24, 48, kernel_size=7, padding=3),
            nn.BatchNorm2d(48),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(48, 96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96),
            act_fn(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_spatial),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act_fn(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(128, 64),
            act_fn(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# Label smoothing loss
# =========================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, weight=None):
        super().__init__()

        if smoothing < 0 or smoothing >= 1:
            raise ValueError("smoothing must be in the range [0, 1).")

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            reduction="none"
        )

    def forward(self, logits, targets):
        if self.smoothing == 0.0:
            return self.ce_loss(logits, targets).mean()

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.smoothing
            )

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = torch.sum(-true_dist * log_probs, dim=1)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


# =========================================================
# Evaluation metrics
# =========================================================
def calculate_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist()
    }


# =========================================================
# Training and evaluation loops
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device, threshold):
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
        preds = (probs[:, 1] >= threshold).long()

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics


def evaluate(model, loader, criterion, device, threshold):
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
            preds = (probs[:, 1] >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)

    return epoch_loss, metrics


# =========================================================
# Helper utilities
# =========================================================
def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    num_samples = len(labels)

    weights = []

    for class_index in range(num_classes):
        if label_counts[class_index] == 0:
            raise ValueError(f"Class {class_index} has zero samples in training set.")

        weight = num_samples / (num_classes * label_counts[class_index])
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def get_all_grid_combinations(grid):
    from itertools import product

    param_names = list(grid.keys())
    param_values = [grid[name] for name in param_names]

    combinations = []

    for combo in product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def get_random_combinations(grid, n_runs, seed=42):
    all_combinations = get_all_grid_combinations(grid)

    rng = random.Random(seed)
    rng.shuffle(all_combinations)

    n_selected = min(n_runs, len(all_combinations))

    return all_combinations[:n_selected]


def get_search_combinations():
    all_combinations = get_all_grid_combinations(HYPERPARAMETER_GRID)

    if SEARCH_METHOD == "grid":
        print("Search method: Full Grid Search")
        return all_combinations

    if SEARCH_METHOD == "random":
        print("Search method: Random Search")
        print(f"Random runs requested: {N_RANDOM_RUNS}")
        print(f"Full grid size: {len(all_combinations)}")

        return get_random_combinations(
            HYPERPARAMETER_GRID,
            n_runs=N_RANDOM_RUNS,
            seed=SEED
        )

    if SEARCH_METHOD == "optuna":
        print("Search method: Optuna")
        return None

    raise ValueError(f"Unsupported SEARCH_METHOD: {SEARCH_METHOD}")


def sample_params_with_optuna(trial):
    params = {
        "dropout_spatial": trial.suggest_float("dropout_spatial", 0.15, 0.45),
        "dropout_classifier": trial.suggest_float("dropout_classifier", 0.20, 0.55),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.10),
        "activation": trial.suggest_categorical("activation", ["ReLU", "GELU"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "threshold": trial.suggest_float("threshold", 0.4, 0.75),
    }

    return params


# =========================================================
# Single training run for one hyperparameter combination
# =========================================================
def run_single_hyperparameter_trial(
    run_idx,
    params,
    train_dataset,
    val_dataset,
    class_weights,
    num_classes,
    results,
    epoch_history,
    global_best_tracker
):
    print(
        f"[{run_idx:03d}] "
        f"D_sp={params['dropout_spatial']} | "
        f"D_cl={params['dropout_classifier']} | "
        f"LS={params['label_smoothing']} | "
        f"Act={params['activation']} | "
        f"WD={params['weight_decay']:.2e} | "
        f"LR={params['lr']:.2e} | "
        f"TH={params['threshold']:.3f}"
    )

    torch.manual_seed(SEED + run_idx)
    torch.cuda.manual_seed_all(SEED + run_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = ConfigurableCNN(
        num_classes=num_classes,
        dropout_spatial=params["dropout_spatial"],
        dropout_classifier=params["dropout_classifier"],
        activation=params["activation"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    criterion = LabelSmoothingLoss(
        num_classes=num_classes,
        smoothing=params["label_smoothing"],
        weight=class_weights
    )

    best_run_val_balanced_acc = -1.0
    best_run_state = None
    best_run_epoch = 0
    patience_counter = 0

    best_run_metrics = None
    best_run_train_metrics = None
    best_run_train_loss = None
    best_run_val_loss = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=params["threshold"]
        )

        val_loss, val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=params["threshold"]
        )

        epoch_row = {
            "run": run_idx,
            "epoch": epoch,
            **params,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_balanced_acc": train_metrics["balanced_accuracy"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_confusion_matrix": json.dumps(val_metrics["confusion_matrix"])
        }

        epoch_history.append(epoch_row)

        current_val_balanced_acc = val_metrics["balanced_accuracy"]

        if current_val_balanced_acc > best_run_val_balanced_acc + MIN_DELTA:
            best_run_val_balanced_acc = current_val_balanced_acc
            best_run_state = copy.deepcopy(model.state_dict())
            best_run_epoch = epoch
            patience_counter = 0

            best_run_metrics = val_metrics
            best_run_train_metrics = train_metrics
            best_run_train_loss = train_loss
            best_run_val_loss = val_loss
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:03d} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Patience: {patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}. Best epoch: {best_run_epoch}")
            break

    if best_run_state is not None:
        model.load_state_dict(best_run_state)

    result = {
        "run": run_idx,
        **params,
        "best_epoch": best_run_epoch,
        "train_loss": best_run_train_loss,
        "train_acc": best_run_train_metrics["accuracy"],
        "train_balanced_acc": best_run_train_metrics["balanced_accuracy"],
        "train_f1": best_run_train_metrics["f1"],
        "train_precision": best_run_train_metrics["precision"],
        "train_recall": best_run_train_metrics["recall"],
        "val_loss": best_run_val_loss,
        "val_acc": best_run_metrics["accuracy"],
        "val_balanced_acc": best_run_metrics["balanced_accuracy"],
        "val_f1": best_run_metrics["f1"],
        "val_precision": best_run_metrics["precision"],
        "val_recall": best_run_metrics["recall"],
        "val_confusion_matrix": json.dumps(best_run_metrics["confusion_matrix"])
    }

    results.append(result)

    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(epoch_history).to_csv(EPOCH_HISTORY_CSV, index=False)

    print(
        f"Run {run_idx} best | "
        f"Epoch: {best_run_epoch} | "
        f"Val F1: {best_run_metrics['f1']:.4f} | "
        f"Val Acc: {best_run_metrics['accuracy']:.4f} | "
        f"Val Balanced Acc: {best_run_metrics['balanced_accuracy']:.4f}"
    )

    print("-" * 120)

    if best_run_val_balanced_acc > global_best_tracker["best_val_balanced_acc"]:
        global_best_tracker["best_val_balanced_acc"] = best_run_val_balanced_acc
        global_best_tracker["best_result"] = result

        torch.save(
            {
                "model_state_dict": best_run_state,
                "params": params,
                "best_epoch": best_run_epoch,
                "val_f1": best_run_metrics["f1"],
                "val_acc": best_run_metrics["accuracy"],
                "val_balanced_acc": best_run_metrics["balanced_accuracy"],
                "val_precision": best_run_metrics["precision"],
                "val_recall": best_run_metrics["recall"],
                "val_confusion_matrix": best_run_metrics["confusion_matrix"],
                "seed": SEED
            },
            BEST_MODEL_PATH
        )

        with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"New global best model saved to: {BEST_MODEL_PATH}")
        print(f"New global best config saved to: {BEST_CONFIG_PATH}")
        print("=" * 120)

    return result


# =========================================================
# Data preparation
# =========================================================
def prepare_data():
    print(f"Using device: {device}")
    print("Loading datasets...")

    train_dataset = ImageCSVDataset(
        TRAIN_CSV,
        transform=train_transform,
        clean_conflicts=False
    )

    val_dataset = ImageCSVDataset(
        VAL_CSV,
        transform=eval_transform,
        clean_conflicts=False,
        label_to_index=train_dataset.label_to_index
    )

    train_labels = [label for _, label in train_dataset.samples]
    train_label_counts = Counter(train_labels)

    num_classes = len(train_dataset.label_to_index)

    if num_classes != 2:
        raise ValueError(
            f"This script expects binary classification, but found {num_classes} classes."
        )

    class_weights = compute_class_weights(train_labels, num_classes).to(device)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Train class distribution: {train_label_counts}")
    print(f"Class weights: {class_weights.detach().cpu().numpy().tolist()}")
    print(f"Label mapping: {train_dataset.label_to_index}")
    print()

    return train_dataset, val_dataset, class_weights, num_classes


# =========================================================
# Random search / grid search execution
# =========================================================
def run_random_or_grid_search():
    train_dataset, val_dataset, class_weights, num_classes = prepare_data()

    combinations = get_search_combinations()
    total_runs = len(combinations)

    print(f"Total runs to test: {total_runs}")
    print(f"Max epochs per run: {EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print("=" * 120)

    results = []
    epoch_history = []

    global_best_tracker = {
        "best_val_balanced_acc": -1.0,
        "best_result": None
    }

    for run_idx, params in enumerate(combinations, start=1):
        run_single_hyperparameter_trial(
            run_idx=run_idx,
            params=params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_weights=class_weights,
            num_classes=num_classes,
            results=results,
            epoch_history=epoch_history,
            global_best_tracker=global_best_tracker
        )

    results_df = pd.DataFrame(results)
    epoch_history_df = pd.DataFrame(epoch_history)

    return results_df, epoch_history_df


# =========================================================
# Optuna search execution
# =========================================================
def run_optuna_search():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is not installed. Install it using: pip install optuna"
        ) from exc

    train_dataset, val_dataset, class_weights, num_classes = prepare_data()

    results = []
    epoch_history = []

    global_best_tracker = {
        "best_val_balanced_acc": -1.0,
        "best_result": None
    }

    def objective(trial):
        run_idx = trial.number + 1
        params = sample_params_with_optuna(trial)

        result = run_single_hyperparameter_trial(
            run_idx=run_idx,
            params=params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            class_weights=class_weights,
            num_classes=num_classes,
            results=results,
            epoch_history=epoch_history,
            global_best_tracker=global_best_tracker
        )

        return result["val_balanced_acc"]

    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS
    )

    print()
    print("=" * 120)
    print("OPTUNA SEARCH COMPLETE")
    print("=" * 120)
    print(f"Best trial: {study.best_trial.number + 1}")
    print(f"Best value - Val Balanced Acc: {study.best_value}")
    print(f"Best params: {study.best_params}")

    results_df = pd.DataFrame(results)
    epoch_history_df = pd.DataFrame(epoch_history)

    return results_df, epoch_history_df


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    print(f"Starting Hyperparameter Search at {datetime.now()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Search method: {SEARCH_METHOD}")
    print()

    if "--dry-run" in os.sys.argv:
        prepare_data()
        raise SystemExit(0)

    if SEARCH_METHOD in ["random", "grid"]:
        results_df, epoch_history_df = run_random_or_grid_search()

    elif SEARCH_METHOD == "optuna":
        results_df, epoch_history_df = run_optuna_search()

    else:
        raise ValueError(f"Unsupported SEARCH_METHOD: {SEARCH_METHOD}")

    print()
    print("=" * 120)
    print("SEARCH COMPLETE - TOP 10 CONFIGURATIONS BY VALIDATION BALANCED ACCURACY")
    print("=" * 120)

    results_df["val_balanced_acc_numeric"] = results_df["val_balanced_acc"].astype(float)

    top_10 = results_df.nlargest(10, "val_balanced_acc_numeric")

    columns_to_show = [
        "run",
        "best_epoch",
        "dropout_spatial",
        "dropout_classifier",
        "label_smoothing",
        "activation",
        "weight_decay",
        "lr",
        "threshold",
        "val_f1",
        "val_acc",
        "val_balanced_acc",
        "val_precision",
        "val_recall"
    ]

    print(top_10[columns_to_show].to_string(index=False))

    best = top_10.iloc[0]

    print()
    print("=" * 120)
    print(f"Best configuration: Run {int(best['run'])}")
    print("=" * 120)

    for col in columns_to_show:
        print(f"{col}: {best[col]}")

    print()
    print(f"Full run results saved to: {RESULTS_CSV}")
    print(f"Epoch history saved to: {EPOCH_HISTORY_CSV}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Best config saved to: {BEST_CONFIG_PATH}")