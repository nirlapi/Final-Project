import os
import random
import json
import copy
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
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# 1. Reproducibility & Setup
# =========================================================
# In deep learning research and academic projects, ensuring reproducibility is critical.
# By fixing the random seeds across all libraries (Python, NumPy, PyTorch), we guarantee 
# that weight initialization, dataset shuffling, and dropout layers behave identically 
# across different runs.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Forcing cuDNN to be deterministic ensures that convolutional operations do not use 
# non-deterministic algorithms to speed up computation, sacrificing a bit of speed for 
# strict mathematical reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Using pathlib ensures cross-platform compatibility (Windows/Linux/macOS) for file paths.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"

# Defining manifests for data splits. Manifest-based loading is a standard data science 
# practice to strictly separate data tracking from directory structures.
TRAIN_CSV = DATA_DIR / "train_split.csv"
VAL_CSV = DATA_DIR / "val_split.csv"
TEST_CSV = DATA_DIR / "test_split.csv"

# Path to the saved best model from hyperparameter_search.py
BEST_MODEL_CHECKPOINT = PROJECT_ROOT / "hyperparameter_search_outputs" / "best_hyperparameter_model.pt"

# Hyperparameters definition
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 50  # Additional epochs for fine-tuning
# Dynamic device allocation (uses GPU if available for faster tensor operations, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 2. Dataset Logic
# =========================================================
# Custom Dataset class inherits from torch.utils.data.Dataset. This is required 
# to bridge our tabular metadata (CSVs) with raw image tensors for batch processing.
class ImageCSVDataset(Dataset):
    def __init__(self, csv_path, transform=None, clean_conflicts=False, label_to_index=None):
        # Support passing a single path or a list of paths for dataset concatenation
        csv_paths = [csv_path] if isinstance(csv_path, (str, Path)) else list(csv_path)
        self.csv_paths = [Path(path) for path in csv_paths]
        self.transform = transform
        self.clean_conflicts = clean_conflicts

        image_column_candidates = ["resized_file_path", "file_path"]
        frames = []

        for csv_file in self.csv_paths:
            data = pd.read_csv(csv_file)
            # Data validation: Ensure target variable exists
            if "pain_label" not in data.columns:
                raise KeyError(f"{csv_file} must contain a 'pain_label' column.")
            image_columns = [col for col in image_column_candidates if col in data.columns]
            if not image_columns:
                raise KeyError(f"{csv_file} must contain one of {image_column_candidates}")
            
            # Drop records with missing target labels to prevent NaN loss during backpropagation
            data = data[data["pain_label"].notna()].copy()
            
            # Fill NaNs in path columns with empty strings to prevent TypeError during parsing
            for column in image_columns:
                data[column] = data[column].fillna("").astype(str)
            frames.append(data)

        # Consolidate all loaded manifests into a single DataFrame
        self.data = pd.concat(frames, ignore_index=True)
        self.image_columns = [col for col in image_column_candidates if col in self.data.columns]
        
        # Resolve dynamic paths ensuring the data pipeline works regardless of the execution directory
        self.data["_resolved_image_path"] = self.data.apply(self._resolve_row_image_path, axis=1)
        
        # Strict filtering: Discard any rows where the underlying image file doesn't actually exist
        self.data = self.data[self.data["_resolved_image_path"].apply(lambda p: Path(p).exists())].copy()
        
        # Dynamically determine classes and map string/non-sequential labels to 0-indexed integers 
        # (Required by PyTorch's CrossEntropyLoss)
        raw_labels = sorted(pd.to_numeric(self.data["pain_label"], errors="coerce").dropna().unique().tolist())
        self.label_to_index = label_to_index if label_to_index is not None else {lbl: i for i, lbl in enumerate(raw_labels)}
        
        self.data["pain_label"] = pd.to_numeric(self.data["pain_label"], errors="coerce").map(self.label_to_index)
        
        # Apply integer casting only to the label column to prevent errors with string IDs
        self.data = self.data[self.data["pain_label"].notna()].copy()
        self.data["pain_label"] = self.data["pain_label"].astype(int)

        # Pre-build a lightweight list of tuples containing (path, label) for fast O(1) retrieval during training
        self.samples = [(row["_resolved_image_path"], int(row["pain_label"])) for _, row in self.data.iterrows()]
        self.classes = [str(self.label_to_index[index]) for index in sorted(self.label_to_index.values())]

    def _resolve_image_path(self, image_path):
        # Helper function to check multiple directory roots to locate the image
        image_path_str = str(image_path).strip()
        raw_path = Path(image_path_str)
        candidates = [raw_path, PROJECT_ROOT / raw_path, DATA_DIR / raw_path]
        for candidate in candidates:
            if candidate.exists(): return candidate.resolve()
        raise FileNotFoundError(f"Could not resolve: {image_path_str}")

    def _resolve_row_image_path(self, row):
        # Iterates through potential image column headers to find a valid path
        for col in self.image_columns:
            p = str(row.get(col, "")).strip()
            if p:
                try: return self._resolve_image_path(p)
                except: continue
        return None

    def __len__(self): 
        # Returns the total number of samples; used by DataLoader to determine epoch size
        return len(self.samples)
        
    def __getitem__(self, index):
        # Memory efficient loading: Images are loaded from disk only when requested by the DataLoader batch
        path, label = self.samples[index]
        with Image.open(path) as image:
            # Enforce 3-channel RGB to prevent dimension mismatch errors with grayscale (1-channel) images
            image = image.convert("RGB")
            if self.transform: image = self.transform(image)
        return image, label

# =========================================================
# 3. Model Architecture
# =========================================================
# A lightweight, custom Convolutional Neural Network (CNN) architecture.
# Designed to be configurable dynamically based on hyperparameters discovered during the search phase.
class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes, dropout_spatial=0.3, dropout_classifier=0.25, activation="ReLU"):
        super().__init__()
        act_fn = nn.ReLU if activation == "ReLU" else nn.GELU

        # Feature Extractor: Hierarchical spatial pattern recognition
        # Uses increasing channel dimensions (24->48->96->128) to learn complex features.
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=9, padding=4),
            nn.BatchNorm2d(24), act_fn(), nn.MaxPool2d(2), nn.Dropout2d(p=dropout_spatial),
            
            nn.Conv2d(24, 48, kernel_size=7, padding=3),
            nn.BatchNorm2d(48), act_fn(), nn.MaxPool2d(2), nn.Dropout2d(p=dropout_spatial),
            
            nn.Conv2d(48, 96, kernel_size=7, padding=3),
            nn.BatchNorm2d(96), act_fn(), nn.MaxPool2d(2), nn.Dropout2d(p=dropout_spatial),
            
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            # Batch Normalization reduces internal covariate shift, allowing for higher learning rates
            nn.BatchNorm2d(128), act_fn(), nn.MaxPool2d(2),
            
            # AdaptiveAvgPool2d crushes the spatial dimensions to 1x1, making the network 
            # agnostic to the exact input image resolution before passing to the linear layers.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier: Maps extracted feature vectors to class predictions
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout acts as strong regularization, zeroing out random neurons to prevent overfitting
            nn.Dropout(p=dropout_classifier),
            nn.Linear(128, 64),
            act_fn(),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Custom Loss Function integrating Label Smoothing
# Label Smoothing converts hard targets (e.g., [1, 0]) into soft targets (e.g., [0.9, 0.1]).
# This prevents the network from becoming overly confident, reducing overfitting and improving calibration.
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0, weight=None):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.weight = weight
        # Initialize standard CrossEntropy without reduction so we can modify the distributions
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        if self.smoothing == 0.0:
            return self.ce_loss(logits, targets).mean()

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Apply the smoothed target distribution to the log-probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = torch.sum(-true_dist * log_probs, dim=1)
        
        # Apply class balancing weights if provided
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()

# =========================================================
# 4. Initialization & Weight Loading
# =========================================================
# Computes inverse-frequency class weights to combat dataset imbalance.
# Underrepresented classes receive higher penalty weights during loss calculation.
def compute_class_weights(labels, num_classes):
    label_counts = Counter(labels)
    num_samples = len(labels)
    weights = [num_samples / (num_classes * label_counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

print(f"Loading best checkpoint from: {BEST_MODEL_CHECKPOINT}")
if not BEST_MODEL_CHECKPOINT.exists():
    raise FileNotFoundError(f"Cannot find {BEST_MODEL_CHECKPOINT}. Run hyperparameter_search.py first.")

# Load the dictionary containing optimal hyperparameters found during the grid/random search phase
checkpoint = torch.load(BEST_MODEL_CHECKPOINT, map_location=device)
best_params = checkpoint['params']

print("\n--- Loaded Hyperparameters ---")
for k, v in best_params.items():
    print(f"{k}: {v}")
print("------------------------------\n")

# Standard normalization (mean=0.5, std=0.5) scales pixel values to [-1, 1].
# This helps the optimizer converge faster by centering the input feature distribution.
train_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
eval_transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

# Initialize datasets and dataloaders
train_dataset = ImageCSVDataset(TRAIN_CSV, transform=train_transform)
# Pass train_dataset.label_to_index to validation/test sets to ensure consistent class mappings
val_dataset = ImageCSVDataset(VAL_CSV, transform=eval_transform, label_to_index=train_dataset.label_to_index)
test_dataset = ImageCSVDataset(TEST_CSV, transform=eval_transform, label_to_index=train_dataset.label_to_index)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

num_classes = len(train_dataset.classes)
class_weights = compute_class_weights([lbl for _, lbl in train_dataset.samples], num_classes).to(device)

# Instantiate the model utilizing the exact architectural parameters that won the search phase
model = ConfigurableCNN(
    num_classes=num_classes,
    dropout_spatial=best_params['dropout_spatial'],
    dropout_classifier=best_params['dropout_classifier'],
    activation=best_params['activation']
).to(device)

# Load the learned weights corresponding to those optimal hyperparameters
model.load_state_dict(checkpoint['model_state_dict'])
print("Successfully loaded model weights from checkpoint.")

# Initialize the Adam optimizer with specific learning rate and L2 regularization (weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=best_params['label_smoothing'], weight=class_weights)

# The optimal probability threshold for binary classification (may not be exactly 0.5 for imbalanced sets)
EVAL_THRESHOLD = best_params['threshold']

# =========================================================
# 5. Continued Training Loop (With Val Metrics & History)
# =========================================================
def train_one_epoch():
    # model.train() enables Dropout and Batch Normalization tracking
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero out gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass -> calculate loss -> backward pass (backpropagation) -> step (update weights)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Calculate training predictions using the optimized threshold
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs >= EVAL_THRESHOLD).long()
        
        # Detach prevents the tracking of gradients for metrics calculation, saving memory
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # F1 score is tracked as it is a more robust metric than accuracy for imbalanced datasets
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate_validation():
    # model.eval() disables Dropout and uses population statistics for BatchNorm
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # torch.no_grad() explicitly turns off the autograd engine, reducing memory footprint and speeding up inference
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate validation predictions
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= EVAL_THRESHOLD).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    
    return epoch_loss, epoch_acc, epoch_f1

best_combined_loss = float('inf')
best_model_weights = None
history = [] # Tracks epoch-by-epoch metrics for later visualization and analysis

print(f"\nResuming training for {EPOCHS} additional epochs on {device}...")
for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1 = train_one_epoch()
    val_loss, val_acc, val_f1 = evaluate_validation()
    
    # Heuristic constraint: Summing training and validation loss helps select a model state 
    # that generalizes well (low val loss) while adequately fitting the data (low train loss).
    combined_loss = train_loss + val_loss
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Combined: {combined_loss:.4f}")
    
    # Save the metrics to our history list
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "combined_loss": combined_loss
    })
    
    # Model Selection Strategy: Save the model only if the combined loss is strictly improving
    if combined_loss < best_combined_loss:
        best_combined_loss = combined_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  -> New best model found! (Combined Loss: {best_combined_loss:.4f})")

# Load the best weights before proceeding to final export phase
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# Save the history to CSV for the Jupyter Notebook (allows for generation of training curve plots)
history_df = pd.DataFrame(history)
history_csv_path = PROJECT_ROOT / "training_history.csv"
history_df.to_csv(history_csv_path, index=False)
print(f"\nTraining history saved to {history_csv_path}")

final_model_path = PROJECT_ROOT / "final_finetuned_model.pt"
torch.save(model.state_dict(), final_model_path)
print(f"Finished training. Best model saved to {final_model_path}")


# =========================================================
# 6. Export Test Predictions for IPYNB Analysis
# =========================================================
# Separating the evaluation and visualization out of the training script into a notebook 
# is best practice. This function persists the raw prediction data required to plot 
# ROC curves, confusion matrices, and conduct error analysis downstream.
def export_test_predictions(model, loader, dataset, threshold, output_csv):
    print(f"\nExporting test predictions to {output_csv}...")
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Map predictions back to exact file paths to enable manual inspection of False Positives/Negatives
    paths = [str(path) for path, _ in dataset.samples]
    
    df_results = pd.DataFrame({
        "file_path": paths,
        "true_label": all_labels,
        "predicted_label": all_preds,
        "probability_class_1": all_probs
    })
    
    df_results.to_csv(output_csv, index=False)
    print("Export complete. You can now load this CSV in your Jupyter Notebook for EDA.")

# Run the export function using the final (best) model state
export_csv_path = PROJECT_ROOT / "test_predictions.csv"
export_test_predictions(model, test_loader, test_dataset, EVAL_THRESHOLD, export_csv_path)