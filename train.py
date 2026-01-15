"""
Training script for image classification
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import random
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from dataset import (
    ImageClassificationDataset, get_transforms, 
    create_data_loaders, load_data_from_folders
)
from model import get_model
from utils import (
    calculate_accuracy, evaluate_model, get_detailed_metrics,
    plot_confusion_matrix, plot_training_history,
    AverageMeter, EarlyStopping
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        
        # Update meters
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % LOG_INTERVAL == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/BatchAccuracy', acc, global_step)
    
    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate the model"""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            acc = calculate_accuracy(outputs, labels)
            
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
    
    if writer:
        writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Val/Accuracy', acc_meter.avg, epoch)
    
    return loss_meter.avg, acc_meter.avg


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
          device, num_epochs, save_dir, writer=None):
    """Main training loop"""
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='max')
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved best model with Val Acc: {val_acc:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, os.path.join(CHECKPOINTS_DIR, f'checkpoint_epoch_{epoch}.pth'))
        
        # Early stopping
        if early_stopping(val_acc):
            print(f'\nEarly stopping triggered after epoch {epoch}')
            break
    
    return history, best_val_acc


def main(args):
    """Main function"""
    # Set seed
    set_seed(RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Load data
    print('\nLoading data...')
    if args.data_dir and os.path.exists(args.data_dir):
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, class_names = \
            load_data_from_folders(args.data_dir, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED)
        
        print(f'Number of classes: {len(class_names)}')
        print(f'Classes: {class_names}')
        print(f'Train samples: {len(train_paths)}')
        print(f'Val samples: {len(val_paths)}')
        print(f'Test samples: {len(test_paths)}')
        
        # Update NUM_CLASSES
        num_classes = len(class_names)
    else:
        print('Warning: No data directory provided or directory does not exist.')
        print('Using dummy data for demonstration purposes.')
        print('Please organize your data in: data_dir/class_name/image.jpg structure')
        return
    
    # Create datasets
    train_transform = get_transforms(IMG_HEIGHT, IMG_WIDTH, augment=USE_AUGMENTATION)
    val_transform = get_transforms(IMG_HEIGHT, IMG_WIDTH, augment=False)
    
    train_dataset = ImageClassificationDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_labels, val_transform)
    test_dataset = ImageClassificationDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE, NUM_WORKERS
    )
    
    # Create model
    print(f'\nCreating model: {MODEL_NAME}')
    model = get_model(MODEL_NAME, num_classes, PRETRAINED, FREEZE_BACKBONE)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=REDUCE_LR_PATIENCE, verbose=True
    )
    
    # Tensorboard writer
    writer = None
    if TENSORBOARD:
        writer = SummaryWriter(log_dir=LOGS_DIR)
    
    # Train model
    print('\nStarting training...')
    history, best_val_acc = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, NUM_EPOCHS, SAVED_MODELS_DIR, writer
    )
    
    # Plot training history
    plot_training_history(history, os.path.join(SAVED_MODELS_DIR, 'training_history.png'))
    
    # Load best model for testing
    print('\nLoading best model for testing...')
    checkpoint = torch.load(os.path.join(SAVED_MODELS_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test model
    print('\nEvaluating on test set...')
    metrics = get_detailed_metrics(model, test_loader, device, class_names)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names, 
        os.path.join(SAVED_MODELS_DIR, 'confusion_matrix.png')
    )
    
    if writer:
        writer.close()
    
    print('\nTraining completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model')
    parser.add_argument('--data_dir', type=str, default=RAW_DATA_DIR,
                       help='Path to data directory')
    
    args = parser.parse_args()
    main(args)
