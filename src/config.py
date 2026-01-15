"""
Configuration file for image classification project
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Model parameters
MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'efficientnet', 'vgg16', 'custom_cnn'
NUM_CLASSES = 10  # Update based on your dataset
PRETRAINED = True
FREEZE_BACKBONE = False  # Set True to freeze pretrained weights initially

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
SAVE_BEST_ONLY = True

# Random seed for reproducibility
RANDOM_SEED = 42

# Device
DEVICE = 'cuda'  # 'cuda' or 'cpu'

# Augmentation parameters
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 15
COLOR_JITTER = True

# Logging
LOG_INTERVAL = 10  # Log every N batches
TENSORBOARD = True
WANDB = False  # Set True if using Weights & Biases

# Inference
CONFIDENCE_THRESHOLD = 0.5
