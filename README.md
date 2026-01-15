# Image Classification Project

A comprehensive deep learning project for image classification using PyTorch, featuring multiple model architectures, data augmentation, and comprehensive evaluation tools.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Prediction](#prediction)
- [Results](#results)

## ✨ Features

- **Multiple Model Architectures**: ResNet50, EfficientNet-B0, VGG16, and Custom CNN
- **Transfer Learning**: Support for pretrained models on ImageNet
- **Data Augmentation**: Comprehensive augmentation using Albumentations
- **Training Features**:
  - Early stopping
  - Learning rate scheduling
  - Checkpoint saving
  - TensorBoard logging
  - Progress tracking with tqdm
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Inference**: Single image and batch prediction support
- **Jupyter Notebook**: Interactive exploration and visualization

## 📁 Project Structure

```
Final-Project/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and hyperparameters
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # Model architectures
│   └── utils.py           # Utility functions and metrics
├── data/
│   ├── raw/               # Raw training data (organized by class)
│   └── processed/         # Processed data
├── models/
│   ├── saved_models/      # Trained model checkpoints
│   └── checkpoints/       # Training checkpoints
├── notebooks/
│   └── exploration.ipynb  # Jupyter notebook for exploration
├── logs/                  # TensorBoard logs
├── train.py              # Training script
├── predict.py            # Inference script
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nirlapi/Final-Project.git
cd Final-Project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Prepare Your Data

Organize your images in the following structure:
```
data/raw/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── classN/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 2. Train a Model

```bash
python train.py --data_dir data/raw
```

### 3. Make Predictions

For a single image:
```bash
python predict.py \
  --model_path models/saved_models/best_model.pth \
  --image_path path/to/your/image.jpg \
  --num_classes 10
```

For a directory of images:
```bash
python predict.py \
  --model_path models/saved_models/best_model.pth \
  --image_dir path/to/images/ \
  --num_classes 10 \
  --output results.json
```

## 📖 Usage

### Configuration

Edit `src/config.py` to customize:
- Model architecture
- Training hyperparameters
- Image dimensions
- Data augmentation settings
- Paths

Key parameters:
```python
MODEL_NAME = 'resnet50'  # 'resnet50', 'efficientnet', 'vgg16', 'custom_cnn'
NUM_CLASSES = 10         # Number of classes in your dataset
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMG_HEIGHT = 224
IMG_WIDTH = 224
```

### Training Options

```bash
python train.py --help
```

Options:
- `--data_dir`: Path to data directory (default: data/raw)

### Prediction Options

```bash
python predict.py --help
```

Options:
- `--model_path`: Path to trained model checkpoint (required)
- `--model_name`: Model architecture name (default: from config)
- `--image_path`: Single image for prediction
- `--image_dir`: Directory of images for batch prediction
- `--class_names`: JSON file with class names
- `--num_classes`: Number of classes
- `--top_k`: Number of top predictions to show (default: 3)
- `--output`: Path to save results as JSON

## 🏗️ Model Architectures

### Available Models

1. **ResNet50** (Default)
   - Deep residual network
   - 50 layers
   - Pretrained on ImageNet

2. **EfficientNet-B0**
   - Efficient architecture
   - Compound scaling
   - Pretrained on ImageNet

3. **VGG16**
   - Classic architecture
   - 16 layers
   - Pretrained on ImageNet

4. **Custom CNN**
   - Lightweight custom architecture
   - 3 convolutional blocks
   - Batch normalization and dropout

### Switching Models

Change `MODEL_NAME` in `src/config.py`:
```python
MODEL_NAME = 'resnet50'  # or 'efficientnet', 'vgg16', 'custom_cnn'
```

## 📊 Dataset Preparation

### Image Requirements

- **Format**: JPG, PNG, JPEG, BMP, GIF
- **Size**: Any size (will be resized to IMG_HEIGHT x IMG_WIDTH)
- **Channels**: RGB (3 channels)

### Data Organization

```
data/raw/
├── class_name_1/
├── class_name_2/
└── class_name_N/
```

The dataset will be automatically split into:
- Training: 70%
- Validation: 15%
- Testing: 15%

Adjust splits in `src/config.py`:
```python
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

## 🎓 Training

### Basic Training

```bash
python train.py --data_dir data/raw
```

### Training Features

- **Automatic checkpointing**: Saves best model and epoch checkpoints
- **Early stopping**: Stops training if validation accuracy doesn't improve
- **Learning rate scheduling**: Reduces learning rate on plateau
- **TensorBoard logging**: Visualize training progress
- **Progress bars**: Real-time training progress with tqdm

### Monitor Training

View TensorBoard logs:
```bash
tensorboard --logdir logs
```

### Training Output

- `models/saved_models/best_model.pth`: Best model checkpoint
- `models/checkpoints/`: Epoch checkpoints
- `models/saved_models/training_history.png`: Training curves
- `models/saved_models/confusion_matrix.png`: Test set confusion matrix

## 🔮 Prediction

### Single Image Prediction

```bash
python predict.py \
  --model_path models/saved_models/best_model.pth \
  --image_path test_image.jpg \
  --num_classes 10 \
  --top_k 5
```

Output:
```
Top 5 Predictions:
1. class_name_1: 95.32%
2. class_name_2: 3.45%
3. class_name_3: 0.89%
4. class_name_4: 0.21%
5. class_name_5: 0.13%
```

### Batch Prediction

```bash
python predict.py \
  --model_path models/saved_models/best_model.pth \
  --image_dir test_images/ \
  --num_classes 10 \
  --output predictions.json
```

### Using Class Names

Create a JSON file with class names:
```json
["cat", "dog", "bird", "fish", "horse"]
```

Use it in prediction:
```bash
python predict.py \
  --model_path models/saved_models/best_model.pth \
  --image_path test.jpg \
  --class_names class_names.json
```

## 📈 Results

After training, you'll get:

1. **Training History Plot**: Loss and accuracy curves
2. **Confusion Matrix**: Visualization of predictions
3. **Classification Report**: Detailed metrics per class
4. **Model Checkpoints**: Best model and epoch checkpoints

### Example Metrics

```
Test Results:
Accuracy: 0.9523
Precision: 0.9534
Recall: 0.9523
F1-Score: 0.9527

Classification Report:
              precision    recall  f1-score   support
     class_0       0.96      0.94      0.95       100
     class_1       0.95      0.97      0.96       100
     ...
```

## 🔧 Advanced Usage

### Custom Data Augmentation

Modify augmentation in `src/dataset.py`:
```python
transform = A.Compose([
    A.Resize(img_height, img_width),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    # Add your custom augmentations here
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

### Fine-tuning Pretrained Models

To freeze backbone layers initially:
```python
# In src/config.py
FREEZE_BACKBONE = True
```

### Jupyter Notebook Exploration

```bash
jupyter notebook notebooks/exploration.ipynb
```

The notebook includes:
- Data exploration
- Visualization of samples
- Augmentation testing
- Model architecture inspection
- Prediction visualization

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Albumentations for data augmentation library
- torchvision for pretrained models