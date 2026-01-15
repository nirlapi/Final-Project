"""
Example script demonstrating how to use the image classification project
This script shows how to prepare data, train a model, and make predictions
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 70)
print("Image Classification Project - Example Usage")
print("=" * 70)

print("\n1. INSTALLATION")
print("-" * 70)
print("First, install the required dependencies:")
print("  $ pip install -r requirements.txt")

print("\n2. DATA PREPARATION")
print("-" * 70)
print("Organize your data in this structure:")
print("  data/raw/")
print("  ├── class1/")
print("  │   ├── image1.jpg")
print("  │   ├── image2.jpg")
print("  │   └── ...")
print("  ├── class2/")
print("  │   ├── image1.jpg")
print("  │   └── ...")
print("  └── classN/")
print("      └── ...")

print("\n3. CONFIGURATION")
print("-" * 70)
print("Edit src/config.py to customize:")
print("  - MODEL_NAME: Choose from 'resnet50', 'efficientnet', 'vgg16', 'custom_cnn'")
print("  - NUM_CLASSES: Set to match your dataset")
print("  - BATCH_SIZE: Adjust based on GPU memory")
print("  - NUM_EPOCHS: Number of training epochs")
print("  - LEARNING_RATE: Learning rate for optimizer")

print("\n4. TRAINING")
print("-" * 70)
print("Train the model:")
print("  $ python train.py --data_dir data/raw")
print("\nThe script will:")
print("  • Load and split your data (70% train, 15% val, 15% test)")
print("  • Apply data augmentation")
print("  • Train the model with early stopping")
print("  • Save the best model to models/saved_models/best_model.pth")
print("  • Generate training history plots")
print("  • Evaluate on test set and show metrics")

print("\n5. MONITORING TRAINING")
print("-" * 70)
print("View training progress with TensorBoard:")
print("  $ tensorboard --logdir logs")
print("Then open http://localhost:6006 in your browser")

print("\n6. INFERENCE - SINGLE IMAGE")
print("-" * 70)
print("Predict class for a single image:")
print("  $ python predict.py \\")
print("      --model_path models/saved_models/best_model.pth \\")
print("      --image_path path/to/image.jpg \\")
print("      --num_classes 10 \\")
print("      --top_k 5")

print("\n7. INFERENCE - BATCH")
print("-" * 70)
print("Predict classes for multiple images:")
print("  $ python predict.py \\")
print("      --model_path models/saved_models/best_model.pth \\")
print("      --image_dir path/to/images/ \\")
print("      --num_classes 10 \\")
print("      --output predictions.json")

print("\n8. USING JUPYTER NOTEBOOK")
print("-" * 70)
print("Explore the project interactively:")
print("  $ jupyter notebook notebooks/exploration.ipynb")
print("\nThe notebook includes:")
print("  • Data exploration and visualization")
print("  • Model architecture inspection")
print("  • Training monitoring")
print("  • Prediction visualization")

print("\n9. ADVANCED: CUSTOM CLASS NAMES")
print("-" * 70)
print("Create a JSON file with class names (e.g., class_names.json):")
print('  ["cat", "dog", "bird", "fish", "horse"]')
print("\nUse it in prediction:")
print("  $ python predict.py \\")
print("      --model_path models/saved_models/best_model.pth \\")
print("      --image_path test.jpg \\")
print("      --class_names class_names.json")

print("\n10. TIPS FOR BETTER RESULTS")
print("-" * 70)
print("• Use more training data (at least 100 images per class)")
print("• Balance your dataset (similar number of images per class)")
print("• Use data augmentation (enabled by default)")
print("• Start with pretrained models (PRETRAINED = True in config)")
print("• Experiment with different model architectures")
print("• Monitor training to detect overfitting")
print("• Use learning rate scheduling (enabled by default)")
print("• Increase image resolution for better accuracy (IMG_HEIGHT, IMG_WIDTH)")

print("\n11. EXAMPLE DATASETS")
print("-" * 70)
print("You can test the project with these popular datasets:")
print("• CIFAR-10: 10 classes, 60,000 images")
print("• Fashion-MNIST: 10 classes, 70,000 images")
print("• Caltech-101: 101 classes, 9,000 images")
print("• Dogs vs Cats: 2 classes, 25,000 images")
print("• ImageNet subset: Various classes")

print("\n12. TROUBLESHOOTING")
print("-" * 70)
print("Common issues:")
print("• CUDA out of memory: Reduce BATCH_SIZE in config.py")
print("• Low accuracy: Increase NUM_EPOCHS, check data quality")
print("• Slow training: Use GPU, reduce image size, or use smaller model")
print("• Import errors: Install missing packages with pip")

print("\n" + "=" * 70)
print("For more information, see README.md")
print("=" * 70 + "\n")

# Try importing modules to verify setup
print("Verifying module imports...")
try:
    from config import *
    print("✓ Configuration module loaded")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("  Make sure you've installed requirements: pip install -r requirements.txt")

print("\n✓ Example script completed!")
