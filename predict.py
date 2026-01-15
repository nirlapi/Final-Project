"""
Prediction/Inference script for image classification
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config import *
from dataset import get_transforms
from model import get_model


def load_model(model_path, model_name, num_classes, device):
    """
    Load trained model
    
    Args:
        model_path: Path to model checkpoint
        model_name: Name of the model architecture
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = get_model(model_name, num_classes, pretrained=False, freeze_backbone=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded from {model_path}')
    print(f'Model was trained for {checkpoint.get("epoch", "unknown")} epochs')
    print(f'Best validation accuracy: {checkpoint.get("val_acc", "unknown")}')
    
    return model


def predict_single_image(model, image_path, transform, device, class_names=None, top_k=3):
    """
    Predict class for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        transform: Image transformations
        device: Device to run inference on
        class_names: List of class names (optional)
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Apply transformations
    if transform:
        transformed = transform(image=image_array)
        image_tensor = transformed['image'].unsqueeze(0)
    else:
        raise ValueError("Transform is required")
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.size(1)))
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
    
    # Format results
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = class_names[idx] if class_names else str(idx)
        predictions.append({
            'class': class_name,
            'class_id': int(idx),
            'probability': float(prob)
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'top_prediction': predictions[0]
    }


def predict_batch(model, image_paths, transform, device, class_names=None, top_k=3):
    """
    Predict classes for multiple images
    
    Args:
        model: Trained model
        image_paths: List of image paths
        transform: Image transformations
        device: Device to run inference on
        class_names: List of class names (optional)
        top_k: Number of top predictions to return
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for image_path in image_paths:
        try:
            result = predict_single_image(model, image_path, transform, device, class_names, top_k)
            results.append(result)
            print(f"✓ Processed: {image_path}")
        except Exception as e:
            print(f"✗ Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def main(args):
    """Main prediction function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
    print(f'Using device: {device}')
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
        print(f'Loaded {len(class_names)} class names')
    elif args.num_classes:
        class_names = [f'class_{i}' for i in range(args.num_classes)]
    else:
        print('Warning: No class names provided. Using class indices.')
    
    # Determine number of classes
    num_classes = len(class_names) if class_names else args.num_classes
    if not num_classes:
        raise ValueError('Please provide either --class_names or --num_classes')
    
    # Load model
    print(f'\nLoading model from {args.model_path}...')
    model = load_model(args.model_path, args.model_name, num_classes, device)
    
    # Get transform
    transform = get_transforms(IMG_HEIGHT, IMG_WIDTH, augment=False)
    
    # Predict
    if args.image_path:
        # Single image prediction
        print(f'\nPredicting for single image: {args.image_path}')
        result = predict_single_image(
            model, args.image_path, transform, device, class_names, args.top_k
        )
        
        print('\n' + '='*60)
        print(f"Image: {result['image_path']}")
        print('='*60)
        print(f"\nTop {args.top_k} Predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"{i}. {pred['class']}: {pred['probability']*100:.2f}%")
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'\nResults saved to {args.output}')
    
    elif args.image_dir:
        # Batch prediction
        print(f'\nPredicting for images in directory: {args.image_dir}')
        
        # Get all images in directory
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_paths = []
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        print(f'Found {len(image_paths)} images')
        
        if len(image_paths) == 0:
            print('No images found in directory')
            return
        
        results = predict_batch(
            model, image_paths, transform, device, class_names, args.top_k
        )
        
        # Print summary
        print('\n' + '='*60)
        print('PREDICTION SUMMARY')
        print('='*60)
        successful = sum(1 for r in results if 'error' not in r)
        print(f'Successfully processed: {successful}/{len(results)} images')
        
        # Save results if output path provided
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f'\nResults saved to {args.output}')
    
    else:
        print('Error: Please provide either --image_path or --image_dir')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict image classes')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                       help='Model architecture name')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                       help='Path to directory containing images for batch prediction')
    parser.add_argument('--class_names', type=str,
                       help='Path to JSON file containing class names')
    parser.add_argument('--num_classes', type=int,
                       help='Number of classes (if class_names not provided)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    parser.add_argument('--output', type=str,
                       help='Path to save prediction results (JSON)')
    
    args = parser.parse_args()
    main(args)
