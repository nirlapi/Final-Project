"""
Model architectures for image classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """Custom CNN architecture for image classification"""
    
    def __init__(self, num_classes=10, img_channels=3):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet50(num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Get ResNet50 model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
    
    Returns:
        ResNet50 model
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def get_efficientnet_b0(num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Get EfficientNet-B0 model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
    
    Returns:
        EfficientNet-B0 model
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    return model


def get_vgg16(num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Get VGG16 model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
    
    Returns:
        VGG16 model
    """
    if pretrained:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg16(weights=None)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final layer
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    return model


def get_model(model_name='resnet50', num_classes=10, pretrained=True, freeze_backbone=False):
    """
    Get model by name
    
    Args:
        model_name (str): Name of the model
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        freeze_backbone (bool): Whether to freeze backbone layers
    
    Returns:
        Model
    """
    models_dict = {
        'resnet50': get_resnet50,
        'efficientnet': get_efficientnet_b0,
        'vgg16': get_vgg16,
        'custom_cnn': lambda nc, pt, fb: CustomCNN(num_classes=nc)
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(models_dict.keys())}")
    
    if model_name == 'custom_cnn':
        return models_dict[model_name](num_classes, pretrained, freeze_backbone)
    else:
        return models_dict[model_name](num_classes, pretrained, freeze_backbone)
