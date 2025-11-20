#!/usr/bin/env python3
"""
Instance Classifier for Tree Species Recognition

Simple classifier that takes a masked tree crown image and predicts species.
No complexity. No special cases. Just extract features and classify.

Two approaches:
1. Masked CNN: Apply mask to image, run through CNN encoder
2. Feature + Mask: Concatenate image features with mask features
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MaskedInstanceClassifier(nn.Module):
    """
    Classify tree species from masked instance regions.
    
    Architecture:
    1. Apply binary mask to image (element-wise multiplication)
    2. Extract features using CNN encoder (e.g., ResNet, EfficientNet)
    3. Global average pooling over spatial dimensions
    4. Classification head (MLP)
    
    Simple. Direct. No tricks.
    
    Parameters
    ----------
    num_classes : int
        Number of tree species classes
    encoder_name : str
        Name of timm model to use as encoder (default: 'resnet50')
    pretrained : bool
        Use ImageNet pretrained weights (default: True)
    dropout : float
        Dropout rate in classifier head (default: 0.3)
    use_mask_features : bool
        If True, extract features from mask itself and concatenate (default: False)
    """
    
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_mask_features: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_mask_features = use_mask_features
        
        # Image encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling (we'll do it ourselves)
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_features = self.encoder(dummy_input)
            feature_dim = dummy_features.shape[1]
        
        print(f"Encoder '{encoder_name}' feature dimension: {feature_dim}")
        
        # Optional mask encoder (same architecture, separate weights)
        if use_mask_features:
            self.mask_encoder = timm.create_model(
                encoder_name,
                pretrained=False,
                num_classes=0,
                global_pool='',
                in_chans=1,  # Single channel mask
            )
            classifier_input_dim = feature_dim * 2  # Concatenate image + mask features
        else:
            self.mask_encoder = None
            classifier_input_dim = feature_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        image : torch.Tensor
            RGB image, shape (B, 3, H, W)
        mask : torch.Tensor
            Binary mask, shape (B, 1, H, W), values in [0, 1]
        
        Returns
        -------
        logits : torch.Tensor
            Classification logits, shape (B, num_classes)
        """
        # Apply mask to image
        masked_image = image * mask
        
        # Extract image features
        image_features = self.encoder(masked_image)  # (B, C, H', W')
        image_features = F.adaptive_avg_pool2d(image_features, 1)  # (B, C, 1, 1)
        image_features = image_features.flatten(1)  # (B, C)
        
        # Optionally extract mask features
        if self.use_mask_features:
            mask_features = self.mask_encoder(mask)
            mask_features = F.adaptive_avg_pool2d(mask_features, 1)
            mask_features = mask_features.flatten(1)
            
            # Concatenate
            features = torch.cat([image_features, mask_features], dim=1)
        else:
            features = image_features
        
        # Classify
        logits = self.classifier(features)
        
        return logits


class SimpleInstanceClassifier(nn.Module):
    """
    Even simpler version: just crop to bbox and classify.
    No explicit mask multiplication.
    
    Assumes input is already cropped to instance bbox.
    """
    
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Image encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',  # Use built-in global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_features = self.encoder(dummy_input)
            feature_dim = dummy_features.shape[1]
        
        print(f"Encoder '{encoder_name}' feature dimension: {feature_dim}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        image : torch.Tensor
            RGB image (already cropped to instance), shape (B, 3, H, W)
        
        Returns
        -------
        logits : torch.Tensor
            Classification logits, shape (B, num_classes)
        """
        features = self.encoder(image)  # (B, feature_dim)
        logits = self.classifier(features)
        return logits


def create_instance_classifier(
    num_classes: int,
    model_type: str = 'masked',
    encoder_name: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.3,
    use_mask_features: bool = False,
) -> nn.Module:
    """
    Factory function to create instance classifier.
    
    Parameters
    ----------
    num_classes : int
        Number of tree species classes
    model_type : str
        Type of classifier: 'masked' or 'simple'
    encoder_name : str
        Name of timm model to use as encoder
    pretrained : bool
        Use ImageNet pretrained weights
    dropout : float
        Dropout rate
    use_mask_features : bool
        Only for 'masked' type: whether to use mask features
    
    Returns
    -------
    model : nn.Module
        Instance classifier model
    """
    if model_type == 'masked':
        return MaskedInstanceClassifier(
            num_classes=num_classes,
            encoder_name=encoder_name,
            pretrained=pretrained,
            dropout=dropout,
            use_mask_features=use_mask_features,
        )
    elif model_type == 'simple':
        return SimpleInstanceClassifier(
            num_classes=num_classes,
            encoder_name=encoder_name,
            pretrained=pretrained,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Common encoder choices (sorted by size)
ENCODER_CONFIGS = {
    # Small/fast models
    'mobilenetv3_small': {'params': '2.5M', 'speed': 'fast'},
    'efficientnet_b0': {'params': '5.3M', 'speed': 'fast'},
    'resnet18': {'params': '11.7M', 'speed': 'fast'},
    
    # Medium models (good balance)
    'resnet34': {'params': '21.8M', 'speed': 'medium'},
    'resnet50': {'params': '25.6M', 'speed': 'medium'},
    'efficientnet_b2': {'params': '9.2M', 'speed': 'medium'},
    
    # Large/accurate models
    'resnet101': {'params': '44.5M', 'speed': 'slow'},
    'efficientnet_b4': {'params': '19.3M', 'speed': 'slow'},
    'convnext_small': {'params': '50.2M', 'speed': 'slow'},
}

