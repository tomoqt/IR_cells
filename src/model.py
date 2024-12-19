import torch
import torch.nn as nn
from spectral.convnext1d import ConvNeXt1D

class IRSpectraModel(nn.Module):
    def __init__(self, 
                 num_treatments: int = 3,
                 num_concentrations: int = 3,
                 pretrained: bool = False):
        super().__init__()
        
        # Initialize backbone
        self.backbone = ConvNeXt1D(
            in_chans=1,
            num_classes=512,  # Use as feature extractor
            depths=[3, 3, 9, 3],
            dims=[96, 192, 384, 768]
        )
        
        # Classification head for treatment type
        self.treatment_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_treatments)
        )
        
        # Changed: Classification head for concentration instead of regression
        self.concentration_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_concentrations)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Get predictions
        treatment_logits = self.treatment_classifier(features)
        concentration_logits = self.concentration_classifier(features)
        
        return treatment_logits, concentration_logits 