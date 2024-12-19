import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from model import IRSpectraModel
from data_module import IRSpectraDataset
from spectral.transforms import GlobalWindowResampler, Normalizer

class IRSpectraInference:
    def __init__(self, 
                 model_path: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize inference module
        
        Args:
            model_path: Path to saved model weights
            device: Device to run inference on
        """
        self.device = device
        
        # Initialize model
        self.model = IRSpectraModel(num_treatments=3, num_concentrations=3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        # Initialize transforms
        self.resampler = GlobalWindowResampler(target_size=1000, window=[800, 4000])
        self.normalizer = Normalizer()
        
        # Class mappings
        self.treatment_idx_to_name = {0: 'Control', 1: 'AMT', 2: 'NP'}
        self.concentration_idx_to_name = {0: '1/2', 1: '1/8', 2: '1/100'}
        
        # Hook for feature maps
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture feature maps"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        # Register hooks for each stage
        for i, stage in enumerate(self.model.backbone.stages):
            stage.register_forward_hook(hook_fn(f'stage_{i}'))
    
    def preprocess_spectrum(self, spectrum: np.ndarray, wavenumbers: np.ndarray) -> torch.Tensor:
        """Preprocess a single spectrum"""
        # Resample
        processed = self.resampler(spectrum, wavenumbers)
        # Normalize
        processed = self.normalizer(processed)
        # Convert to tensor
        tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, spectrum: torch.Tensor) -> Tuple[str, str, Dict[str, np.ndarray]]:
        """
        Run inference on a preprocessed spectrum
        
        Returns:
            Tuple of (treatment prediction, concentration prediction, feature maps)
        """
        # Forward pass
        treatment_logits, concentration_logits = self.model(spectrum)
        
        # Get predictions
        treatment_idx = treatment_logits.argmax(dim=1).item()
        concentration_idx = concentration_logits.argmax(dim=1).item()
        
        # Convert to class names
        treatment_pred = self.treatment_idx_to_name[treatment_idx]
        concentration_pred = self.concentration_idx_to_name[concentration_idx]
        
        # Get feature maps
        feature_maps = {k: v.cpu().numpy() for k, v in self.feature_maps.items()}
        
        return treatment_pred, concentration_pred, feature_maps
    
    def visualize_feature_maps(self,
                             spectrum: np.ndarray,
                             wavenumbers: np.ndarray,
                             feature_maps: Dict[str, np.ndarray],
                             save_path: Optional[str] = None):
        """
        Visualize feature maps along with input spectrum
        
        Args:
            spectrum: Original spectrum
            wavenumbers: Corresponding wavenumbers
            feature_maps: Dictionary of feature maps from each stage
            save_path: Optional path to save the plot
        """
        num_stages = len(feature_maps)
        fig, axes = plt.subplots(num_stages + 1, 1, figsize=(12, 4*num_stages + 4),
                                gridspec_kw={'height_ratios': [2] + [1]*num_stages})
        
        # Plot original spectrum
        axes[0].plot(wavenumbers, spectrum)
        axes[0].set_xlabel('Wavenumber (cm⁻¹)')
        axes[0].set_ylabel('Absorbance')
        axes[0].set_title('Input Spectrum')
        axes[0].grid(True, alpha=0.3)
        
        # Plot feature maps
        for i, (name, fmap) in enumerate(feature_maps.items(), 1):
            # Average across channels and normalize
            fmap = fmap.squeeze()  # Remove batch dimension
            if fmap.ndim > 2:  # If we have channel dimension
                fmap_avg = np.mean(fmap, axis=0)  # Average across channels
            else:
                fmap_avg = fmap
                
            # Normalize
            fmap_norm = (fmap_avg - fmap_avg.min()) / (fmap_avg.max() - fmap_avg.min() + 1e-8)
            
            # Plot heatmap
            sns.heatmap(fmap_norm[np.newaxis, :], ax=axes[i], cmap='viridis',
                       cbar_kws={'label': 'Normalized Activation'})
            axes[i].set_title(f'Feature Map - {name}')
            axes[i].set_ylabel('Channels')
            # Remove x-ticks for cleaner visualization
            axes[i].set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    """Example usage"""
    # Initialize inference module
    model_path = 'models/best_model.pth'
    inference = IRSpectraInference(model_path)
    
    # Load a test spectrum
    dataset = IRSpectraDataset('data/SRmicroFTIR_cellule')
    spectrum, _, _ = dataset[0]  # Get first spectrum
    
    # Get predictions and feature maps
    treatment_pred, concentration_pred, feature_maps = inference.predict(spectrum)
    print(f"Predicted treatment: {treatment_pred}")
    print(f"Predicted concentration: {concentration_pred}")
    
    # Visualize feature maps
    inference.visualize_feature_maps(
        spectrum=spectrum.squeeze().numpy(),
        wavenumbers=np.linspace(800, 4000, 1000),
        feature_maps=feature_maps,
        save_path='feature_maps.png'
    )

if __name__ == '__main__':
    main() 