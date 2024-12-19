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
                             save_path: Optional[str] = None,
                             num_bins: int = 100):
        """
        Visualize feature maps and their aggregated activation patterns over the input spectrum
        
        Args:
            spectrum: Original spectrum
            wavenumbers: Corresponding wavenumbers
            feature_maps: Dictionary of feature maps from each stage
            save_path: Optional path to save the plot
            num_bins: Number of bins for aggregating activations over wavenumbers
        """
        # Create figure
        fig, (ax_spectrum, ax_dist) = plt.subplots(2, 1, figsize=(12, 8),
                                                  gridspec_kw={'height_ratios': [2, 1]},
                                                  sharex=True)
        
        # Plot original spectrum
        ax_spectrum.plot(wavenumbers, spectrum, 'b-', label='Spectrum', alpha=0.7)
        ax_spectrum.set_xlabel('Wavenumber (cm⁻¹)')
        ax_spectrum.set_ylabel('Absorbance')
        ax_spectrum.set_title('Input Spectrum with Model Attention')
        ax_spectrum.grid(True, alpha=0.3)
        
        # Aggregate activations across all feature maps
        bin_edges = np.linspace(wavenumbers.min(), wavenumbers.max(), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        aggregated_activations = np.zeros(num_bins)
        
        # Process each feature map
        for name, fmap in feature_maps.items():
            # Remove batch dimension and handle channel dimension
            fmap = fmap.squeeze(0)  # Remove batch dimension
            
            if fmap.ndim == 3:  # If shape is (channels, height, width)
                # For each channel, get the activation pattern
                for channel in range(fmap.shape[0]):
                    channel_activations = fmap[channel]
                    # Normalize channel activations
                    channel_activations = (channel_activations - channel_activations.min()) / \
                                        (channel_activations.max() - channel_activations.min() + 1e-8)
                    
                    # Interpolate to match wavenumber scale
                    interp_activations = np.interp(
                        bin_centers,
                        np.linspace(wavenumbers.min(), wavenumbers.max(), channel_activations.shape[-1]),
                        channel_activations.mean(axis=0)  # Average across height if needed
                    )
                    
                    # Add to aggregate
                    aggregated_activations += interp_activations
            
            elif fmap.ndim == 2:  # If already 2D
                # Normalize activations
                fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                
                # Interpolate to match wavenumber scale
                interp_activations = np.interp(
                    bin_centers,
                    np.linspace(wavenumbers.min(), wavenumbers.max(), fmap_norm.shape[-1]),
                    fmap_norm.mean(axis=0)
                )
                
                # Add to aggregate
                aggregated_activations += interp_activations
        
        # Normalize final aggregated activations
        aggregated_activations = (aggregated_activations - aggregated_activations.min()) / \
                               (aggregated_activations.max() - aggregated_activations.min())
        
        # Plot aggregated attention
        ax_dist.fill_between(bin_centers, 0, aggregated_activations, 
                            alpha=0.3, color='r', label='Aggregated Attention')
        ax_dist.plot(bin_centers, aggregated_activations, 'r-', alpha=0.7)
        ax_dist.set_ylabel('Normalized Activation')
        ax_dist.set_xlabel('Wavenumber (cm⁻¹)')
        ax_dist.grid(True, alpha=0.3)
        
        # Add attention overlay on spectrum
        ax_spectrum_twin = ax_spectrum.twinx()
        ax_spectrum_twin.fill_between(bin_centers, 0, aggregated_activations, 
                                    alpha=0.2, color='r', label='Model Attention')
        ax_spectrum_twin.plot(bin_centers, aggregated_activations, 'r-', alpha=0.5)
        ax_spectrum_twin.set_ylabel('Normalized Activation', color='r')
        ax_spectrum_twin.tick_params(axis='y', labelcolor='r')
        
        # Add legends
        ax_spectrum.legend(loc='upper left')
        ax_spectrum_twin.legend(loc='upper right')
        
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