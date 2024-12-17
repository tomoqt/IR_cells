import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
from spectral_analysis import SpectralData
from spectral.transforms import GlobalWindowResampler, Normalizer

class IRSpectraDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform_params: Dict = None,
                 split: str = 'train'):
        """
        Args:
            data_dir: Path to data directory
            transform_params: Dictionary with preprocessing parameters
            split: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Initialize transforms
        self.transform_params = transform_params or {
            'target_size': 1000,
            'window': [800, 4000],
            'normalize': True
        }
        
        self.resampler = GlobalWindowResampler(
            target_size=self.transform_params['target_size'],
            window=self.transform_params['window']
        )
        self.normalizer = Normalizer() if self.transform_params['normalize'] else None
        
        # Load and preprocess all data
        self.spectra_data = self._load_all_spectra()
        
        # Create label encodings
        self.treatment_to_idx = {
            'Control': 0,
            'AMT': 1,
            'NP': 2
        }
        
        self.concentration_to_float = {
            '1-2': 0.5,
            '1-8': 0.125,
            '1-100': 0.01,
            None: 0.0  # For control samples
        }
        
    def _load_all_spectra(self) -> List[Dict]:
        all_data = []
        
        # Load all .dat files recursively
        for file_path in self.data_dir.rglob('*.dat'):
            spec_data = SpectralData(file_path)
            
            # Get raw spectra and preprocess
            wavenumbers = spec_data.data['wavenumbers']
            spectra = spec_data.data['spectra'].T
            
            for spectrum in spectra:
                # Apply preprocessing
                processed_spectrum = self.resampler(spectrum, wavenumbers)
                if self.normalizer:
                    processed_spectrum = self.normalizer(processed_spectrum)
                
                all_data.append({
                    'spectrum': processed_spectrum,
                    'cell_line': spec_data.cell_line,
                    'treatment': spec_data.treatment,
                    'concentration': spec_data.concentration
                })
        
        return all_data
    
    def __len__(self) -> int:
        return len(self.spectra_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.spectra_data[idx]
        
        # Prepare input spectrum
        spectrum = torch.FloatTensor(item['spectrum']).unsqueeze(0)  # Add channel dimension
        
        # Prepare classification target (treatment type)
        treatment_idx = self.treatment_to_idx[item['treatment']]
        treatment_target = torch.LongTensor([treatment_idx])
        
        # Prepare regression target (concentration)
        concentration = self.concentration_to_float[item['concentration']]
        concentration_target = torch.FloatTensor([concentration])
        
        return spectrum, treatment_target, concentration_target

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    transform_params: Optional[Dict] = None,
    train_val_split: float = 0.8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create full dataset
    full_dataset = IRSpectraDataset(data_dir, transform_params, split='train')
    
    # Split into train and validation
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader 