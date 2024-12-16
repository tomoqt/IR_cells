from typing import List, Optional
from .transforms import GlobalWindowResampler, Normalizer

class PreprocessingPipeline:
    """Builder for preprocessing pipeline"""
    
    @staticmethod
    def create_standard_pipeline(
        resample_size: int = 1000,
        normalize: bool = True,
        window: List[float] = [200, 4000]
    ) -> List:
        """Create standard preprocessing pipeline with global window
        
        Args:
            resample_size: Target number of points for resampling
            normalize: Whether to normalize spectra
            window: Global window [min_wavelength, max_wavelength]
            
        Returns:
            List of preprocessing transforms
        """
        transforms = []
        
        # Always include global window resampler
        transforms.append(GlobalWindowResampler(
            target_size=resample_size,
            window=window
        ))
        
        if normalize:
            transforms.append(Normalizer())
            
        return transforms 