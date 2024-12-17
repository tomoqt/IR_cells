import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import Optional, Tuple, List, Union
import logging

class GlobalWindowResampler:
    """Resample spectrum to global window with fixed size"""
    
    def __init__(self, target_size: int, window: List[float] = [200, 4000], method: str = 'cubic'):
        """
        Args:
            target_size: Desired number of points
            window: Global window [min_wavelength, max_wavelength]
            method: Interpolation method ('cubic' or 'linear')
        """
        self.target_size = target_size
        self.window = self.validate_window(window)
        self.method = method
        self.global_domain = np.linspace(window[0], window[1], target_size)
        logging.debug(f"Initialized GlobalWindowResampler with window {self.window} and {target_size} points")
    
    def validate_window(self, window: List[float]) -> List[float]:
        """Validate and normalize window parameters"""
        if not isinstance(window, (list, tuple, np.ndarray)) or len(window) != 2:
            raise ValueError("Window must be a list/tuple of [min, max] wavenumbers")
        if window[0] >= window[1]:
            raise ValueError(f"Invalid window range: [{window[0]}, {window[1]}]")
        return [float(window[0]), float(window[1])]
    
    def determine_window_bounds(self, domain: np.ndarray) -> Tuple[float, float]:
        """Determine effective window bounds based on data domain and global window"""
        min_wavenumber = max(np.min(domain), self.window[0])
        max_wavenumber = min(np.max(domain), self.window[1])
        logging.debug(f"Effective window bounds: [{min_wavenumber}, {max_wavenumber}]")
        return min_wavenumber, max_wavenumber
    
    def __call__(self, spectrum: np.ndarray, domain: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Process single spectrum efficiently"""
        # Convert inputs to numpy arrays if needed
        if isinstance(domain, list):
            domain = np.array(domain)
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
            
        # Sort wavenumbers and spectrum if not already sorted
        if not np.all(np.diff(domain) > 0):
            logging.debug("Sorting wavenumbers in ascending order")
            sort_idx = np.argsort(domain)
            domain = domain[sort_idx]
            spectrum = spectrum[sort_idx]
            
        logging.debug(f"Input spectrum shape: {spectrum.shape}, domain range: [{domain[0]}, {domain[-1]}]")
            
        # Create padded spectrum initialized with zeros
        padded_spectrum = np.zeros(self.target_size)
        
        # Determine effective window bounds
        min_wave, max_wave = self.determine_window_bounds(domain)
        
        # Find indices where original domain falls within global window
        start_idx = np.searchsorted(self.global_domain, min_wave)
        end_idx = np.searchsorted(self.global_domain, max_wave, side='right')
        
        if start_idx < end_idx:
            section_domain = self.global_domain[start_idx:end_idx]
            
            # Filter domain and spectrum to window
            mask = (domain >= min_wave) & (domain <= max_wave)
            window_domain = domain[mask]
            window_spectrum = spectrum[mask]
            
            if len(window_domain) > 0:
                # Verify the data is still sorted after windowing
                if not np.all(np.diff(window_domain) > 0):
                    logging.warning("Data not strictly increasing after windowing, sorting again")
                    sort_idx = np.argsort(window_domain)
                    window_domain = window_domain[sort_idx]
                    window_spectrum = window_spectrum[sort_idx]
                
                # Use CubicSpline if we have enough points, otherwise linear
                if len(window_domain) > 3 and self.method == 'cubic':
                    try:
                        interpolator = CubicSpline(window_domain, window_spectrum, extrapolate=False)
                        interpolated_values = interpolator(section_domain)
                    except ValueError as e:
                        logging.warning(f"CubicSpline failed, falling back to linear: {str(e)}")
                        interpolator = interp1d(window_domain, window_spectrum, kind='linear', 
                                             bounds_error=False, fill_value=0)
                        interpolated_values = interpolator(section_domain)
                else:
                    interpolator = interp1d(window_domain, window_spectrum, kind='linear', 
                                         bounds_error=False, fill_value=0)
                    interpolated_values = interpolator(section_domain)
                
                # Remove any NaN values that might have been introduced
                interpolated_values = np.nan_to_num(interpolated_values, 0)
                padded_spectrum[start_idx:end_idx] = interpolated_values
                
                logging.debug(f"Resampled spectrum range: [{np.min(interpolated_values)}, {np.max(interpolated_values)}]")
            else:
                logging.warning("No data points found within window")
        else:
            logging.warning(f"No overlap between data domain [{domain[0]}, {domain[-1]}] "
                          f"and window [{self.window[0]}, {self.window[1]}]")
        
        return padded_spectrum

class Normalizer:
    """Normalize spectrum to [0,1] range"""
    
    def __call__(self, spectrum: Union[np.ndarray, List[float]], domain: Optional[np.ndarray] = None) -> np.ndarray:
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
            
        non_zero_mask = spectrum != 0
        if non_zero_mask.any():
            min_val = np.min(spectrum[non_zero_mask])
            max_val = np.max(spectrum[non_zero_mask])
            
            if not np.isclose(min_val, max_val):
                spectrum = spectrum.copy()
                spectrum[non_zero_mask] = (spectrum[non_zero_mask] - min_val) / (max_val - min_val)
                logging.debug(f"Normalized spectrum range: [{min_val}, {max_val}] -> [0, 1]")
            else:
                logging.warning("All non-zero values are identical, skipping normalization")
        else:
            logging.warning("Spectrum contains only zeros, skipping normalization")
            
        return spectrum