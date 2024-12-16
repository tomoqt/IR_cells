import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

class SpectralData:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.cell_line = self._get_cell_line()
        self.treatment = self._get_treatment()
        self.concentration = self._get_concentration()
        self.data = self._load_data()
        
    def _get_cell_line(self):
        return 'HaCaT' if 'HaCaT' in self.file_path.name else 'HeLa'
    
    def _get_treatment(self):
        if 'AMT' in self.file_path.name:
            return 'AMT'
        elif 'NP' in self.file_path.name:
            return 'NP'
        else:
            return 'Control'
    
    def _get_concentration(self):
        if self.treatment == 'Control':
            return None
        for conc in ['1-2', '1-8', '1-100']:
            if conc in self.file_path.name:
                return conc
        return None
    
    def _load_data(self):
        data = pd.read_csv(self.file_path, delim_whitespace=True, header=None)
        # Reverse wavenumbers and spectra to have increasing wavenumbers
        return {
            'wavenumbers': data.iloc[::-1, 0].values,  # Reverse wavenumbers
            'spectra': data.iloc[::-1, 1:].values      # Reverse spectra to match
        }
    
    def plot_average_spectrum(self, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_spectrum = np.mean(self.data['spectra'], axis=1)
        std_spectrum = np.std(self.data['spectra'], axis=1)
        
        ax.plot(self.data['wavenumbers'], mean_spectrum, 
                label=label or f"{self.cell_line}_{self.treatment}_{self.concentration}")
        ax.fill_between(self.data['wavenumbers'], 
                       mean_spectrum - std_spectrum,
                       mean_spectrum + std_spectrum,
                       alpha=0.2)
        
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Absorbance')
        ax.legend()
        
        return ax

def main():
    # Define the data directory
    data_dir = Path(r'D:\Experiments\IR_cells\data\SRmicroFTIR_cellule')
    
    # Create output directory for figures if it doesn't exist
    output_dir = Path(r'D:\Experiments\IR_cells\figures')
    output_dir.mkdir(exist_ok=True)
    
    # Create figures for different comparisons
    figures = {
        'HaCaT_AMT': plt.subplots(figsize=(12, 6)),
        'HeLa_AMT': plt.subplots(figsize=(12, 6)),
        'HaCaT_NP': plt.subplots(figsize=(12, 6)),
        'HeLa_NP': plt.subplots(figsize=(12, 6)),
        'All_Comparison': plt.subplots(figsize=(12, 6))
    }
    
    # Process HaCaT AMT concentrations
    hacat_amt_files = list(data_dir.glob('HaCaT/AMT/*.dat'))
    for file_path in hacat_amt_files:
        spec_data = SpectralData(file_path)
        spec_data.plot_average_spectrum(ax=figures['HaCaT_AMT'][1])
        spec_data.plot_average_spectrum(
            ax=figures['All_Comparison'][1], 
            label=f'HaCaT_{spec_data.treatment}_{spec_data.concentration}'
        )
    
    # Process HeLa AMT concentrations
    hela_amt_files = list(data_dir.glob('HeLa/AMT/*.dat'))
    for file_path in hela_amt_files:
        spec_data = SpectralData(file_path)
        spec_data.plot_average_spectrum(ax=figures['HeLa_AMT'][1])
        spec_data.plot_average_spectrum(
            ax=figures['All_Comparison'][1], 
            label=f'HeLa_{spec_data.treatment}_{spec_data.concentration}'
        )
    
    # Process HaCaT NP concentrations
    hacat_np_files = list(data_dir.glob('HaCaT/NP/*.dat'))
    for file_path in hacat_np_files:
        spec_data = SpectralData(file_path)
        spec_data.plot_average_spectrum(ax=figures['HaCaT_NP'][1])
        spec_data.plot_average_spectrum(
            ax=figures['All_Comparison'][1], 
            label=f'HaCaT_{spec_data.treatment}_{spec_data.concentration}'
        )
    
    # Process HeLa NP concentrations
    hela_np_files = list(data_dir.glob('HeLa/NP/*.dat'))
    for file_path in hela_np_files:
        spec_data = SpectralData(file_path)
        spec_data.plot_average_spectrum(ax=figures['HeLa_NP'][1])
        spec_data.plot_average_spectrum(
            ax=figures['All_Comparison'][1], 
            label=f'HeLa_{spec_data.treatment}_{spec_data.concentration}'
        )
    
    # Try to load control data if available
    try:
        hela_control = SpectralData(data_dir / 'HeLa/HeLa-controllo_complete-no-baseline.dat')
        hela_control.plot_average_spectrum(
            ax=figures['All_Comparison'][1], 
            label='HeLa_Control'
        )
    except FileNotFoundError:
        print("HeLa control file not found")
    
    # Set titles and adjust layouts
    titles = {
        'HaCaT_AMT': 'HaCaT Cells - Different AMT Concentrations',
        'HeLa_AMT': 'HeLa Cells - Different AMT Concentrations',
        'HaCaT_NP': 'HaCaT Cells - Different NP Concentrations',
        'HeLa_NP': 'HeLa Cells - Different NP Concentrations',
        'All_Comparison': 'All Cell Lines and Treatments Comparison'
    }
    
    # Configure and save each figure
    for name, (fig, ax) in figures.items():
        ax.set_title(titles[name])
        ax.grid(True, alpha=0.3)
        ax.set_xlim(800, 4000)  # Changed from (4000, 800) to (800, 4000)
        fig.tight_layout()
        
        # Save figure with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{name}_{timestamp}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
    
    plt.show()

if __name__ == "__main__":
    main() 