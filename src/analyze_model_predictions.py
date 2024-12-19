import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from inference import IRSpectraInference
from spectral_analysis import SpectralData

def analyze_spectrum(inference_model, spectrum_data, output_dir, name_suffix=""):
    """Analyze a single spectrum and save visualizations"""
    # Get the spectrum and wavenumbers
    spectrum = np.mean(spectrum_data.data['spectra'], axis=1)
    wavenumbers = spectrum_data.data['wavenumbers']
    
    # Preprocess the spectrum
    processed_spectrum = inference_model.preprocess_spectrum(spectrum, wavenumbers)
    
    # Get predictions and feature maps
    treatment_pred, concentration_pred, feature_maps = inference_model.predict(processed_spectrum)
    
    # Clean up concentration values for filename
    def clean_value(val):
        if val is None:
            return "none"
        # Replace problematic characters
        return str(val).replace('/', '-').replace('\\', '-').replace(' ', '_').lower()
    
    # Create filename with actual and predicted values
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = (f"{clean_value(spectrum_data.cell_line)}_"
               f"{clean_value(spectrum_data.treatment)}_"
               f"{clean_value(spectrum_data.concentration)}_pred_"
               f"{clean_value(treatment_pred)}_"
               f"{clean_value(concentration_pred)}_"
               f"{timestamp}{clean_value(name_suffix)}.png")
    
    # Visualize and save feature maps
    inference_model.visualize_feature_maps(
        spectrum=spectrum,
        wavenumbers=wavenumbers,
        feature_maps=feature_maps,
        save_path=output_dir / filename
    )
    
    return {
        'actual_treatment': spectrum_data.treatment,
        'predicted_treatment': treatment_pred,
        'actual_concentration': spectrum_data.concentration,
        'predicted_concentration': concentration_pred
    }

def main():
    # Setup paths
    model_path = Path('models/best_model.pth')
    data_dir = Path('data/SRmicroFTIR_cellule')
    output_dir = Path('figures/model_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference model
    inference_model = IRSpectraInference(str(model_path))
    
    # Define cell lines, treatments, and concentrations to analyze
    cell_lines = ['HaCaT', 'HeLa']
    treatments = ['AMT', 'NP']
    concentrations = ['1-2', '1-8', '1-100']
    
    # Store results for summary
    results = []
    
    # Analyze control spectra first
    for cell_line in cell_lines:
        try:
            control_file = next(data_dir.glob(f'{cell_line}/*controllo*.dat'))
            spectrum_data = SpectralData(control_file)
            result = analyze_spectrum(
                inference_model, 
                spectrum_data, 
                output_dir,
                name_suffix="_control"
            )
            result['cell_line'] = cell_line
            results.append(result)
        except StopIteration:
            print(f"No control file found for {cell_line}")
    
    # Analyze treatment spectra
    for cell_line in cell_lines:
        for treatment in treatments:
            for concentration in concentrations:
                try:
                    # Find matching spectrum file
                    spectrum_file = next(data_dir.glob(
                        f'{cell_line}/{treatment}/*{concentration}*.dat'
                    ))
                    spectrum_data = SpectralData(spectrum_file)
                    result = analyze_spectrum(
                        inference_model, 
                        spectrum_data, 
                        output_dir
                    )
                    result['cell_line'] = cell_line
                    results.append(result)
                except StopIteration:
                    print(f"No file found for {cell_line}/{treatment}/{concentration}")
    
    # Generate summary report
    report_path = output_dir / f'analysis_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(report_path, 'w') as f:
        f.write("Model Analysis Summary\n")
        f.write("=====================\n\n")
        
        correct_treatment = sum(1 for r in results 
                              if r['actual_treatment'] == r['predicted_treatment'])
        correct_concentration = sum(1 for r in results 
                                  if r['actual_concentration'] == r['predicted_concentration'])
        
        f.write(f"Overall Accuracy:\n")
        f.write(f"Treatment: {correct_treatment/len(results):.2%}\n")
        f.write(f"Concentration: {correct_concentration/len(results):.2%}\n\n")
        
        f.write("Individual Predictions:\n")
        for result in results:
            f.write(f"\nCell Line: {result['cell_line']}\n")
            f.write(f"Treatment: {result['actual_treatment']} -> {result['predicted_treatment']}\n")
            f.write(f"Concentration: {result['actual_concentration']} -> "
                   f"{result['predicted_concentration']}\n")
            f.write("-" * 50)

if __name__ == "__main__":
    main() 