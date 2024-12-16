# IR Cell Spectral Analysis

## Introduction
This project analyzes infrared spectroscopy data from HaCaT (normal) and HeLa (cancer) cells treated with different concentrations of aminopterine in both molecular (AMT) and gold nanoparticle-conjugated (NP) forms. The analysis aims to understand the spectral differences between cell types and their responses to treatments.

## Installation

```bash
pip install -r requirements.txt
```

### Requirements 

Required packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- umap-learn
- torch (for future deep learning implementation)


### Project Structure

IR_cells/
├── data/
│ └── SRmicroFTIR_cellule/
│ ├── HaCaT/ # Normal cells
│ │ ├── AMT/ # Molecular aminopterine
│ │ └── NP/ # Nanoparticle conjugated
│ └── HeLa/ # Cancer cells
│ ├── AMT/
│ └── NP/
├── src/
│ ├── spectral_analysis.py
│ └── spectral_analysis_dim_reduction.py
├── figures/
│ └── dim_reduction/
└── legacy_code/
├── pipeline.py
└── convnext1d.py



## Current Analyses

### 1. Basic Spectral Analysis
- Direct visualization of IR spectra
- Comparison between cell lines and treatments
- Statistical analysis of spectral variations

#### Results

##### HaCaT AMT Comparison
![HaCaT AMT Comparison](figures/simple_plots/HaCaT_AMT_20241216_120035.png)
[Discussion needed]

##### HaCaT NP Comparison
![HaCaT NP Comparison](figures/simple_plots/HaCaT_NP_20241216_120037.png)
[Discussion needed]

##### HeLa AMT Comparison
![HeLa AMT Comparison](figures/simple_plots/HeLa_AMT_20241216_120036.png)
[Discussion needed]

##### HeLa NP Comparison
![HeLa NP Comparison](figures/simple_plots/HeLa_NP_20241216_120038.png)
[Discussion needed]

##### All Cell Lines Comparison
![All Cell Lines Comparison](figures/simple_plots/All_Comparison_20241216_120038.png)
[Discussion needed]

### 2. Dimensionality Reduction Analysis
- Principal Component Analysis (PCA)
- t-SNE Analysis
- UMAP Analysis
- Multiple visualization schemes:
  - By treatment type
  - By cell line
  - By concentration
  - Control vs treated

#### Results
##### PCA Analysis
![PCA Explained Variance](figures/dim_reduction/pca_explained_variance.png)
[Discussion needed]

![PCA by Treatment](figures/dim_reduction/pca_by_treatment.png)
[Discussion needed]

##### t-SNE Analysis
![t-SNE by Cell Line](figures/dim_reduction/tsne_by_cell_line.png)
[Discussion needed]

##### UMAP Analysis
![UMAP by Concentration](figures/dim_reduction/umap_by_concentration.png)
[Discussion needed]

## Development Roadmap

### Current Development
1. Spectral preprocessing pipeline
2. Basic statistical analysis
3. Dimensionality reduction techniques

### Planned Development
1. Implementation of deep learning models:
   - Adaptation of ConvNeXt architecture for 1D spectral data
   - Classification of cell treatments
   - Feature extraction and interpretation

2. Enhanced preprocessing pipeline:
   - Global window resampling
   - Normalization techniques
   - Data augmentation strategies

### Deep Learning Implementation
The project will utilize the ConvNeXt1D architecture (from `legacy_code/convnext1d.py`) for:
- Classification of treatment types
- Identification of cell lines
- Prediction of treatment concentrations

The preprocessing pipeline (`legacy_code/pipeline.py`) will be integrated to ensure:
- Consistent data preprocessing
- Proper window selection
- Appropriate normalization

## Usage

### Basic Spectral Analysis
python
from src.spectral_analysis import SpectralData, main
main()

### Dimensionality Reduction Analysis
python
from src.spectral_analysis_dim_reduction import SpectralAnalysis
analysis = SpectralAnalysis("path/to/data")
analysis.run_analysis()
