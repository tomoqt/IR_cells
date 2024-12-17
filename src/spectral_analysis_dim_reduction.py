import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import itertools
from spectral_analysis import SpectralData

class SpectralAnalysis:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(r'D:/Experiments/IR_cells/figures/dim_reduction')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.spectra_data = self._load_all_spectra()
        
    def _load_all_spectra(self):
        """Load and organize all spectra with their metadata"""
        all_data = []
        
        # Load all .dat files recursively
        for file_path in self.data_dir.rglob('*.dat'):
            spec_data = SpectralData(file_path)
            spectra = spec_data.data['spectra'].T  # Transpose to have samples as rows
            
            # Create metadata for each spectrum
            for i in range(spectra.shape[0]):
                all_data.append({
                    'spectrum': spectra[i],
                    'cell_line': spec_data.cell_line,
                    'treatment': spec_data.treatment,
                    'concentration': spec_data.concentration,
                    'is_control': spec_data.treatment == 'Control'
                })
        
        return pd.DataFrame(all_data)
    
    def preprocess_spectra(self):
        """Preprocess spectra using StandardScaler"""
        X = np.vstack(self.spectra_data['spectrum'].values)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def perform_pca(self, X_scaled):
        """Perform PCA and return results"""
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance ratio
        exp_var_ratio = pca.explained_variance_ratio_
        cum_exp_var_ratio = np.cumsum(exp_var_ratio)
        
        # Plot explained variance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(exp_var_ratio) + 1), cum_exp_var_ratio, 'bo-')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance Ratio')
        ax.grid(True)
        fig.savefig(self.output_dir / 'pca_explained_variance.png', dpi=300, bbox_inches='tight')
        
        return X_pca, pca
    
    def perform_tsne(self, X_scaled):
        """Perform t-SNE"""
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        return X_tsne
    
    def perform_umap(self, X_scaled):
        """Perform UMAP"""
        umap = UMAP(random_state=42)
        X_umap = umap.fit_transform(X_scaled)
        return X_umap
    
    def plot_dim_reduction(self, X_reduced, method_name, color_by='treatment'):
        """Plot dimensionality reduction results with different color schemes"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color scheme based on the selected attribute
        unique_values = self.spectra_data[color_by].unique()
        colors = sns.color_palette('husl', n_colors=len(unique_values))
        color_dict = dict(zip(unique_values, colors))
        
        # Plot points
        for value in unique_values:
            mask = self.spectra_data[color_by] == value
            ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                      label=value, alpha=0.7,
                      c=[color_dict[value]])
        
        ax.set_xlabel(f'{method_name} Component 1')
        ax.set_ylabel(f'{method_name} Component 2')
        ax.set_title(f'{method_name} - Color by {color_by}')
        ax.legend()
        
        # Save figure
        filename = f'{method_name.lower()}_by_{color_by}.png'
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig, ax
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        # Preprocess data
        X_scaled = self.preprocess_spectra()
        
        # Perform dimensionality reduction
        X_pca, pca_obj = self.perform_pca(X_scaled)
        X_tsne = self.perform_tsne(X_scaled)
        X_umap = self.perform_umap(X_scaled)
        
        # Plot results with different color schemes
        color_schemes = ['treatment', 'cell_line', 'concentration', 'is_control']
        
        for method_name, X_reduced in [('PCA', X_pca[:, :2]), 
                                     ('t-SNE', X_tsne), 
                                     ('UMAP', X_umap)]:
            for color_by in color_schemes:
                if color_by in self.spectra_data.columns:
                    self.plot_dim_reduction(X_reduced, method_name, color_by)
        
        # Additional PCA analysis
        self.plot_pca_loadings(pca_obj, X_pca)
        
        plt.show()
    
    def plot_pca_loadings(self, pca, X_pca):
        """Plot PCA loadings and score plots"""
        # Plot loadings
        fig, ax = plt.subplots(figsize=(12, 8))
        loadings = pca.components_[:2].T
        
        # Create a scatter plot of loadings
        ax.scatter(loadings[:, 0], loadings[:, 1], alpha=0.5)
        ax.set_xlabel('PC1 Loadings')
        ax.set_ylabel('PC2 Loadings')
        ax.set_title('PCA Loadings Plot')
        
        # Save loadings plot
        fig.savefig(self.output_dir / 'pca_loadings.png', dpi=300, bbox_inches='tight')

def main():
    data_dir = Path(r'D:\Experiments\IR_cells\data\SRmicroFTIR_cellule')
    analysis = SpectralAnalysis(data_dir)
    analysis.run_analysis()

if __name__ == "__main__":
    main() 