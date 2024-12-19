import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from model import IRSpectraModel
from data_module import get_dataloaders
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

class IRSpectraTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wandb_config: dict = None,
                 debug_plots: bool = False):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize wandb if config is provided
        self.use_wandb = wandb_config is not None
        if self.use_wandb:
            wandb.init(
                project=wandb_config.get('project', 'ir-spectra-analysis'),
                name=wandb_config.get('name', None),
                config={
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'model_type': model.__class__.__name__,
                    'batch_size': train_loader.batch_size,
                    'device': device,
                    **wandb_config
                }
            )
            # Watch model gradients
            wandb.watch(model)
        
        # Loss functions
        self.treatment_criterion = nn.CrossEntropyLoss()
        self.concentration_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Disable debug plots
        self.debug_plots = False
        
    def plot_batch_spectra(self, spectra, treatment_targets, concentration_targets, phase='train'):
        """Plot a batch of spectra with their labels in a style matching spectral_analysis.py"""
        if self.debug_plot_count >= self.max_debug_plots:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot up to 5 spectra from the batch with their std deviation
        num_spectra = min(5, len(spectra))
        treatment_names = ['Control', 'AMT', 'NP']
        
        # Generate x-axis values (wavenumbers) - assuming 800-4000 range with 1000 points
        wavenumbers = np.linspace(800, 4000, spectra.size(-1))
        
        for i in range(num_spectra):
            # Squeeze out extra dimensions and convert to numpy array
            spectrum = spectra[i].squeeze().cpu().numpy()
            treatment = treatment_names[treatment_targets[i].item()]
            concentration = concentration_targets[i].item()
            
            # Plot the spectrum
            ax.plot(wavenumbers, spectrum, 
                   label=f'{treatment} (conc: {concentration:.2f})')
            
            # Add shaded area for uncertainty (if you have multiple measurements)
            std_dev = np.abs(spectrum) * 0.1  # 10% of absolute value as example
            ax.fill_between(wavenumbers, 
                          spectrum - std_dev,
                          spectrum + std_dev,
                          alpha=0.2)
        
        # Set proper axes labels and title
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Absorbance')
        ax.set_title(f'Sample Spectra from {phase.capitalize()} Batch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set proper x-axis limits
        ax.set_xlim(800, 4000)
        
        # Save the plot with tight layout
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.debug_dir / f'spectra_batch_{phase}_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.debug_plot_count += 1

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        treatment_correct = 0
        concentration_correct = 0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            spectra, treatment_targets, concentration_targets = [
                x.to(self.device) for x in batch
            ]
            
            # Ensure proper dimensions
            batch_size = spectra.size(0)
            spectra = spectra.squeeze(1)
            treatment_targets = treatment_targets.view(-1)
            concentration_targets = concentration_targets.view(-1)
            
            # Forward pass
            treatment_logits, concentration_logits = self.model(spectra)
            
            # Calculate losses
            treatment_loss = self.treatment_criterion(treatment_logits, treatment_targets)
            concentration_loss = self.concentration_criterion(concentration_logits, concentration_targets)
            
            # Combined loss
            loss = treatment_loss + concentration_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item() * batch_size
            treatment_pred = treatment_logits.argmax(dim=1)
            concentration_pred = concentration_logits.argmax(dim=1)
            
            treatment_correct += (treatment_pred == treatment_targets).sum().item()
            concentration_correct += (concentration_pred == concentration_targets).sum().item()
            num_samples += batch_size
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/treatment_loss': treatment_loss.item(),
                    'batch/concentration_loss': concentration_loss.item(),
                    'batch/treatment_accuracy': (treatment_pred == treatment_targets).float().mean().item(),
                    'batch/concentration_accuracy': (concentration_pred == concentration_targets).float().mean().item(),
                })
        
        metrics = {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_acc': concentration_correct / num_samples
        }
        
        if self.use_wandb:
            wandb.log({f'train/{k}': v for k, v in metrics.items()})
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        treatment_correct = 0
        concentration_correct = 0
        num_samples = 0
        
        all_treatment_preds = []
        all_treatment_targets = []
        all_concentration_preds = []
        all_concentration_targets = []
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            spectra, treatment_targets, concentration_targets = [
                x.to(self.device) for x in batch
            ]
            
            # Add debug plotting for first validation batch
            if self.debug_plots and batch_idx == 0:
                self.plot_batch_spectra(spectra, treatment_targets, 
                                      concentration_targets, phase='val')
            
            # Ensure proper dimensions
            batch_size = spectra.size(0)
            spectra = spectra.squeeze(1)  # Remove extra channel dim if present
            treatment_targets = treatment_targets.view(-1)  # Flatten to 1D
            concentration_targets = concentration_targets.view(-1)  # Flatten to 1D
            
            # Forward pass
            treatment_logits, concentration_logits = self.model(spectra)
            
            # Calculate losses
            treatment_loss = self.treatment_criterion(treatment_logits, treatment_targets)
            concentration_loss = self.concentration_criterion(concentration_logits, concentration_targets)
            
            loss = treatment_loss + concentration_loss
            
            # Calculate metrics
            total_loss += loss.item() * batch_size
            treatment_pred = treatment_logits.argmax(dim=1)
            concentration_pred = concentration_logits.argmax(dim=1)
            
            treatment_correct += (treatment_pred == treatment_targets).sum().item()
            concentration_correct += (concentration_pred == concentration_targets).sum().item()
            num_samples += batch_size
            
            # Collect predictions for confusion matrix
            all_treatment_preds.extend(treatment_pred.cpu().numpy())
            all_treatment_targets.extend(treatment_targets.cpu().numpy())
            all_concentration_preds.extend(concentration_pred.cpu().numpy())
            all_concentration_targets.extend(concentration_targets.cpu().numpy())
        
        metrics = {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_acc': concentration_correct / num_samples
        }
        
        if self.use_wandb:
            # Log validation metrics
            wandb.log({f'val/{k}': v for k, v in metrics.items()})
            
            # Log confusion matrices for both treatment and concentration
            wandb.log({
                'val/treatment_confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_treatment_targets,
                    preds=all_treatment_preds,
                    class_names=['Control', 'AMT', 'NP']
                ),
                'val/concentration_confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_concentration_targets,
                    preds=all_concentration_preds,
                    class_names=['1/2', '1/8', '1/100']
                )
            })
            
            # Log concentration prediction scatter plot
            wandb.log({
                'val/concentration_predictions': wandb.plot.scatter(
                    wandb.Table(data=[[x, y] for x, y in zip(all_concentration_targets, all_concentration_preds)],
                              columns=['True', 'Predicted']),
                    'True',
                    'Predicted',
                    title='Concentration Predictions vs True Values'
                )
            })
        
        return metrics
    
    def train(self, 
              num_epochs: int,
              save_dir: str = 'models',
              save_freq: int = 5):
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            print(f"Train metrics: {train_metrics}")
            
            # Validation
            val_metrics = self.validate()
            print(f"Validation metrics: {val_metrics}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), 
                         save_dir / 'best_model.pth')
                if self.use_wandb:
                    wandb.run.summary['best_val_loss'] = best_val_loss
            

        
        if self.use_wandb:
            wandb.finish()

def main():
    # Configuration
    DATA_DIR = r'data/SRmicroFTIR_cellule'
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.2
    
    # WandB configuration
    wandb_config = {
        'project': 'ir-spectra-analysis',
        'name': 'convnext1d_baseline',
        'tags': ['baseline', 'convnext1d'],
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'num_epochs': NUM_EPOCHS,
        'architecture': 'ConvNeXt1D',
        'notes': 'Baseline training with ConvNeXt1D for IR spectra classification and regression'
    }
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        transform_params={
            'target_size': 1000,
            'window': [800, 4000],
            'normalize': True
        }
    )
    
    # Initialize model with number of concentration classes
    model = IRSpectraModel(num_treatments=3, num_concentrations=3)
    
    # Initialize trainer
    trainer = IRSpectraTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        wandb_config=wandb_config,
        debug_plots=False  # Set to False
    )
    
    # Train model
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir='models'
    )

if __name__ == "__main__":
    main() 