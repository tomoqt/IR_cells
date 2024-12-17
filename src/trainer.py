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

class IRSpectraTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 wandb_config: dict = None):
        
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
        self.concentration_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        treatment_correct = 0
        concentration_mse = 0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            spectra, treatment_targets, concentration_targets = [
                x.to(self.device) for x in batch
            ]
            
            # Forward pass
            treatment_logits, concentration_preds = self.model(spectra)
            
            # Calculate losses
            treatment_loss = self.treatment_criterion(
                treatment_logits, treatment_targets.squeeze()
            )
            concentration_loss = self.concentration_criterion(
                concentration_preds, concentration_targets
            )
            
            # Combined loss
            loss = treatment_loss + concentration_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item() * spectra.size(0)
            treatment_pred = treatment_logits.argmax(dim=1)
            treatment_correct += (treatment_pred == treatment_targets.squeeze()).sum().item()
            concentration_mse += concentration_loss.item() * spectra.size(0)
            num_samples += spectra.size(0)
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    'batch/loss': loss.item(),
                    'batch/treatment_loss': treatment_loss.item(),
                    'batch/concentration_loss': concentration_loss.item(),
                    'batch/treatment_accuracy': (treatment_pred == treatment_targets.squeeze()).float().mean().item(),
                })
        
        metrics = {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_mse': concentration_mse / num_samples
        }
        
        if self.use_wandb:
            wandb.log({f'train/{k}': v for k, v in metrics.items()})
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        treatment_correct = 0
        concentration_mse = 0
        num_samples = 0
        
        all_treatment_preds = []
        all_treatment_targets = []
        all_concentration_preds = []
        all_concentration_targets = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            spectra, treatment_targets, concentration_targets = [
                x.to(self.device) for x in batch
            ]
            
            # Forward pass
            treatment_logits, concentration_preds = self.model(spectra)
            
            # Calculate losses
            treatment_loss = self.treatment_criterion(
                treatment_logits, treatment_targets.squeeze()
            )
            concentration_loss = self.concentration_criterion(
                concentration_preds, concentration_targets
            )
            
            loss = treatment_loss + concentration_loss
            
            # Calculate metrics
            total_loss += loss.item() * spectra.size(0)
            treatment_pred = treatment_logits.argmax(dim=1)
            treatment_correct += (treatment_pred == treatment_targets.squeeze()).sum().item()
            concentration_mse += concentration_loss.item() * spectra.size(0)
            num_samples += spectra.size(0)
            
            # Collect predictions for confusion matrix
            all_treatment_preds.extend(treatment_pred.cpu().numpy())
            all_treatment_targets.extend(treatment_targets.squeeze().cpu().numpy())
            all_concentration_preds.extend(concentration_preds.cpu().numpy())
            all_concentration_targets.extend(concentration_targets.cpu().numpy())
        
        metrics = {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_mse': concentration_mse / num_samples
        }
        
        if self.use_wandb:
            # Log validation metrics
            wandb.log({f'val/{k}': v for k, v in metrics.items()})
            
            # Log confusion matrix
            wandb.log({
                'val/confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_treatment_targets,
                    preds=all_treatment_preds,
                    class_names=['Control', 'AMT', 'NP']
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
            
            # Regular checkpoints
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                
                if self.use_wandb:
                    wandb.save(str(checkpoint_path))
        
        if self.use_wandb:
            wandb.finish()

def main():
    # Configuration
    DATA_DIR = r'data/SRmicroFTIR_cellule'
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    # WandB configuration
    wandb_config = {
        'project': 'ir-spectra-analysis',
        'name': 'convnext1d_baseline',
        'tags': ['baseline', 'convnext1d'],
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
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
    
    # Initialize model
    model = IRSpectraModel(num_treatments=3)
    
    # Initialize trainer
    trainer = IRSpectraTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        wandb_config=wandb_config
    )
    
    # Train model
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir='models'
    )

if __name__ == "__main__":
    main() 