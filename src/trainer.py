import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from model import IRSpectraModel
from data_module import get_dataloaders

class IRSpectraTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
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
        
        for batch in tqdm(self.train_loader, desc='Training'):
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
        
        return {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_mse': concentration_mse / num_samples
        }
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        treatment_correct = 0
        concentration_mse = 0
        num_samples = 0
        
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
        
        return {
            'loss': total_loss / num_samples,
            'treatment_acc': treatment_correct / num_samples,
            'concentration_mse': concentration_mse / num_samples
        }
    
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
            
            # Regular checkpoints
            if (epoch + 1) % save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')

def main():
    # Configuration
    DATA_DIR = r'D:\Experiments\IR_cells\data\SRmicroFTIR_cellule'
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    
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
        learning_rate=LEARNING_RATE
    )
    
    # Train model
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_dir='models'
    )

if __name__ == "__main__":
    main() 