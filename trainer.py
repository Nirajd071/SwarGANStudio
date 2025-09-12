"""
Training utilities for SwarGAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import config
from models.autovc import AutoVC, AutoVCLoss
from utils.feature_extractor import FeatureExtractor
from utils.audio_utils import load_audio

class VoiceDataset(Dataset):
    """Dataset for voice conversion"""
    
    def __init__(self, audio_files: List[str], feature_extractor: FeatureExtractor):
        self.audio_files = audio_files
        self.feature_extractor = feature_extractor
        self.features_cache = {}
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        
        # Check cache first
        if audio_file in self.features_cache:
            return self.features_cache[audio_file]
        
        # Load and process audio
        try:
            audio, _ = load_audio(audio_file)
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
            
            # Convert to tensor
            mel_tensor = torch.from_numpy(mel_spec).float()
            
            # Cache the result
            self.features_cache[audio_file] = {
                'mel_spec': mel_tensor,
                'speaker_id': idx % 2,  # Simple speaker assignment for demo
                'file_path': audio_file
            }
            
            return self.features_cache[audio_file]
            
        except Exception as e:
            # Return dummy data if loading fails
            dummy_mel = torch.zeros(config.N_MELS, 100)
            return {
                'mel_spec': dummy_mel,
                'speaker_id': 0,
                'file_path': audio_file
            }

class Trainer:
    """Trainer for AutoVC model"""
    
    def __init__(self, model: AutoVC, device: torch.device = config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = AutoVCLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_content_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Get batch data
                mel_specs = batch['mel_spec'].to(self.device)
                
                # For demonstration, use the same mel-spec as source and target
                # In real training, you'd have pairs or use random sampling
                outputs = self.model(mel_specs, mel_specs)
                
                # Calculate loss
                loss_dict = self.criterion(outputs, mel_specs)
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_rec_loss += loss_dict['rec_loss'].item()
                total_content_loss += loss_dict['content_loss'].item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if num_batches == 0:
            return {'total_loss': 0.0, 'rec_loss': 0.0, 'content_loss': 0.0}
        
        return {
            'total_loss': total_loss / num_batches,
            'rec_loss': total_rec_loss / num_batches,
            'content_loss': total_content_loss / num_batches
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_content_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Get batch data
                    mel_specs = batch['mel_spec'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(mel_specs, mel_specs)
                    
                    # Calculate loss
                    loss_dict = self.criterion(outputs, mel_specs)
                    
                    # Accumulate losses
                    total_loss += loss_dict['total_loss'].item()
                    total_rec_loss += loss_dict['rec_loss'].item()
                    total_content_loss += loss_dict['content_loss'].item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        if num_batches == 0:
            return {'total_loss': 0.0, 'rec_loss': 0.0, 'content_loss': 0.0}
        
        return {
            'total_loss': total_loss / num_batches,
            'rec_loss': total_rec_loss / num_batches,
            'content_loss': total_content_loss / num_batches
        }
    
    def train(self, train_dataloader: DataLoader, 
              val_dataloader: Optional[DataLoader] = None,
              num_epochs: int = config.NUM_EPOCHS,
              save_dir: str = config.CHECKPOINT_DIR) -> Dict[str, List[float]]:
        """
        Full training loop
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            self.train_losses.append(train_metrics['total_loss'])
            history['train_loss'].append(train_metrics['total_loss'])
            
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Rec Loss: {train_metrics['rec_loss']:.4f}, "
                  f"Content Loss: {train_metrics['content_loss']:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate_epoch(val_dataloader)
                self.val_losses.append(val_metrics['total_loss'])
                history['val_loss'].append(val_metrics['total_loss'])
                
                print(f"Val Loss: {val_metrics['total_loss']:.4f}, "
                      f"Val Rec Loss: {val_metrics['rec_loss']:.4f}, "
                      f"Val Content Loss: {val_metrics['content_loss']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
            
            print("-" * 50)
        
        # Save final model
        self.save_checkpoint(os.path.join(save_dir, "final_model.pth"))
        
        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            print(f"Checkpoint loaded from {path}")
            return True
        else:
            print(f"No checkpoint found at {path}")
            return False

def create_trainer(model_type: str = "autovc") -> Trainer:
    """
    Create trainer instance
    
    Args:
        model_type: Type of model to train
        
    Returns:
        Trainer instance
    """
    if model_type == "autovc":
        model = AutoVC()
        return Trainer(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
