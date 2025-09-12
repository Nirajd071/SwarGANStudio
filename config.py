"""
Configuration file for SwarGAN Singing Voice Style Transfer
"""
import os

# Audio Configuration
SAMPLE_RATE = 22050
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024
N_MELS = 80
F_MIN = 80
F_MAX = 8000

# Model Configuration
CONTENT_DIM = 256
STYLE_DIM = 8
BOTTLENECK_DIM = 32
HIDDEN_DIM = 512

# Training Configuration
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
SAVE_INTERVAL = 10

# Paths
MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"
AUDIO_CACHE_DIR = "audio_cache"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
