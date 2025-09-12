"""
Simple vocoder implementation for mel-spectrogram to waveform conversion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Optional
import config

class SimpleVocoder(nn.Module):
    """
    Simple vocoder using Griffin-Lim algorithm
    This is a basic implementation - in production, use HiFi-GAN or similar
    """
    
    def __init__(self, 
                 sr: int = config.SAMPLE_RATE,
                 n_fft: int = config.N_FFT,
                 hop_length: int = config.HOP_LENGTH,
                 win_length: int = config.WIN_LENGTH,
                 n_mels: int = config.N_MELS):
        super(SimpleVocoder, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        # Create mel filter banks
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                       fmin=config.F_MIN, fmax=config.F_MAX)
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        # Create inverse mel filter (pseudo-inverse)
        inv_mel_basis = np.linalg.pinv(mel_basis)
        self.register_buffer('inv_mel_basis', torch.from_numpy(inv_mel_basis).float())
    
    def mel_to_linear(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to linear spectrogram
        
        Args:
            mel_spec: Mel-spectrogram (B, n_mels, T) or (n_mels, T)
            
        Returns:
            Linear spectrogram
        """
        if mel_spec.dim() == 2:
            # Add batch dimension
            mel_spec = mel_spec.unsqueeze(0)
        
        # Convert from log to linear scale
        linear_spec = torch.pow(10.0, mel_spec / 10.0)
        
        # Convert mel to linear spectrogram
        linear_spec = torch.matmul(self.inv_mel_basis, linear_spec)
        
        return linear_spec
    
    def griffin_lim(self, linear_spec: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
        """
        Griffin-Lim algorithm for phase reconstruction
        
        Args:
            linear_spec: Linear spectrogram (B, n_freq, T)
            n_iter: Number of iterations
            
        Returns:
            Reconstructed waveform
        """
        # Initialize with random phase
        angles = torch.rand_like(linear_spec) * 2 * np.pi - np.pi
        complex_spec = linear_spec * torch.exp(1j * angles)
        
        for _ in range(n_iter):
            # ISTFT
            waveform = torch.istft(complex_spec, 
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  center=True,
                                  normalized=False,
                                  return_complex=False)
            
            # STFT
            complex_spec = torch.stft(waveform,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    win_length=self.win_length,
                                    center=True,
                                    normalized=False,
                                    return_complex=True)
            
            # Replace magnitude with original
            angles = torch.angle(complex_spec)
            complex_spec = linear_spec * torch.exp(1j * angles)
        
        # Final ISTFT
        waveform = torch.istft(complex_spec,
                              n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              center=True,
                              normalized=False,
                              return_complex=False)
        
        return waveform
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform
        
        Args:
            mel_spec: Mel-spectrogram (B, n_mels, T) or (n_mels, T)
            
        Returns:
            Waveform (B, L) or (L,)
        """
        # Ensure batch dimension
        squeeze_output = False
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
            squeeze_output = True
        
        # Convert to linear spectrogram
        linear_spec = self.mel_to_linear(mel_spec)
        
        # Apply Griffin-Lim
        waveform = self.griffin_lim(linear_spec)
        
        if squeeze_output:
            waveform = waveform.squeeze(0)
        
        return waveform

class NeuralVocoder(nn.Module):
    """
    Simple neural vocoder - lightweight version for demonstration
    In production, use HiFi-GAN or similar state-of-the-art vocoder
    """
    
    def __init__(self, n_mels: int = config.N_MELS, hidden_dim: int = 512):
        super(NeuralVocoder, self).__init__()
        
        self.upsample = nn.ConvTranspose1d(n_mels, hidden_dim, 
                                          kernel_size=16, stride=8, padding=4)
        
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 4, 1, kernel_size=3, padding=1)
        ])
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram to waveform using neural network
        
        Args:
            mel_spec: Mel-spectrogram (B, n_mels, T)
            
        Returns:
            Waveform (B, 1, L)
        """
        x = self.upsample(mel_spec)
        x = self.activation(x)
        
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i < len(self.conv_layers) - 1:
                x = self.activation(x)
            else:
                x = torch.tanh(x)
        
        return x

def create_vocoder(vocoder_type: str = "griffin_lim") -> nn.Module:
    """
    Create vocoder instance
    
    Args:
        vocoder_type: Type of vocoder ("griffin_lim" or "neural")
        
    Returns:
        Vocoder instance
    """
    if vocoder_type == "griffin_lim":
        return SimpleVocoder()
    elif vocoder_type == "neural":
        return NeuralVocoder()
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")
