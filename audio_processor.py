"""
Audio processing utilities for the Streamlit app
"""
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import tempfile
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from utils.feature_extractor import FeatureExtractor
from utils.audio_utils import load_audio, save_audio, trim_silence, normalize_loudness
from models.autovc import AutoVC
from models.vocoder import create_vocoder
import config

class AudioProcessor:
    """Main audio processor for the application"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.vocoder = create_vocoder("griffin_lim")
        self.device = config.DEVICE
        
        # Load pre-trained model if available
        self.load_pretrained_model()
    
    def load_pretrained_model(self):
        """Load pre-trained AutoVC model if available"""
        model_path = os.path.join(config.CHECKPOINT_DIR, "final_model.pth")
        if os.path.exists(model_path):
            try:
                self.model = AutoVC().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print("Pre-trained model loaded successfully!")
            except Exception as e:
                print(f"Error loading pre-trained model: {str(e)}")
                self.model = None
        else:
            print("No pre-trained model found. Training required.")
            self.model = AutoVC().to(self.device)
    
    def process_uploaded_audio(self, uploaded_file) -> Tuple[np.ndarray, int, Dict]:
        """
        Process uploaded audio file
        
        Args:
            uploaded_file: Streamlit uploaded file
            
        Returns:
            Tuple of (audio_data, sample_rate, features_dict)
        """
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load audio
            audio, sr = load_audio(tmp_path, sr=config.SAMPLE_RATE)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Preprocess audio
            audio = trim_silence(audio)
            audio = normalize_loudness(audio)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(audio)
            
            return audio, sr, features
            
        except Exception as e:
            raise ValueError(f"Error processing audio file: {str(e)}")
    
    def create_mel_spectrogram_plot(self, mel_spec: np.ndarray) -> plt.Figure:
        """
        Create matplotlib figure for mel-spectrogram
        
        Args:
            mel_spec: Mel-spectrogram array
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Display mel-spectrogram
        img = ax.imshow(mel_spec, aspect='auto', origin='lower', 
                       interpolation='nearest', cmap='viridis')
        
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Mel Frequency Bins')
        ax.set_title('Mel-Spectrogram')
        
        # Add colorbar
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Magnitude (dB)')
        
        plt.tight_layout()
        return fig
    
    def create_f0_plot(self, f0: np.ndarray, voiced_flag: np.ndarray) -> plt.Figure:
        """
        Create matplotlib figure for F0 contour
        
        Args:
            f0: F0 contour
            voiced_flag: Voiced/unvoiced flags
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        time_frames = np.arange(len(f0)) * config.HOP_LENGTH / config.SAMPLE_RATE
        
        # Plot F0 only for voiced frames
        voiced_f0 = np.where(voiced_flag, np.exp(f0), np.nan)
        ax.plot(time_frames, voiced_f0, 'b-', linewidth=2, label='F0')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Fundamental Frequency (F0) Contour')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def convert_voice(self, source_audio: np.ndarray, 
                     target_features: Optional[Dict] = None) -> np.ndarray:
        """
        Perform voice conversion
        
        Args:
            source_audio: Source audio data
            target_features: Target speaker features (optional)
            
        Returns:
            Converted audio
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        try:
            # Extract source features
            source_mel = self.feature_extractor.extract_mel_spectrogram(source_audio)
            source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(self.device)
            
            # Prepare target features
            if target_features is not None and 'mel_spec' in target_features:
                target_mel_tensor = torch.from_numpy(target_features['mel_spec']).float().unsqueeze(0).to(self.device)
            else:
                target_mel_tensor = None
            
            # Perform conversion
            with torch.no_grad():
                outputs = self.model(source_mel_tensor, target_mel_tensor)
                converted_mel = outputs['converted']
            
            # Convert mel-spectrogram back to audio using vocoder
            converted_audio = self.vocoder(converted_mel)
            
            # Convert to numpy
            if isinstance(converted_audio, torch.Tensor):
                converted_audio = converted_audio.squeeze().cpu().numpy()
            
            return converted_audio
            
        except Exception as e:
            raise ValueError(f"Error during voice conversion: {str(e)}")
    
    def save_converted_audio(self, audio: np.ndarray, 
                           filename: str = "converted_audio.wav") -> str:
        """
        Save converted audio to file
        
        Args:
            audio: Audio data
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(config.AUDIO_CACHE_DIR, filename)
        save_audio(audio, output_path, config.SAMPLE_RATE)
        return output_path
    
    def create_waveform_comparison_plot(self, original: np.ndarray, 
                                      converted: np.ndarray) -> plt.Figure:
        """
        Create comparison plot of original and converted waveforms
        
        Args:
            original: Original audio
            converted: Converted audio
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Time axis
        time_orig = np.arange(len(original)) / config.SAMPLE_RATE
        time_conv = np.arange(len(converted)) / config.SAMPLE_RATE
        
        # Original waveform
        ax1.plot(time_orig, original, 'b-', alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original Audio')
        ax1.grid(True, alpha=0.3)
        
        # Converted waveform
        ax2.plot(time_conv, converted, 'r-', alpha=0.7)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Converted Audio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded", "parameters": 0}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "Model loaded",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "model_type": "AutoVC"
        }
