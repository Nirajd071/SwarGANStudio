# SwarGAN - Singing Voice Style Transfer

## Overview

SwarGAN is an interactive deep learning application for singing voice style transfer built with Streamlit. The system transforms singing voices by transferring the timbre and style characteristics of a target singer to a source singer's voice while preserving the original melody and lyrics. The application uses an AutoVC (Autoencoder for Voice Conversion) architecture with content-style disentanglement to perform non-parallel voice conversion, making it practical for real-world singing voice transformation scenarios.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Interface**: Interactive web application providing file upload capabilities, real-time audio processing visualization, and playback controls
- **Session State Management**: Maintains audio data, extracted features, and conversion results across user interactions
- **Real-time Visualization**: Matplotlib integration for displaying mel-spectrograms and audio waveforms

### Backend Architecture
- **AudioProcessor Class**: Central processing engine that orchestrates audio loading, feature extraction, model inference, and vocoding
- **AutoVC Model**: Deep learning architecture implementing content-style disentanglement through:
  - Content Encoder: Extracts linguistic and melodic information using 1D convolutions and bottleneck layers
  - Style Encoder: Captures speaker-specific timbre characteristics
  - Decoder: Reconstructs mel-spectrograms with transferred style
- **Feature Extraction Pipeline**: Librosa-based mel-spectrogram extraction with configurable parameters (80 mel bins, 22.05kHz sampling)
- **Griffin-Lim Vocoder**: Converts mel-spectrograms back to audio waveforms using phase reconstruction

### Data Processing Architecture
- **Multi-format Audio Support**: Handles various audio formats through librosa backend
- **Feature Caching**: Session-based caching of extracted features to avoid recomputation
- **Audio Preprocessing**: Automatic normalization, silence trimming, and loudness adjustment
- **PyWorld Integration**: Advanced pitch and spectral feature extraction for singing voice analysis

### Training Infrastructure
- **VoiceDataset Class**: Custom PyTorch dataset with feature caching for efficient training
- **Trainer Module**: Comprehensive training pipeline with checkpoint management, loss tracking, and model persistence
- **Loss Functions**: Multi-component loss including L1 mel loss, cycle consistency, and identity preservation
- **GPU Acceleration**: CUDA support with automatic device detection and optimization

### Model Architecture Design
- **Content-Style Disentanglement**: Separates what is said (content) from who says it (style) using bottleneck architecture
- **Non-parallel Training**: Works with unpaired audio data, eliminating need for parallel recordings
- **Configurable Dimensions**: Adjustable content (256), style (8), and bottleneck (32) dimensions
- **Modular Vocoder System**: Pluggable vocoder architecture supporting Griffin-Lim with extensibility for advanced vocoders

## External Dependencies

### Core ML/Audio Libraries
- **PyTorch**: Deep learning framework for model implementation and training
- **Librosa**: Primary audio processing and feature extraction library
- **SoundFile**: High-quality audio I/O operations
- **NumPy**: Numerical computations and array operations
- **PyWorld**: Advanced pitch analysis and spectral processing for singing voice

### Web Framework
- **Streamlit**: Interactive web application framework providing the user interface
- **Matplotlib**: Visualization of audio features and spectrograms

### System Dependencies
- **CUDA/GPU Support**: Optional GPU acceleration for model training and inference
- **File System**: Local storage for model checkpoints, audio cache, and temporary files
- **Audio Codecs**: Support for various audio formats through librosa's backend dependencies