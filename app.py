"""
SwarGAN: Singing Voice Style Transfer Application
Interactive Streamlit application for voice conversion using deep learning
"""
import streamlit as st
import numpy as np
import torch
import os
import tempfile
from typing import Optional, Dict
import matplotlib.pyplot as plt

# Local imports
from audio_processor import AudioProcessor
from trainer import Trainer, VoiceDataset, create_trainer
from utils.feature_extractor import FeatureExtractor
from utils.audio_utils import load_audio
import config

# Page configuration
st.set_page_config(
    page_title="SwarGAN - Singing Voice Style Transfer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'source_audio' not in st.session_state:
    st.session_state.source_audio = None
if 'target_audio' not in st.session_state:
    st.session_state.target_audio = None
if 'source_features' not in st.session_state:
    st.session_state.source_features = None
if 'target_features' not in st.session_state:
    st.session_state.target_features = None
if 'converted_audio' not in st.session_state:
    st.session_state.converted_audio = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# Main title and description
st.title("🎵 SwarGAN - Singing Voice Style Transfer")
st.markdown("""
### Transform singing voices while preserving melody and lyrics

Upload audio files to perform style transfer between different singers using deep learning.
The system extracts content (melody/lyrics) from the source and applies the timbre of the target singer.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Audio Processing", "Model Training", "Voice Conversion", "Analysis & Visualization", "Model Information"]
)

# Audio Processing Page
if page == "Audio Processing":
    st.header("🎤 Audio Upload & Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Audio (Singer to Convert)")
        source_file = st.file_uploader(
            "Upload source singing audio",
            type=['wav', 'mp3', 'flac', 'ogg'],
            key="source_upload"
        )
        
        if source_file is not None:
            try:
                with st.spinner("Processing source audio..."):
                    audio, sr, features = st.session_state.audio_processor.process_uploaded_audio(source_file)
                    st.session_state.source_audio = (audio, sr)
                    st.session_state.source_features = features
                
                st.success("Source audio processed successfully!")
                st.audio(source_file, format='audio/wav')
                
                # Display basic info
                st.write(f"**Duration:** {len(audio) / sr:.2f} seconds")
                st.write(f"**Sample Rate:** {sr} Hz")
                st.write(f"**Audio Shape:** {audio.shape}")
                
            except Exception as e:
                st.error(f"Error processing source audio: {str(e)}")
    
    with col2:
        st.subheader("Target Audio (Style Reference)")
        target_file = st.file_uploader(
            "Upload target singer audio (for style)",
            type=['wav', 'mp3', 'flac', 'ogg'],
            key="target_upload"
        )
        
        if target_file is not None:
            try:
                with st.spinner("Processing target audio..."):
                    audio, sr, features = st.session_state.audio_processor.process_uploaded_audio(target_file)
                    st.session_state.target_audio = (audio, sr)
                    st.session_state.target_features = features
                
                st.success("Target audio processed successfully!")
                st.audio(target_file, format='audio/wav')
                
                # Display basic info
                st.write(f"**Duration:** {len(audio) / sr:.2f} seconds")
                st.write(f"**Sample Rate:** {sr} Hz")
                st.write(f"**Audio Shape:** {audio.shape}")
                
            except Exception as e:
                st.error(f"Error processing target audio: {str(e)}")

# Model Training Page
elif page == "Model Training":
    st.header("🏋️ Model Training")
    
    st.info("""
    **Training Requirements:**
    - Upload multiple audio files for training
    - Training can take significant time depending on dataset size
    - GPU is recommended for faster training
    """)
    
    # Training configuration
    st.subheader("Training Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
    
    with col2:
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0001, format="%.4f")
        save_interval = st.number_input("Save Interval", min_value=1, max_value=50, value=5)
    
    with col3:
        device_option = st.selectbox("Device", ["auto", "cpu", "cuda"])
        model_type = st.selectbox("Model Type", ["autovc"])
    
    # Training data upload
    st.subheader("Training Data")
    training_files = st.file_uploader(
        "Upload training audio files",
        type=['wav', 'mp3', 'flac', 'ogg'],
        accept_multiple_files=True,
        key="training_upload"
    )
    
    if training_files:
        st.write(f"**Uploaded {len(training_files)} training files**")
        
        # Show file list
        with st.expander("View uploaded files"):
            for i, file in enumerate(training_files):
                st.write(f"{i+1}. {file.name}")
    
    # Training control
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Training", type="primary"):
            if not training_files:
                st.error("Please upload training files first!")
            else:
                try:
                    # Save training files temporarily
                    temp_files = []
                    for file in training_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                            tmp.write(file.getvalue())
                            temp_files.append(tmp.name)
                    
                    # Create trainer
                    trainer = create_trainer(model_type)
                    
                    # Create dataset
                    feature_extractor = FeatureExtractor()
                    dataset = VoiceDataset(temp_files, feature_extractor)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start training
                    with st.spinner("Training in progress..."):
                        history = trainer.train(
                            train_dataloader=dataloader,
                            num_epochs=num_epochs
                        )
                        st.session_state.training_history = history
                    
                    st.success("Training completed!")
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    
                    # Update audio processor with new model
                    st.session_state.audio_processor.model = trainer.model
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with col2:
        if st.button("Load Pretrained Model"):
            try:
                st.session_state.audio_processor.load_pretrained_model()
                st.success("Pretrained model loaded!")
            except Exception as e:
                st.error(f"Failed to load pretrained model: {str(e)}")
    
    with col3:
        if st.button("Reset Model"):
            st.session_state.audio_processor.model = None
            st.success("Model reset!")
    
    # Training history visualization
    if st.session_state.training_history is not None:
        st.subheader("Training History")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        history = st.session_state.training_history
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Voice Conversion Page
elif page == "Voice Conversion":
    st.header("🎭 Voice Conversion")
    
    if st.session_state.source_audio is None:
        st.warning("Please upload source audio in the Audio Processing page first.")
    elif st.session_state.audio_processor.model is None:
        st.warning("Please train or load a model in the Model Training page first.")
    else:
        st.info("Ready for voice conversion! Configure settings below and click Convert.")
        
        # Conversion settings
        st.subheader("Conversion Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            use_target = st.checkbox("Use target audio for style", value=True)
            if use_target and st.session_state.target_audio is None:
                st.warning("No target audio uploaded. Will use source audio style.")
                use_target = False
        
        with col2:
            vocoder_type = st.selectbox("Vocoder Type", ["griffin_lim", "neural"])
        
        # Conversion button
        if st.button("🎵 Convert Voice", type="primary"):
            try:
                with st.spinner("Converting voice... This may take a moment."):
                    source_audio, _ = st.session_state.source_audio
                    target_features = st.session_state.target_features if use_target else None
                    
                    converted_audio = st.session_state.audio_processor.convert_voice(
                        source_audio, target_features
                    )
                    
                    st.session_state.converted_audio = converted_audio
                
                st.success("Voice conversion completed!")
                
                # Audio playback
                st.subheader("Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Original Audio**")
                    # Save original audio temporarily for playback
                    temp_orig = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    st.session_state.audio_processor.save_converted_audio(
                        source_audio, temp_orig.name
                    )
                    st.audio(temp_orig.name)
                
                with col2:
                    if use_target and st.session_state.target_audio is not None:
                        st.write("**Target Style Reference**")
                        target_audio, _ = st.session_state.target_audio
                        temp_target = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        st.session_state.audio_processor.save_converted_audio(
                            target_audio, temp_target.name
                        )
                        st.audio(temp_target.name)
                
                with col3:
                    st.write("**Converted Audio**")
                    # Save converted audio
                    converted_path = st.session_state.audio_processor.save_converted_audio(
                        converted_audio, "converted_output.wav"
                    )
                    st.audio(converted_path)
                    
                    # Download button
                    with open(converted_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Converted Audio",
                            data=f.read(),
                            file_name="converted_voice.wav",
                            mime="audio/wav"
                        )
                
                # Waveform comparison
                st.subheader("Waveform Comparison")
                fig = st.session_state.audio_processor.create_waveform_comparison_plot(
                    source_audio, converted_audio
                )
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Conversion failed: {str(e)}")

# Analysis & Visualization Page
elif page == "Analysis & Visualization":
    st.header("📊 Analysis & Visualization")
    
    # Source audio analysis
    if st.session_state.source_features is not None:
        st.subheader("Source Audio Analysis")
        
        features = st.session_state.source_features
        
        # Mel-spectrogram visualization
        st.write("**Mel-Spectrogram**")
        mel_fig = st.session_state.audio_processor.create_mel_spectrogram_plot(
            features['mel_spec']
        )
        st.pyplot(mel_fig)
        
        # F0 visualization
        st.write("**Fundamental Frequency (F0) Contour**")
        f0_fig = st.session_state.audio_processor.create_f0_plot(
            features['f0'], features['voiced_flag']
        )
        st.pyplot(f0_fig)
        
        # Feature statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mel-Spec Shape", f"{features['mel_spec'].shape}")
            st.metric("F0 Range (Hz)", 
                     f"{np.exp(features['f0'][features['voiced_flag']]).min():.1f} - {np.exp(features['f0'][features['voiced_flag']]).max():.1f}")
        
        with col2:
            voiced_ratio = np.mean(features['voiced_flag'])
            st.metric("Voiced Ratio", f"{voiced_ratio:.2f}")
            st.metric("Duration (frames)", len(features['f0']))
        
        with col3:
            mean_f0 = np.mean(np.exp(features['f0'][features['voiced_flag']]))
            st.metric("Mean F0 (Hz)", f"{mean_f0:.1f}")
            spectral_centroid = np.mean(features['mel_spec'])
            st.metric("Spectral Centroid", f"{spectral_centroid:.2f}")
    
    else:
        st.info("Upload and process source audio to see analysis.")
    
    # Target audio analysis
    if st.session_state.target_features is not None:
        st.subheader("Target Audio Analysis")
        
        features = st.session_state.target_features
        
        # Mel-spectrogram visualization
        st.write("**Target Mel-Spectrogram**")
        mel_fig = st.session_state.audio_processor.create_mel_spectrogram_plot(
            features['mel_spec']
        )
        st.pyplot(mel_fig)
        
        # F0 visualization
        st.write("**Target F0 Contour**")
        f0_fig = st.session_state.audio_processor.create_f0_plot(
            features['f0'], features['voiced_flag']
        )
        st.pyplot(f0_fig)

# Model Information Page
elif page == "Model Information":
    st.header("🤖 Model Information")
    
    # Get model info
    model_info = st.session_state.audio_processor.get_model_info()
    
    # Display model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Status")
        st.info(model_info["status"])
        
        if "total_parameters" in model_info:
            st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
            st.metric("Trainable Parameters", f"{model_info['trainable_parameters']:,}")
        
        if "device" in model_info:
            st.metric("Device", model_info["device"])
        
        if "model_type" in model_info:
            st.metric("Model Type", model_info["model_type"])
    
    with col2:
        st.subheader("System Information")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**CUDA Available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**CUDA Version:** {torch.version.cuda}")
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.write(f"**Audio Sample Rate:** {config.SAMPLE_RATE} Hz")
        st.write(f"**Mel Bins:** {config.N_MELS}")
        st.write(f"**Hop Length:** {config.HOP_LENGTH}")
    
    # Architecture diagram (text-based)
    st.subheader("AutoVC Architecture")
    st.markdown("""
    ```
    Source Mel-Spec ──► Content Encoder ──► Bottleneck Features
                                                    │
                                                    ▼
    Target Mel-Spec ──► Speaker Encoder ──► Style Embedding
                                                    │
                                                    ▼
                                            Concatenation
                                                    │
                                                    ▼
                                               Decoder
                                                    │
                                                    ▼
                                          Converted Mel-Spec ──► Vocoder ──► Audio Output
    ```
    
    **Key Components:**
    - **Content Encoder**: Extracts linguistic/musical content while removing speaker characteristics
    - **Speaker Encoder**: Extracts speaker-specific timbre information
    - **Decoder**: Reconstructs mel-spectrogram with new speaker characteristics
    - **Vocoder**: Converts mel-spectrogram back to audio waveform
    """)
    
    # Configuration
    st.subheader("Current Configuration")
    config_dict = {
        "Sample Rate": config.SAMPLE_RATE,
        "Mel Bins": config.N_MELS,
        "Hop Length": config.HOP_LENGTH,
        "Win Length": config.WIN_LENGTH,
        "N-FFT": config.N_FFT,
        "Content Dim": config.CONTENT_DIM,
        "Style Dim": config.STYLE_DIM,
        "Bottleneck Dim": config.BOTTLENECK_DIM,
        "Hidden Dim": config.HIDDEN_DIM,
        "Batch Size": config.BATCH_SIZE,
        "Learning Rate": config.LEARNING_RATE
    }
    
    for key, value in config_dict.items():
        st.write(f"**{key}:** {value}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p>🎵 SwarGAN - Singing Voice Style Transfer using Deep Learning</p>
<p>Built with Streamlit, PyTorch, and librosa</p>
</div>
""", unsafe_allow_html=True)
