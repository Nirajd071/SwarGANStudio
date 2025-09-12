"""
Vocal separation utilities using Demucs for singing voice extraction
"""
import os
import tempfile
import shutil
import subprocess
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torchaudio
import yt_dlp
from pathlib import Path
import config

class VocalSeparator:
    """
    Vocal separator using Demucs for high-quality voice extraction
    """
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        Initialize the vocal separator
        
        Args:
            model_name: Demucs model to use ('htdemucs', 'htdemucs_ft', 'htdemucs_6s')
        """
        self.model_name = model_name
        self.device = config.DEVICE
        self.temp_dir = tempfile.mkdtemp(prefix="vocal_sep_")
        
    def download_audio_from_url(self, url: str, output_path: str) -> bool:
        """
        Download audio from various platforms using yt-dlp
        
        Args:
            url: URL to download from (YouTube, etc.)
            output_path: Path to save the downloaded audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False, {'error': 'ffmpeg is required but not found. Please install ffmpeg.'}
            
            # Configure yt-dlp options for audio extraction
            ydl_opts = {
                'format': 'bestaudio[ext=mp3]/bestaudio[ext=m4a]/bestaudio',
                'outtmpl': output_path + '.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'extractaudio': True,
                'audioformat': 'mp3',
                'noplaylist': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download and extract audio
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # Find the actual downloaded file
                possible_extensions = ['.mp3', '.m4a', '.webm', '.mp4']
                downloaded_file = None
                
                for ext in possible_extensions:
                    potential_file = output_path + ext
                    if os.path.exists(potential_file):
                        downloaded_file = potential_file
                        break
                
                if downloaded_file and os.path.exists(downloaded_file):
                    # Rename to standardized format
                    final_path = output_path + '.mp3'
                    if downloaded_file != final_path:
                        shutil.move(downloaded_file, final_path)
                    
                    return True, {'title': title, 'duration': duration, 'file': final_path}
                else:
                    return False, {'error': 'Downloaded file not found'}
                    
        except Exception as e:
            return False, {'error': f"Download failed: {str(e)}"}
    
    def separate_vocals_demucs(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """
        Separate vocals using Demucs via command line
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated stems
            
        Returns:
            Dictionary with paths to separated stems
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Run Demucs separation
            cmd = [
                'python', '-m', 'demucs.separate',
                '--name', self.model_name,
                '--out', output_dir,
                input_path
            ]
            
            # Execute Demucs with longer timeout for model downloads and processing
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise Exception(f"Demucs failed: {result.stderr}")
            
            # Find the separated files
            # Demucs creates: output_dir/model_name/filename/vocals.wav, drums.wav, bass.wav, other.wav
            input_filename = Path(input_path).stem
            separated_dir = os.path.join(output_dir, self.model_name, input_filename)
            
            # Dynamically find all available stems (supports 4-stem and 6-stem models)
            stems = {}
            if os.path.exists(separated_dir):
                for stem_file in os.listdir(separated_dir):
                    if stem_file.endswith('.wav'):
                        stem_path = os.path.join(separated_dir, stem_file)
                        stem_name = stem_file.replace('.wav', '')
                        stems[stem_name] = stem_path
            
            return {'success': True, 'stems': stems}
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Separation timed out (>10 minutes). Try with a shorter audio file.'}
        except Exception as e:
            return {'success': False, 'error': f"Separation failed: {str(e)}"}
    
    def process_url(self, url: str) -> Dict:
        """
        Complete pipeline: download from URL and separate vocals
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary with results and file paths
        """
        try:
            # Create temporary paths
            download_path = os.path.join(self.temp_dir, "downloaded_audio")
            output_dir = os.path.join(self.temp_dir, "separated")
            
            # Download audio
            success, download_result = self.download_audio_from_url(url, download_path)
            
            if not success:
                return {
                    'success': False, 
                    'error': download_result.get('error', 'Download failed'),
                    'step': 'download'
                }
            
            downloaded_file = download_result['file']
            
            # Separate vocals
            separation_result = self.separate_vocals_demucs(downloaded_file, output_dir)
            
            if not separation_result.get('success', False):
                return {
                    'success': False,
                    'error': separation_result.get('error', 'Separation failed'),
                    'step': 'separation'
                }
            
            return {
                'success': True,
                'title': download_result.get('title', 'Unknown'),
                'duration': download_result.get('duration', 0),
                'original_file': downloaded_file,
                'stems': separation_result['stems']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}",
                'step': 'general'
            }
    
    def process_uploaded_file(self, file_path: str) -> Dict:
        """
        Process an uploaded file for vocal separation
        
        Args:
            file_path: Path to uploaded audio file
            
        Returns:
            Dictionary with separation results
        """
        try:
            output_dir = os.path.join(self.temp_dir, "separated")
            
            # Separate vocals
            separation_result = self.separate_vocals_demucs(file_path, output_dir)
            
            if not separation_result.get('success', False):
                return {
                    'success': False,
                    'error': separation_result.get('error', 'Separation failed')
                }
            
            return {
                'success': True,
                'original_file': file_path,
                'stems': separation_result['stems']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def get_audio_info(self, file_path: str) -> Dict:
        """
        Get information about an audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            audio, sample_rate = torchaudio.load(file_path)
            duration = audio.shape[1] / sample_rate
            channels = audio.shape[0]
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'shape': audio.shape
            }
        except Exception as e:
            return {'error': f"Could not read audio info: {str(e)}"}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup()

def is_supported_url(url: str) -> bool:
    """
    Check if URL is from a supported platform
    
    Args:
        url: URL to check
        
    Returns:
        True if supported, False otherwise
    """
    supported_domains = [
        'youtube.com', 'youtu.be', 'soundcloud.com', 'bandcamp.com',
        'vimeo.com', 'dailymotion.com', 'twitch.tv'
    ]
    
    return any(domain in url.lower() for domain in supported_domains)

def get_platform_name(url: str) -> str:
    """
    Get the platform name from URL
    
    Args:
        url: URL to analyze
        
    Returns:
        Platform name
    """
    if 'youtube.com' in url or 'youtu.be' in url:
        return 'YouTube'
    elif 'soundcloud.com' in url:
        return 'SoundCloud'
    elif 'bandcamp.com' in url:
        return 'Bandcamp'
    elif 'vimeo.com' in url:
        return 'Vimeo'
    elif 'dailymotion.com' in url:
        return 'Dailymotion'
    elif 'twitch.tv' in url:
        return 'Twitch'
    else:
        return 'Unknown Platform'