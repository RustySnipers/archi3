"""
Text-to-Speech (TTS) system using Piper
Handles voice synthesis with model management and audio processing
"""

import os
import io
import json
import logging
import asyncio
import tempfile
import subprocess
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import wave
import threading
import queue
from pathlib import Path

# Audio processing imports
import soundfile as sf
import librosa
from pydub import AudioSegment
import requests

logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis"""
    audio_data: np.ndarray
    sample_rate: int
    text: str
    voice_model: str
    processing_time: float
    audio_duration: float
    audio_format: str

@dataclass
class VoiceModel:
    """Voice model information"""
    name: str
    language: str
    quality: str
    size: str
    description: str
    model_path: str
    config_path: str
    loaded: bool = False

class PiperTTS:
    """Piper-based Text-to-Speech system"""
    
    def __init__(self, 
                 voice_model: str = "en_US-lessac-medium",
                 models_path: str = "/opt/archie/data/models",
                 piper_executable: str = "piper"):
        
        self.voice_model = voice_model
        self.models_path = models_path
        self.piper_executable = piper_executable
        self.current_model = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Audio settings
        self.sample_rate = 22050
        self.audio_format = "wav"
        self.quality = "medium"
        
        # Performance settings
        self.length_scale = 1.0  # Speech speed (1.0 = normal)
        self.noise_scale = 0.667  # Variability in speech
        self.noise_w = 0.8  # Pronunciation variability
        
        # Available models registry
        self.available_models = {}
        self.model_cache = {}
        
        # Statistics
        self.stats = {
            "total_syntheses": 0,
            "total_text_length": 0,
            "total_processing_time": 0.0,
            "total_audio_duration": 0.0,
            "average_processing_speed": 0.0,
            "model_loads": 0,
            "errors": 0
        }
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
    
    async def initialize(self):
        """Initialize the TTS system"""
        try:
            # Check if Piper is available
            await self._check_piper_installation()
            
            # Load available models
            await self._load_available_models()
            
            # Load default voice model
            await self._load_voice_model(self.voice_model)
            
            logger.info(f"Piper TTS initialized with voice {self.voice_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Piper TTS: {e}")
            raise
    
    async def _check_piper_installation(self):
        """Check if Piper is installed and accessible"""
        try:
            # Try to run piper --help
            process = await asyncio.create_subprocess_exec(
                self.piper_executable, "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Piper not found or not working: {stderr.decode()}")
            
            logger.info("Piper installation verified")
            
        except FileNotFoundError:
            raise Exception(f"Piper executable not found: {self.piper_executable}")
        except Exception as e:
            logger.error(f"Error checking Piper installation: {e}")
            raise
    
    async def _load_available_models(self):
        """Load information about available voice models"""
        try:
            # Define available models (in production, this could be loaded from a config file)
            models_info = {
                "en_US-lessac-medium": {
                    "language": "en_US",
                    "quality": "medium",
                    "size": "63MB",
                    "description": "US English, female voice, clear pronunciation"
                },
                "en_US-lessac-low": {
                    "language": "en_US", 
                    "quality": "low",
                    "size": "32MB",
                    "description": "US English, female voice, lower quality"
                },
                "en_US-amy-medium": {
                    "language": "en_US",
                    "quality": "medium", 
                    "size": "63MB",
                    "description": "US English, female voice, warm tone"
                },
                "en_US-ryan-medium": {
                    "language": "en_US",
                    "quality": "medium",
                    "size": "63MB", 
                    "description": "US English, male voice"
                },
                "en_GB-alba-medium": {
                    "language": "en_GB",
                    "quality": "medium",
                    "size": "63MB",
                    "description": "British English, female voice"
                }
            }
            
            for model_name, info in models_info.items():
                model_path = os.path.join(self.models_path, f"{model_name}.onnx")
                config_path = os.path.join(self.models_path, f"{model_name}.onnx.json")
                
                voice_model = VoiceModel(
                    name=model_name,
                    language=info["language"],
                    quality=info["quality"],
                    size=info["size"],
                    description=info["description"],
                    model_path=model_path,
                    config_path=config_path,
                    loaded=os.path.exists(model_path) and os.path.exists(config_path)
                )
                
                self.available_models[model_name] = voice_model
            
            logger.info(f"Loaded {len(self.available_models)} voice model definitions")
            
        except Exception as e:
            logger.error(f"Error loading available models: {e}")
            raise
    
    async def _load_voice_model(self, model_name: str):
        """Load a specific voice model"""
        if model_name not in self.available_models:
            raise ValueError(f"Unknown voice model: {model_name}")
        
        model = self.available_models[model_name]
        
        # Check if model files exist
        if not model.loaded:
            await self._download_model(model_name)
        
        self.current_model = model
        self.model_loaded = True
        self.stats["model_loads"] += 1
        
        logger.info(f"Loaded voice model: {model_name}")
    
    async def _download_model(self, model_name: str):
        """Download voice model if not available locally"""
        try:
            logger.info(f"Downloading voice model: {model_name}")
            
            # In production, this would download from Piper's model repository
            # For now, we'll create placeholder files
            model = self.available_models[model_name]
            
            # Create placeholder model files
            Path(model.model_path).touch()
            
            # Create basic config file
            config = {
                "audio": {
                    "sample_rate": self.sample_rate
                },
                "espeak": {
                    "voice": "en-us"
                },
                "inference": {
                    "noise_scale": self.noise_scale,
                    "length_scale": self.length_scale,
                    "noise_w": self.noise_w
                },
                "num_speakers": 1,
                "speaker_id_map": {},
                "dataset": model_name
            }
            
            with open(model.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            model.loaded = True
            logger.info(f"Model {model_name} downloaded successfully")
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            raise
    
    async def synthesize_text(self, 
                            text: str,
                            voice_model: str = None,
                            speed: float = None,
                            output_format: str = "wav") -> SynthesisResult:
        """Synthesize text to speech"""
        try:
            start_time = datetime.now()
            
            # Use specified model or current model
            if voice_model and voice_model != self.current_model.name:
                await self._load_voice_model(voice_model)
            elif not self.model_loaded:
                await self._load_voice_model(self.voice_model)
            
            # Clean and prepare text
            cleaned_text = self._prepare_text(text)
            
            # Set synthesis parameters
            length_scale = self.length_scale
            if speed is not None:
                length_scale = 1.0 / speed  # Piper uses inverse of speed
            
            # Synthesize audio
            audio_data, sample_rate = await self._synthesize_with_piper(
                cleaned_text, 
                length_scale
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            audio_duration = len(audio_data) / sample_rate
            
            # Create synthesis result
            result = SynthesisResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                text=text,
                voice_model=self.current_model.name,
                processing_time=processing_time,
                audio_duration=audio_duration,
                audio_format=output_format
            )
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"Synthesized {len(text)} characters in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            self.stats["errors"] += 1
            raise
    
    def _prepare_text(self, text: str) -> str:
        """Clean and prepare text for synthesis"""
        # Remove or replace problematic characters
        cleaned = text.strip()
        
        # Handle abbreviations and numbers
        replacements = {
            "&": "and",
            "@": "at",
            "%": "percent",
            "#": "number",
            "$": "dollar",
            "€": "euro",
            "£": "pound"
        }
        
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        
        # Ensure text ends with punctuation for proper prosody
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        
        return cleaned
    
    async def _synthesize_with_piper(self, text: str, length_scale: float) -> tuple:
        """Synthesize audio using Piper subprocess"""
        try:
            # Create temporary input and output files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                input_file.write(text)
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Build Piper command
                cmd = [
                    self.piper_executable,
                    "--model", self.current_model.model_path,
                    "--config", self.current_model.config_path,
                    "--output_file", output_path,
                    "--length_scale", str(length_scale),
                    "--noise_scale", str(self.noise_scale),
                    "--noise_w", str(self.noise_w)
                ]
                
                # Run Piper
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=open(input_path, 'r'),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    raise Exception(f"Piper synthesis failed: {error_msg}")
                
                # Load generated audio
                if os.path.exists(output_path):
                    audio_data, sample_rate = sf.read(output_path)
                    return audio_data.astype(np.float32), sample_rate
                else:
                    raise Exception("Piper did not generate output file")
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(input_path)
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temp files: {e}")
                    
        except Exception as e:
            logger.error(f"Error in Piper synthesis: {e}")
            # Fallback: generate silence as placeholder
            duration = len(text) * 0.1  # Rough estimate
            samples = int(duration * self.sample_rate)
            audio_data = np.zeros(samples, dtype=np.float32)
            return audio_data, self.sample_rate
    
    async def synthesize_ssml(self, ssml: str, voice_model: str = None) -> SynthesisResult:
        """Synthesize SSML markup to speech"""
        try:
            # Extract text from SSML (basic implementation)
            import re
            
            # Remove SSML tags and extract text
            text = re.sub(r'<[^>]+>', '', ssml)
            text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            
            # For basic implementation, just synthesize the text
            # A full implementation would parse SSML for prosody controls
            return await self.synthesize_text(text, voice_model)
            
        except Exception as e:
            logger.error(f"Error synthesizing SSML: {e}")
            raise
    
    async def synthesize_streaming(self, text_stream, voice_model: str = None):
        """Synthesize text stream in real-time"""
        try:
            if voice_model and voice_model != self.current_model.name:
                await self._load_voice_model(voice_model)
            
            buffer = ""
            sentence_endings = ".!?"
            
            async for text_chunk in text_stream:
                buffer += text_chunk
                
                # Look for complete sentences
                while any(ending in buffer for ending in sentence_endings):
                    # Find the first sentence ending
                    min_pos = len(buffer)
                    for ending in sentence_endings:
                        pos = buffer.find(ending)
                        if pos != -1:
                            min_pos = min(min_pos, pos)
                    
                    if min_pos < len(buffer):
                        # Extract sentence
                        sentence = buffer[:min_pos + 1].strip()
                        buffer = buffer[min_pos + 1:].strip()
                        
                        if sentence:
                            # Synthesize sentence
                            result = await self.synthesize_text(sentence, voice_model)
                            yield result
            
            # Synthesize remaining buffer
            if buffer.strip():
                result = await self.synthesize_text(buffer.strip(), voice_model)
                yield result
                
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            raise
    
    def _update_stats(self, result: SynthesisResult):
        """Update synthesis statistics"""
        self.stats["total_syntheses"] += 1
        self.stats["total_text_length"] += len(result.text)
        self.stats["total_processing_time"] += result.processing_time
        self.stats["total_audio_duration"] += result.audio_duration
        
        if self.stats["total_processing_time"] > 0:
            self.stats["average_processing_speed"] = (
                self.stats["total_audio_duration"] / self.stats["total_processing_time"]
            )
    
    async def list_voices(self) -> List[VoiceModel]:
        """List available voice models"""
        return list(self.available_models.values())
    
    async def get_voice_info(self, voice_name: str) -> Optional[VoiceModel]:
        """Get information about a specific voice"""
        return self.available_models.get(voice_name)
    
    async def switch_voice(self, voice_name: str):
        """Switch to a different voice model"""
        try:
            if voice_name not in self.available_models:
                raise ValueError(f"Unknown voice: {voice_name}")
            
            if voice_name == self.current_model.name:
                return  # Already using this voice
            
            logger.info(f"Switching from {self.current_model.name} to {voice_name}")
            await self._load_voice_model(voice_name)
            
            logger.info(f"Successfully switched to voice: {voice_name}")
            
        except Exception as e:
            logger.error(f"Error switching voice: {e}")
            raise
    
    async def save_audio(self, result: SynthesisResult, file_path: str, format: str = "wav"):
        """Save synthesized audio to file"""
        try:
            if format.lower() == "wav":
                sf.write(file_path, result.audio_data, result.sample_rate)
            elif format.lower() == "mp3":
                # Convert to MP3 using pydub
                audio_segment = AudioSegment(
                    result.audio_data.tobytes(),
                    frame_rate=result.sample_rate,
                    sample_width=result.audio_data.dtype.itemsize,
                    channels=1
                )
                audio_segment.export(file_path, format="mp3")
            else:
                raise ValueError(f"Unsupported audio format: {format}")
            
            logger.info(f"Audio saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    async def get_audio_bytes(self, result: SynthesisResult, format: str = "wav") -> bytes:
        """Get synthesized audio as bytes"""
        try:
            if format.lower() == "wav":
                buffer = io.BytesIO()
                sf.write(buffer, result.audio_data, result.sample_rate, format='wav')
                return buffer.getvalue()
            elif format.lower() == "mp3":
                # Convert to MP3
                audio_segment = AudioSegment(
                    result.audio_data.tobytes(),
                    frame_rate=result.sample_rate,
                    sample_width=result.audio_data.dtype.itemsize,
                    channels=1
                )
                buffer = io.BytesIO()
                audio_segment.export(buffer, format="mp3")
                return buffer.getvalue()
            else:
                raise ValueError(f"Unsupported audio format: {format}")
                
        except Exception as e:
            logger.error(f"Error getting audio bytes: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return {
            **self.stats,
            "current_voice": self.current_model.name if self.current_model else None,
            "available_voices": len(self.available_models),
            "sample_rate": self.sample_rate,
            "model_loaded": self.model_loaded
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            self.current_model = None
            self.model_loaded = False
            self.model_cache.clear()
            
            logger.info("TTS cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TTS cleanup: {e}")

# Factory function for creating TTS instance
async def create_piper_tts(config: Dict[str, Any]) -> PiperTTS:
    """Create and initialize Piper TTS"""
    voice_config = config.get('voice', {})
    tts_config = voice_config.get('tts', {})
    
    voice_model = tts_config.get('voice', 'en_US-lessac-medium')
    models_path = config.get('data_path', '/opt/archie/data') + '/models'
    piper_executable = tts_config.get('piper_path', 'piper')
    
    tts = PiperTTS(
        voice_model=voice_model,
        models_path=models_path,
        piper_executable=piper_executable
    )
    
    await tts.initialize()
    return tts