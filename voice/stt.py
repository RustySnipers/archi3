"""
Speech-to-Text (STT) system using Whisper
Handles audio transcription with model management and optimization
"""

import os
import io
import json
import logging
import asyncio
import tempfile
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import wave
import threading
import queue

# Whisper and audio processing imports
import whisper
import torch
import soundfile as sf
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """Result of speech transcription"""
    text: str
    confidence: float
    language: str
    segments: List[Dict[str, Any]]
    processing_time: float
    model_used: str
    audio_duration: float

@dataclass
class AudioChunk:
    """Audio chunk for processing"""
    data: np.ndarray
    sample_rate: int
    timestamp: datetime
    chunk_id: str
    duration: float

class WhisperSTT:
    """Whisper-based Speech-to-Text system"""
    
    def __init__(self, 
                 model_name: str = "small.en",
                 device: str = "auto",
                 models_path: str = "/opt/archie/data/models"):
        
        self.model_name = model_name
        self.device = self._get_device(device)
        self.models_path = models_path
        self.model = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
        
        # Audio processing settings
        self.target_sample_rate = 16000
        self.chunk_length = 30  # seconds
        self.overlap_length = 5  # seconds for chunk overlap
        
        # Performance settings
        self.fp16 = torch.cuda.is_available()
        self.batch_size = 1
        
        # Language settings
        self.default_language = "en"
        self.auto_detect_language = True
        
        # Statistics
        self.stats = {
            "total_transcriptions": 0,
            "total_audio_time": 0.0,
            "total_processing_time": 0.0,
            "average_processing_speed": 0.0,
            "model_loads": 0,
            "errors": 0
        }
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for processing"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    async def initialize(self):
        """Initialize the STT system"""
        try:
            await self._load_model()
            logger.info(f"Whisper STT initialized with model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper STT: {e}")
            raise
    
    async def _load_model(self):
        """Load Whisper model"""
        if self.model_loaded:
            return
        
        with self.loading_lock:
            if self.model_loaded:  # Double-check after acquiring lock
                return
            
            try:
                logger.info(f"Loading Whisper model: {self.model_name}")
                start_time = datetime.now()
                
                # Load model in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None,
                    self._load_model_sync
                )
                
                load_time = (datetime.now() - start_time).total_seconds()
                self.model_loaded = True
                self.stats["model_loads"] += 1
                
                logger.info(f"Model loaded in {load_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        try:
            # Try to load from local path first
            model_path = os.path.join(self.models_path, f"whisper-{self.model_name}.pt")
            
            if os.path.exists(model_path):
                model = whisper.load_model(model_path, device=self.device)
            else:
                # Download model if not found locally
                model = whisper.load_model(self.model_name, device=self.device)
                # Save model locally for future use
                torch.save(model.state_dict(), model_path)
            
            return model
            
        except Exception as e:
            logger.error(f"Error in synchronous model loading: {e}")
            raise
    
    async def transcribe_audio(self, 
                             audio_data: Union[str, bytes, np.ndarray],
                             language: str = None,
                             temperature: float = 0.0,
                             beam_size: int = 5,
                             best_of: int = 5) -> TranscriptionResult:
        """Transcribe audio data to text"""
        try:
            start_time = datetime.now()
            
            # Ensure model is loaded
            await self._load_model()
            
            # Preprocess audio
            audio_array, sample_rate, duration = await self._preprocess_audio(audio_data)
            
            # Determine language
            if language is None:
                language = self.default_language if not self.auto_detect_language else None
            
            # Transcribe in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_array,
                language,
                temperature,
                beam_size,
                best_of
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create transcription result
            transcription_result = TranscriptionResult(
                text=result["text"].strip(),
                confidence=self._calculate_confidence(result),
                language=result.get("language", language or "en"),
                segments=result.get("segments", []),
                processing_time=processing_time,
                model_used=self.model_name,
                audio_duration=duration
            )
            
            # Update statistics
            self._update_stats(transcription_result)
            
            logger.info(f"Transcribed {duration:.2f}s audio in {processing_time:.2f}s")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            self.stats["errors"] += 1
            raise
    
    def _transcribe_sync(self, 
                        audio: np.ndarray,
                        language: str,
                        temperature: float,
                        beam_size: int,
                        best_of: int) -> Dict[str, Any]:
        """Synchronous transcription"""
        try:
            options = {
                "language": language,
                "temperature": temperature,
                "beam_size": beam_size,
                "best_of": best_of,
                "fp16": self.fp16,
                "verbose": False
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            result = self.model.transcribe(audio, **options)
            return result
            
        except Exception as e:
            logger.error(f"Error in synchronous transcription: {e}")
            raise
    
    async def _preprocess_audio(self, audio_data: Union[str, bytes, np.ndarray]) -> tuple:
        """Preprocess audio data for transcription"""
        try:
            if isinstance(audio_data, str):
                # File path
                audio_array, sample_rate = librosa.load(audio_data, sr=self.target_sample_rate)
            elif isinstance(audio_data, bytes):
                # Raw audio bytes
                audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
                if sample_rate != self.target_sample_rate:
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.target_sample_rate)
                    sample_rate = self.target_sample_rate
            elif isinstance(audio_data, np.ndarray):
                # NumPy array
                audio_array = audio_data
                sample_rate = self.target_sample_rate  # Assume target sample rate
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")
            
            # Ensure audio is float32 and in correct range
            audio_array = audio_array.astype(np.float32)
            
            # Normalize audio
            audio_array = self._normalize_audio(audio_array)
            
            # Calculate duration
            duration = len(audio_array) / sample_rate
            
            return audio_array, sample_rate, duration
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Ensure audio is in [-1, 1] range
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        
        return audio
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.8  # Default confidence
            
            # Calculate average confidence from segments
            confidences = []
            for segment in segments:
                # Whisper doesn't provide direct confidence, so we estimate
                # based on segment properties
                segment_confidence = 1.0
                
                # Lower confidence for very short segments
                duration = segment.get("end", 0) - segment.get("start", 0)
                if duration < 0.5:
                    segment_confidence *= 0.8
                
                # Lower confidence for segments with many tokens per second
                tokens = segment.get("tokens", [])
                if duration > 0 and len(tokens) / duration > 10:
                    segment_confidence *= 0.9
                
                confidences.append(segment_confidence)
            
            return np.mean(confidences) if confidences else 0.8
            
        except Exception:
            return 0.8  # Default confidence on error
    
    def _update_stats(self, result: TranscriptionResult):
        """Update transcription statistics"""
        self.stats["total_transcriptions"] += 1
        self.stats["total_audio_time"] += result.audio_duration
        self.stats["total_processing_time"] += result.processing_time
        
        if self.stats["total_processing_time"] > 0:
            self.stats["average_processing_speed"] = (
                self.stats["total_audio_time"] / self.stats["total_processing_time"]
            )
    
    async def transcribe_long_audio(self, 
                                  audio_data: Union[str, bytes, np.ndarray],
                                  language: str = None) -> TranscriptionResult:
        """Transcribe long audio by chunking"""
        try:
            start_time = datetime.now()
            
            # Preprocess audio
            audio_array, sample_rate, duration = await self._preprocess_audio(audio_data)
            
            # If audio is short enough, use regular transcription
            if duration <= self.chunk_length:
                return await self.transcribe_audio(audio_data, language)
            
            # Split audio into chunks
            chunks = self._split_audio_into_chunks(audio_array, sample_rate)
            
            # Transcribe each chunk
            all_segments = []
            full_text = ""
            total_confidence = 0.0
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)}")
                
                chunk_result = await self.transcribe_audio(chunk.data, language)
                
                # Adjust segment timestamps
                time_offset = chunk.timestamp.timestamp() if hasattr(chunk.timestamp, 'timestamp') else 0
                adjusted_segments = []
                for segment in chunk_result.segments:
                    segment = segment.copy()
                    segment["start"] += time_offset
                    segment["end"] += time_offset
                    adjusted_segments.append(segment)
                
                all_segments.extend(adjusted_segments)
                full_text += chunk_result.text + " "
                total_confidence += chunk_result.confidence
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = TranscriptionResult(
                text=full_text.strip(),
                confidence=total_confidence / len(chunks) if chunks else 0.0,
                language=language or "en",
                segments=all_segments,
                processing_time=processing_time,
                model_used=self.model_name,
                audio_duration=duration
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing long audio: {e}")
            raise
    
    def _split_audio_into_chunks(self, audio: np.ndarray, sample_rate: int) -> List[AudioChunk]:
        """Split audio into overlapping chunks"""
        chunks = []
        chunk_samples = int(self.chunk_length * sample_rate)
        overlap_samples = int(self.overlap_length * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        for i in range(0, len(audio), step_samples):
            chunk_audio = audio[i:i + chunk_samples]
            
            if len(chunk_audio) < sample_rate:  # Skip chunks shorter than 1 second
                break
            
            chunk = AudioChunk(
                data=chunk_audio,
                sample_rate=sample_rate,
                timestamp=datetime.now(),
                chunk_id=f"chunk_{i}",
                duration=len(chunk_audio) / sample_rate
            )
            chunks.append(chunk)
        
        return chunks
    
    async def transcribe_realtime(self, audio_stream, callback_func):
        """Real-time transcription from audio stream"""
        try:
            buffer = np.array([], dtype=np.float32)
            buffer_duration = 0.0
            min_chunk_duration = 2.0  # Minimum 2 seconds before transcription
            
            async for audio_chunk in audio_stream:
                # Add to buffer
                buffer = np.concatenate([buffer, audio_chunk])
                buffer_duration = len(buffer) / self.target_sample_rate
                
                # Transcribe when we have enough audio
                if buffer_duration >= min_chunk_duration:
                    try:
                        result = await self.transcribe_audio(buffer)
                        await callback_func(result)
                        
                        # Keep last second for context
                        keep_samples = self.target_sample_rate
                        buffer = buffer[-keep_samples:]
                        buffer_duration = len(buffer) / self.target_sample_rate
                        
                    except Exception as e:
                        logger.error(f"Error in real-time transcription: {e}")
            
            # Transcribe remaining buffer
            if buffer_duration > 0.5:  # At least 0.5 seconds
                try:
                    result = await self.transcribe_audio(buffer)
                    await callback_func(result)
                except Exception as e:
                    logger.error(f"Error transcribing final buffer: {e}")
                    
        except Exception as e:
            logger.error(f"Error in real-time transcription: {e}")
            raise
    
    async def detect_language(self, audio_data: Union[str, bytes, np.ndarray]) -> str:
        """Detect the language of audio"""
        try:
            await self._load_model()
            
            # Use only first 30 seconds for language detection
            audio_array, sample_rate, duration = await self._preprocess_audio(audio_data)
            
            if duration > 30:
                audio_array = audio_array[:30 * sample_rate]
            
            # Detect language
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: whisper.detect_language(self.model, audio_array)
            )
            
            language, probability = result
            logger.info(f"Detected language: {language} (probability: {probability:.2f})")
            
            return language
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"  # Default to English
    
    async def get_available_models(self) -> List[str]:
        """Get list of available Whisper models"""
        return [
            "tiny", "tiny.en",
            "base", "base.en", 
            "small", "small.en",
            "medium", "medium.en",
            "large", "large-v1", "large-v2"
        ]
    
    async def switch_model(self, new_model: str):
        """Switch to a different Whisper model"""
        try:
            if new_model == self.model_name:
                return
            
            logger.info(f"Switching from {self.model_name} to {new_model}")
            
            # Unload current model
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.model = None
            self.model_loaded = False
            self.model_name = new_model
            
            # Load new model
            await self._load_model()
            
            logger.info(f"Successfully switched to model: {new_model}")
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics"""
        return {
            **self.stats,
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "target_sample_rate": self.target_sample_rate
        }
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
                self.model_loaded = False
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            logger.info("STT cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during STT cleanup: {e}")

# Factory function for creating STT instance
async def create_whisper_stt(config: Dict[str, Any]) -> WhisperSTT:
    """Create and initialize Whisper STT"""
    voice_config = config.get('voice', {})
    stt_config = voice_config.get('stt', {})
    
    model_name = stt_config.get('model', 'small.en')
    device = stt_config.get('device', 'auto')
    models_path = config.get('data_path', '/opt/archie/data') + '/models'
    
    stt = WhisperSTT(
        model_name=model_name,
        device=device,
        models_path=models_path
    )
    
    await stt.initialize()
    return stt