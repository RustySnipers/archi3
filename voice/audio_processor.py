"""
Audio processing pipeline for Archie
Handles audio input/output, preprocessing, and real-time processing
"""

import os
import io
import logging
import asyncio
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

# Audio processing imports
import sounddevice as sd
import soundfile as sf
import librosa
from scipy import signal
import webrtcvad
import collections

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "float32"
    device_index: Optional[int] = None
    latency: str = "low"

@dataclass
class AudioChunk:
    """Audio data chunk"""
    data: np.ndarray
    timestamp: datetime
    sample_rate: int
    chunk_id: str
    is_speech: bool = False
    volume_level: float = 0.0

@dataclass
class VoiceActivityResult:
    """Voice activity detection result"""
    is_voice: bool
    confidence: float
    timestamp: datetime

class AudioProcessor:
    """Core audio processing system"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.is_recording = False
        self.is_playing = False
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Processing queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.callbacks = {}
        
        # Voice Activity Detection
        self.vad = None
        self.vad_enabled = True
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        
        # Audio preprocessing
        self.noise_reduction_enabled = True
        self.auto_gain_control = True
        self.echo_cancellation = True
        
        # Buffer management
        self.audio_buffer = collections.deque(maxlen=50)  # Keep last 50 chunks
        self.speech_buffer = collections.deque(maxlen=100)
        
        # Statistics
        self.stats = {
            "total_chunks_processed": 0,
            "speech_chunks_detected": 0,
            "total_audio_time": 0.0,
            "processing_errors": 0,
            "current_volume": 0.0,
            "peak_volume": 0.0
        }
        
        # Initialize VAD
        self._initialize_vad()
    
    def _initialize_vad(self):
        """Initialize Voice Activity Detection"""
        try:
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            logger.info(f"VAD initialized with aggressiveness level {self.vad_aggressiveness}")
        except Exception as e:
            logger.warning(f"Failed to initialize VAD: {e}")
            self.vad = None
            self.vad_enabled = False
    
    async def initialize(self):
        """Initialize audio system"""
        try:
            # Check available audio devices
            await self._check_audio_devices()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._audio_processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("Audio processor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            raise
    
    async def _check_audio_devices(self):
        """Check and log available audio devices"""
        try:
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            
            # Log input devices
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            logger.info(f"Input devices: {len(input_devices)}")
            for i, device in enumerate(input_devices):
                logger.debug(f"  {i}: {device['name']} (channels: {device['max_input_channels']})")
            
            # Log output devices  
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            logger.info(f"Output devices: {len(output_devices)}")
            for i, device in enumerate(output_devices):
                logger.debug(f"  {i}: {device['name']} (channels: {device['max_output_channels']})")
            
            # Set default device if not specified
            if self.config.device_index is None:
                default_device = sd.query_devices(kind='input')
                logger.info(f"Using default input device: {default_device['name']}")
                
        except Exception as e:
            logger.error(f"Error checking audio devices: {e}")
            raise
    
    def _audio_processing_loop(self):
        """Background audio processing loop"""
        logger.info("Audio processing loop started")
        
        while True:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=1.0)
                
                if chunk is None:  # Shutdown signal
                    break
                
                # Process the chunk
                processed_chunk = self._process_audio_chunk(chunk)
                
                # Add to buffer
                self.audio_buffer.append(processed_chunk)
                
                # Update statistics
                self._update_stats(processed_chunk)
                
                # Notify callbacks
                self._notify_callbacks('audio_chunk', processed_chunk)
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                self.stats["processing_errors"] += 1
    
    def _process_audio_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """Process individual audio chunk"""
        try:
            # Make a copy to avoid modifying original
            processed_data = chunk.data.copy()
            
            # Apply preprocessing
            if self.noise_reduction_enabled:
                processed_data = self._apply_noise_reduction(processed_data)
            
            if self.auto_gain_control:
                processed_data = self._apply_agc(processed_data)
            
            # Calculate volume level
            volume_level = np.sqrt(np.mean(processed_data ** 2))
            
            # Voice activity detection
            is_speech = False
            if self.vad_enabled and self.vad:
                is_speech = self._detect_voice_activity(processed_data)
            
            # Create processed chunk
            processed_chunk = AudioChunk(
                data=processed_data,
                timestamp=chunk.timestamp,
                sample_rate=chunk.sample_rate,
                chunk_id=chunk.chunk_id,
                is_speech=is_speech,
                volume_level=volume_level
            )
            
            # Add to speech buffer if speech detected
            if is_speech:
                self.speech_buffer.append(processed_chunk)
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return chunk  # Return original on error
    
    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction"""
        try:
            # Simple spectral subtraction-based noise reduction
            # This is a basic implementation; production would use more sophisticated methods
            
            # Apply high-pass filter to remove low-frequency noise
            nyquist = self.config.sample_rate / 2
            low_cutoff = 80.0  # Hz
            high_cutoff = 7000.0  # Hz
            
            # Design Butterworth bandpass filter
            low = low_cutoff / nyquist
            high = high_cutoff / nyquist
            
            if high < 1.0:  # Ensure we don't exceed Nyquist frequency
                b, a = signal.butter(4, [low, high], btype='band')
                filtered_data = signal.filtfilt(b, a, audio_data)
            else:
                # Just apply high-pass filter
                b, a = signal.butter(4, low, btype='high')
                filtered_data = signal.filtfilt(b, a, audio_data)
            
            return filtered_data.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Error in noise reduction: {e}")
            return audio_data
    
    def _apply_agc(self, audio_data: np.ndarray, target_level: float = 0.3) -> np.ndarray:
        """Apply Automatic Gain Control"""
        try:
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > 0:
                # Calculate gain factor
                gain = target_level / rms
                
                # Limit gain to prevent excessive amplification
                max_gain = 10.0
                min_gain = 0.1
                gain = np.clip(gain, min_gain, max_gain)
                
                # Apply gain with smooth transitions
                return audio_data * gain
            
            return audio_data
            
        except Exception as e:
            logger.warning(f"Error in AGC: {e}")
            return audio_data
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detect voice activity in audio chunk"""
        try:
            if not self.vad:
                return False
            
            # Convert float32 to int16 for WebRTC VAD
            if audio_data.dtype == np.float32:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # WebRTC VAD requires specific frame sizes (10, 20, or 30 ms)
            frame_duration_ms = 30
            frame_size = int(self.config.sample_rate * frame_duration_ms / 1000)
            
            # Pad or truncate to frame size
            if len(audio_int16) < frame_size:
                padded = np.zeros(frame_size, dtype=np.int16)
                padded[:len(audio_int16)] = audio_int16
                audio_int16 = padded
            elif len(audio_int16) > frame_size:
                audio_int16 = audio_int16[:frame_size]
            
            # Check if this is voice
            is_speech = self.vad.is_speech(audio_int16.tobytes(), self.config.sample_rate)
            
            return is_speech
            
        except Exception as e:
            logger.warning(f"Error in voice activity detection: {e}")
            return False
    
    def _update_stats(self, chunk: AudioChunk):
        """Update processing statistics"""
        self.stats["total_chunks_processed"] += 1
        
        if chunk.is_speech:
            self.stats["speech_chunks_detected"] += 1
        
        # Update volume stats
        self.stats["current_volume"] = chunk.volume_level
        if chunk.volume_level > self.stats["peak_volume"]:
            self.stats["peak_volume"] = chunk.volume_level
        
        # Update total audio time
        chunk_duration = len(chunk.data) / chunk.sample_rate
        self.stats["total_audio_time"] += chunk_duration
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
    
    async def start_recording(self):
        """Start audio recording"""
        try:
            if self.is_recording:
                logger.warning("Recording already started")
                return
            
            logger.info("Starting audio recording")
            
            # Audio callback function
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                try:
                    # Create audio chunk
                    chunk = AudioChunk(
                        data=indata[:, 0].copy(),  # Take first channel
                        timestamp=datetime.now(),
                        sample_rate=self.config.sample_rate,
                        chunk_id=f"chunk_{int(time.inputBufferAdcTime * 1000)}"
                    )
                    
                    # Add to processing queue
                    if not self.audio_queue.full():
                        self.audio_queue.put(chunk, block=False)
                    else:
                        logger.warning("Audio queue full, dropping chunk")
                        
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
            
            # Start input stream
            self.input_stream = sd.InputStream(
                device=self.config.device_index,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                dtype=self.config.format,
                callback=audio_callback,
                latency=self.config.latency
            )
            
            self.input_stream.start()
            self.is_recording = True
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            raise
    
    async def stop_recording(self):
        """Stop audio recording"""
        try:
            if not self.is_recording:
                return
            
            logger.info("Stopping audio recording")
            
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            
            self.is_recording = False
            logger.info("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
    
    async def play_audio(self, audio_data: np.ndarray, sample_rate: int = None):
        """Play audio data"""
        try:
            if sample_rate is None:
                sample_rate = self.config.sample_rate
            
            logger.info(f"Playing audio: {len(audio_data)} samples at {sample_rate} Hz")
            
            # Ensure audio is in correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Reshape for stereo if needed
            if self.config.channels == 2 and audio_data.ndim == 1:
                audio_data = np.column_stack([audio_data, audio_data])
            
            # Play audio synchronously
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()  # Wait until playback is finished
            
            logger.info("Audio playback completed")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            raise
    
    async def record_audio_for_duration(self, duration: float) -> np.ndarray:
        """Record audio for a specific duration"""
        try:
            logger.info(f"Recording audio for {duration} seconds")
            
            # Calculate number of samples
            num_samples = int(duration * self.config.sample_rate)
            
            # Record audio
            audio_data = sd.rec(
                frames=num_samples,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.format,
                device=self.config.device_index
            )
            
            # Wait for recording to complete
            sd.wait()
            
            # Return first channel if mono
            if self.config.channels == 1:
                return audio_data[:, 0]
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            raise
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for audio events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        logger.info(f"Registered callback for {event_type}")
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """Unregister callback"""
        if event_type in self.callbacks:
            try:
                self.callbacks[event_type].remove(callback)
                logger.info(f"Unregistered callback for {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for {event_type}")
    
    async def get_speech_segments(self, min_speech_duration: float = 0.5) -> List[AudioChunk]:
        """Get recent speech segments"""
        try:
            speech_segments = []
            current_segment = []
            
            # Process speech buffer
            for chunk in self.speech_buffer:
                if chunk.is_speech:
                    current_segment.append(chunk)
                else:
                    # End of speech segment
                    if current_segment:
                        # Check if segment is long enough
                        segment_duration = sum(
                            len(c.data) / c.sample_rate for c in current_segment
                        )
                        
                        if segment_duration >= min_speech_duration:
                            # Concatenate chunks
                            segment_data = np.concatenate([c.data for c in current_segment])
                            
                            speech_chunk = AudioChunk(
                                data=segment_data,
                                timestamp=current_segment[0].timestamp,
                                sample_rate=current_segment[0].sample_rate,
                                chunk_id=f"speech_segment_{len(speech_segments)}",
                                is_speech=True,
                                volume_level=np.mean([c.volume_level for c in current_segment])
                            )
                            
                            speech_segments.append(speech_chunk)
                        
                        current_segment = []
            
            # Handle final segment
            if current_segment:
                segment_duration = sum(
                    len(c.data) / c.sample_rate for c in current_segment
                )
                
                if segment_duration >= min_speech_duration:
                    segment_data = np.concatenate([c.data for c in current_segment])
                    
                    speech_chunk = AudioChunk(
                        data=segment_data,
                        timestamp=current_segment[0].timestamp,
                        sample_rate=current_segment[0].sample_rate,
                        chunk_id=f"speech_segment_{len(speech_segments)}",
                        is_speech=True,
                        volume_level=np.mean([c.volume_level for c in current_segment])
                    )
                    
                    speech_segments.append(speech_chunk)
            
            return speech_segments
            
        except Exception as e:
            logger.error(f"Error getting speech segments: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audio processing statistics"""
        return {
            **self.stats,
            "is_recording": self.is_recording,
            "is_playing": self.is_playing,
            "queue_size": self.audio_queue.qsize(),
            "buffer_size": len(self.audio_buffer),
            "speech_buffer_size": len(self.speech_buffer),
            "vad_enabled": self.vad_enabled,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels
        }
    
    async def cleanup(self):
        """Clean up audio resources"""
        try:
            logger.info("Cleaning up audio processor")
            
            # Stop recording
            await self.stop_recording()
            
            # Stop processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.audio_queue.put(None)  # Shutdown signal
                self.processing_thread.join(timeout=5.0)
            
            # Clear buffers
            self.audio_buffer.clear()
            self.speech_buffer.clear()
            
            # Clear callbacks
            self.callbacks.clear()
            
            logger.info("Audio processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during audio cleanup: {e}")

# Factory function for creating audio processor
def create_audio_processor(config: Dict[str, Any]) -> AudioProcessor:
    """Create audio processor with configuration"""
    voice_config = config.get('voice', {})
    audio_config = voice_config.get('audio', {})
    
    audio_cfg = AudioConfig(
        sample_rate=audio_config.get('sample_rate', 16000),
        channels=audio_config.get('channels', 1),
        chunk_size=audio_config.get('chunk_size', 1024),
        format=audio_config.get('format', 'float32'),
        device_index=audio_config.get('device_index'),
        latency=audio_config.get('latency', 'low')
    )
    
    return AudioProcessor(audio_cfg)