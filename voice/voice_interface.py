"""
Voice Interface Management System
Coordinates STT, TTS, and audio processing for voice interaction
"""

import os
import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from .stt import WhisperSTT, TranscriptionResult, create_whisper_stt
from .tts import PiperTTS, SynthesisResult, create_piper_tts
from .audio_processor import AudioProcessor, AudioChunk, AudioConfig, create_audio_processor

logger = logging.getLogger(__name__)

class VoiceState(Enum):
    """Voice interface states"""
    IDLE = "idle"
    LISTENING = "listening" 
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class VoiceCommand:
    """Voice command structure"""
    text: str
    confidence: float
    timestamp: datetime
    language: str
    processing_time: float
    audio_duration: float

@dataclass
class VoiceResponse:
    """Voice response structure"""
    text: str
    audio_data: bytes
    voice_model: str
    processing_time: float
    audio_duration: float
    format: str

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    turn_id: str
    user_input: VoiceCommand
    agent_response: VoiceResponse
    timestamp: datetime
    processing_time: float

class VoiceInterface:
    """Main voice interface coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = VoiceState.IDLE
        
        # Voice components
        self.stt = None
        self.tts = None
        self.audio_processor = None
        
        # Voice settings
        self.wake_word = config.get('voice', {}).get('wake_word', 'archie')
        self.continuous_listening = False
        self.auto_response = True
        self.response_timeout = 5.0  # seconds
        
        # Conversation management
        self.conversation_history = []
        self.current_turn = None
        self.max_history_length = 100
        
        # Callbacks for external integration
        self.callbacks = {
            'on_wake_word': [],
            'on_command': [],
            'on_response': [],
            'on_error': [],
            'on_state_change': []
        }
        
        # Voice activity settings
        self.silence_timeout = 2.0  # seconds of silence before processing
        self.max_recording_duration = 30.0  # maximum recording length
        self.min_speech_duration = 0.5  # minimum speech to process
        
        # Performance monitoring
        self.stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "total_conversation_time": 0.0,
            "average_response_time": 0.0,
            "wake_word_detections": 0,
            "false_wake_words": 0
        }
        
        # Internal state
        self.listening_task = None
        self.recording_start_time = None
        self.last_activity_time = datetime.now()
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all voice components"""
        try:
            logger.info("Initializing voice interface...")
            
            # Initialize STT
            self.stt = await create_whisper_stt(self.config)
            logger.info("STT initialized")
            
            # Initialize TTS
            self.tts = await create_piper_tts(self.config)
            logger.info("TTS initialized")
            
            # Initialize audio processor
            self.audio_processor = create_audio_processor(self.config)
            await self.audio_processor.initialize()
            logger.info("Audio processor initialized")
            
            # Register audio callbacks
            self.audio_processor.register_callback('audio_chunk', self._on_audio_chunk)
            
            # Start listening if continuous mode enabled
            if self.continuous_listening:
                await self.start_listening()
            
            logger.info("Voice interface fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice interface: {e}")
            await self._set_state(VoiceState.ERROR)
            raise
    
    async def _set_state(self, new_state: VoiceState):
        """Set voice interface state and notify callbacks"""
        old_state = self.state
        self.state = new_state
        
        if old_state != new_state:
            logger.info(f"Voice state changed: {old_state.value} -> {new_state.value}")
            await self._notify_callbacks('on_state_change', {
                'old_state': old_state.value,
                'new_state': new_state.value,
                'timestamp': datetime.now().isoformat()
            })
    
    async def start_listening(self):
        """Start continuous listening mode"""
        try:
            if self.state == VoiceState.LISTENING:
                logger.warning("Already listening")
                return
            
            logger.info("Starting continuous listening mode")
            await self._set_state(VoiceState.LISTENING)
            
            # Start audio recording
            await self.audio_processor.start_recording()
            
            # Start listening task
            self.listening_task = asyncio.create_task(self._listening_loop())
            
            self.continuous_listening = True
            
        except Exception as e:
            logger.error(f"Error starting listening: {e}")
            await self._set_state(VoiceState.ERROR)
            raise
    
    async def stop_listening(self):
        """Stop continuous listening mode"""
        try:
            logger.info("Stopping continuous listening mode")
            
            self.continuous_listening = False
            
            # Stop listening task
            if self.listening_task:
                self.listening_task.cancel()
                try:
                    await self.listening_task
                except asyncio.CancelledError:
                    pass
                self.listening_task = None
            
            # Stop audio recording
            await self.audio_processor.stop_recording()
            
            await self._set_state(VoiceState.IDLE)
            
        except Exception as e:
            logger.error(f"Error stopping listening: {e}")
    
    async def _listening_loop(self):
        """Main listening loop for continuous mode"""
        try:
            while self.continuous_listening and not self._shutdown_event.is_set():
                try:
                    # Check for speech segments
                    speech_segments = await self.audio_processor.get_speech_segments(
                        min_speech_duration=self.min_speech_duration
                    )
                    
                    for segment in speech_segments:
                        if not self.continuous_listening:
                            break
                        
                        await self._process_audio_segment(segment)
                    
                    # Clear processed segments
                    self.audio_processor.speech_buffer.clear()
                    
                    # Small delay to prevent excessive CPU usage
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in listening loop: {e}")
                    await asyncio.sleep(1.0)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Listening loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in listening loop: {e}")
            await self._set_state(VoiceState.ERROR)
    
    async def _process_audio_segment(self, segment: AudioChunk):
        """Process a speech audio segment"""
        try:
            await self._set_state(VoiceState.PROCESSING)
            start_time = datetime.now()
            
            # Transcribe audio
            transcription = await self.stt.transcribe_audio(segment.data)
            
            # Check if this contains wake word or is a command
            if await self._contains_wake_word(transcription.text):
                # Extract command after wake word
                command_text = self._extract_command_from_text(transcription.text)
                
                if command_text.strip():
                    # Create voice command
                    voice_command = VoiceCommand(
                        text=command_text,
                        confidence=transcription.confidence,
                        timestamp=start_time,
                        language=transcription.language,
                        processing_time=transcription.processing_time,
                        audio_duration=transcription.audio_duration
                    )
                    
                    # Process the command
                    await self._handle_voice_command(voice_command)
                else:
                    # Just wake word, wait for actual command
                    await self._notify_callbacks('on_wake_word', {
                        'timestamp': datetime.now().isoformat(),
                        'confidence': transcription.confidence
                    })
                    self.stats["wake_word_detections"] += 1
            
            await self._set_state(VoiceState.LISTENING)
            
        except Exception as e:
            logger.error(f"Error processing audio segment: {e}")
            await self._set_state(VoiceState.ERROR)
    
    async def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains the wake word"""
        try:
            # Simple case-insensitive check
            wake_words = [self.wake_word.lower()]
            text_lower = text.lower()
            
            for wake_word in wake_words:
                if wake_word in text_lower:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking wake word: {e}")
            return False
    
    def _extract_command_from_text(self, text: str) -> str:
        """Extract command text after wake word"""
        try:
            text_lower = text.lower()
            wake_word_lower = self.wake_word.lower()
            
            # Find wake word position
            wake_pos = text_lower.find(wake_word_lower)
            if wake_pos == -1:
                return text  # No wake word found, return full text
            
            # Extract text after wake word
            command_start = wake_pos + len(wake_word_lower)
            command_text = text[command_start:].strip()
            
            # Remove common filler words at the beginning
            filler_words = ['please', 'can you', 'could you', 'would you']
            for filler in filler_words:
                if command_text.lower().startswith(filler):
                    command_text = command_text[len(filler):].strip()
                    break
            
            return command_text
            
        except Exception as e:
            logger.error(f"Error extracting command: {e}")
            return text
    
    async def _handle_voice_command(self, command: VoiceCommand):
        """Handle a processed voice command"""
        try:
            logger.info(f"Processing voice command: '{command.text}' (confidence: {command.confidence:.2f})")
            
            # Update statistics
            self.stats["total_commands"] += 1
            
            # Notify callbacks
            await self._notify_callbacks('on_command', asdict(command))
            
            # Start new conversation turn
            turn_id = f"turn_{len(self.conversation_history)}_{int(datetime.now().timestamp())}"
            self.current_turn = ConversationTurn(
                turn_id=turn_id,
                user_input=command,
                agent_response=None,
                timestamp=command.timestamp,
                processing_time=0.0
            )
            
            # If auto-response is enabled, we'll wait for the agent to provide a response
            # Otherwise, external system needs to call provide_response()
            
        except Exception as e:
            logger.error(f"Error handling voice command: {e}")
            self.stats["failed_commands"] += 1
            await self._notify_callbacks('on_error', {
                'error': str(e),
                'command': command.text if command else None,
                'timestamp': datetime.now().isoformat()
            })
    
    async def provide_response(self, response_text: str, voice_model: str = None) -> VoiceResponse:
        """Provide text response that will be converted to speech"""
        try:
            if not self.current_turn:
                logger.warning("No active conversation turn for response")
                return None
            
            await self._set_state(VoiceState.SPEAKING)
            start_time = datetime.now()
            
            # Synthesize speech
            synthesis_result = await self.tts.synthesize_text(
                response_text,
                voice_model=voice_model
            )
            
            # Get audio bytes
            audio_bytes = await self.tts.get_audio_bytes(synthesis_result, format="wav")
            
            # Create voice response
            voice_response = VoiceResponse(
                text=response_text,
                audio_data=audio_bytes,
                voice_model=synthesis_result.voice_model,
                processing_time=synthesis_result.processing_time,
                audio_duration=synthesis_result.audio_duration,
                format="wav"
            )
            
            # Play the response
            await self.audio_processor.play_audio(
                synthesis_result.audio_data,
                synthesis_result.sample_rate
            )
            
            # Complete conversation turn
            total_processing_time = (datetime.now() - self.current_turn.timestamp).total_seconds()
            self.current_turn.agent_response = voice_response
            self.current_turn.processing_time = total_processing_time
            
            # Add to conversation history
            self.conversation_history.append(self.current_turn)
            
            # Trim history if needed
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Update statistics
            self.stats["successful_commands"] += 1
            self.stats["total_conversation_time"] += total_processing_time
            self.stats["average_response_time"] = (
                self.stats["total_conversation_time"] / self.stats["successful_commands"]
            )
            
            # Notify callbacks
            await self._notify_callbacks('on_response', asdict(voice_response))
            
            # Reset current turn
            self.current_turn = None
            
            # Return to listening state
            await self._set_state(VoiceState.LISTENING)
            
            logger.info(f"Completed voice response in {total_processing_time:.2f}s")
            return voice_response
            
        except Exception as e:
            logger.error(f"Error providing voice response: {e}")
            self.stats["failed_commands"] += 1
            await self._set_state(VoiceState.ERROR)
            await self._notify_callbacks('on_error', {
                'error': str(e),
                'response_text': response_text,
                'timestamp': datetime.now().isoformat()
            })
            return None
    
    async def record_command(self, duration: float = None) -> Optional[VoiceCommand]:
        """Record a single voice command (non-continuous mode)"""
        try:
            logger.info(f"Recording voice command for {duration or 'variable'} duration")
            await self._set_state(VoiceState.LISTENING)
            
            if duration:
                # Fixed duration recording
                audio_data = await self.audio_processor.record_audio_for_duration(duration)
            else:
                # Variable duration with voice activity detection
                await self.audio_processor.start_recording()
                
                # Wait for speech and silence
                speech_detected = False
                silence_start = None
                start_time = datetime.now()
                
                while True:
                    # Check timeout
                    if (datetime.now() - start_time).total_seconds() > self.max_recording_duration:
                        logger.warning("Recording timeout reached")
                        break
                    
                    # Get recent speech segments
                    speech_segments = await self.audio_processor.get_speech_segments()
                    
                    if speech_segments:
                        speech_detected = True
                        silence_start = None
                        # Continue recording
                    elif speech_detected:
                        # Speech was detected but now we have silence
                        if silence_start is None:
                            silence_start = datetime.now()
                        elif (datetime.now() - silence_start).total_seconds() > self.silence_timeout:
                            # Enough silence, stop recording
                            break
                    
                    await asyncio.sleep(0.1)
                
                await self.audio_processor.stop_recording()
                
                # Get all recorded speech
                speech_segments = await self.audio_processor.get_speech_segments()
                if not speech_segments:
                    logger.warning("No speech detected in recording")
                    await self._set_state(VoiceState.IDLE)
                    return None
                
                # Concatenate all speech segments
                import numpy as np
                audio_data = np.concatenate([seg.data for seg in speech_segments])
            
            await self._set_state(VoiceState.PROCESSING)
            
            # Transcribe the recorded audio
            transcription = await self.stt.transcribe_audio(audio_data)
            
            # Create voice command
            voice_command = VoiceCommand(
                text=transcription.text,
                confidence=transcription.confidence,
                timestamp=datetime.now(),
                language=transcription.language,
                processing_time=transcription.processing_time,
                audio_duration=transcription.audio_duration
            )
            
            await self._set_state(VoiceState.IDLE)
            
            logger.info(f"Recorded command: '{voice_command.text}' (confidence: {voice_command.confidence:.2f})")
            return voice_command
            
        except Exception as e:
            logger.error(f"Error recording command: {e}")
            await self._set_state(VoiceState.ERROR)
            return None
    
    async def _on_audio_chunk(self, chunk: AudioChunk):
        """Handle audio chunk from processor"""
        try:
            # Update last activity time if speech detected
            if chunk.is_speech:
                self.last_activity_time = datetime.now()
            
            # Update current volume in stats
            self.stats["current_volume"] = chunk.volume_level
            
        except Exception as e:
            logger.error(f"Error handling audio chunk: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for voice events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def unregister_callback(self, event_type: str, callback: Callable):
        """Unregister callback"""
        if event_type in self.callbacks:
            try:
                self.callbacks[event_type].remove(callback)
                logger.info(f"Unregistered callback for {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for {event_type}")
    
    async def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}")
    
    def get_conversation_history(self, limit: int = None) -> List[ConversationTurn]:
        """Get conversation history"""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice interface statistics"""
        stats = {
            **self.stats,
            "state": self.state.value,
            "continuous_listening": self.continuous_listening,
            "conversation_turns": len(self.conversation_history),
            "last_activity": self.last_activity_time.isoformat(),
            "wake_word": self.wake_word
        }
        
        # Add component stats
        if self.stt:
            stats["stt_stats"] = self.stt.get_stats()
        
        if self.tts:
            stats["tts_stats"] = self.tts.get_stats()
        
        if self.audio_processor:
            stats["audio_stats"] = self.audio_processor.get_stats()
        
        return stats
    
    async def set_wake_word(self, new_wake_word: str):
        """Change the wake word"""
        old_wake_word = self.wake_word
        self.wake_word = new_wake_word.lower()
        logger.info(f"Wake word changed from '{old_wake_word}' to '{self.wake_word}'")
    
    async def set_voice_model(self, voice_model: str):
        """Change TTS voice model"""
        if self.tts:
            await self.tts.switch_voice(voice_model)
            logger.info(f"Voice model changed to: {voice_model}")
    
    async def shutdown(self):
        """Shutdown voice interface"""
        try:
            logger.info("Shutting down voice interface")
            self._shutdown_event.set()
            
            # Stop listening
            await self.stop_listening()
            
            # Cleanup components
            if self.audio_processor:
                await self.audio_processor.cleanup()
            
            if self.stt:
                await self.stt.cleanup()
            
            if self.tts:
                await self.tts.cleanup()
            
            logger.info("Voice interface shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during voice interface shutdown: {e}")

# Factory function for creating voice interface
async def create_voice_interface(config: Dict[str, Any]) -> VoiceInterface:
    """Create and initialize voice interface"""
    interface = VoiceInterface(config)
    await interface.initialize()
    return interface