"""
Voice Agent Integration
Integrates voice interface with Archie's agent orchestration system
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .voice_interface import VoiceInterface, VoiceCommand, VoiceResponse, create_voice_interface
from .wake_word import WakeWordDetector, WakeWordDetection, create_wake_word_detector

logger = logging.getLogger(__name__)

class VoiceAgentBridge:
    """Bridge between voice interface and agent orchestration"""
    
    def __init__(self, orchestrator, config: Dict[str, Any]):
        self.orchestrator = orchestrator
        self.config = config
        
        # Voice components
        self.voice_interface = None
        self.wake_word_detector = None
        
        # Integration settings
        self.auto_response_enabled = True
        self.voice_priority = True  # Voice commands get priority
        self.conversation_timeout = 30.0  # seconds
        
        # State management
        self.active_voice_session = None
        self.pending_responses = {}
        
        # Statistics
        self.stats = {
            "voice_commands_processed": 0,
            "agent_responses_synthesized": 0,
            "integration_errors": 0,
            "average_response_time": 0.0
        }
    
    async def initialize(self):
        """Initialize voice-agent integration"""
        try:
            logger.info("Initializing voice-agent integration...")
            
            # Initialize voice interface
            self.voice_interface = await create_voice_interface(self.config)
            
            # Initialize wake word detector if enabled
            if self.config.get('voice', {}).get('wake_word_detection', True):
                self.wake_word_detector = await create_wake_word_detector(self.config)
                self.wake_word_detector.register_callback(self._on_wake_word_detected)
            
            # Register voice interface callbacks
            self.voice_interface.register_callback('on_command', self._on_voice_command)
            self.voice_interface.register_callback('on_error', self._on_voice_error)
            
            # Register as voice agent with orchestrator
            await self._register_voice_agent()
            
            # Start voice interface if auto-start enabled
            if self.config.get('voice', {}).get('auto_start', True):
                await self.start_voice_interface()
            
            logger.info("Voice-agent integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice-agent integration: {e}")
            raise
    
    async def _register_voice_agent(self):
        """Register voice capabilities with the orchestrator"""
        try:
            # Update voice agent capabilities
            if 'voice_agent' in self.orchestrator.agents:
                voice_agent = self.orchestrator.agents['voice_agent']
                
                # Add voice-specific capabilities
                voice_agent.capabilities.extend([
                    "voice_command_processing",
                    "speech_synthesis",
                    "wake_word_detection",
                    "continuous_listening"
                ])
                
                # Override process_message to handle voice commands
                original_process_message = voice_agent.process_message
                
                async def enhanced_process_message(message: str, context: Dict[str, Any] = None):
                    # Check if this is a voice command
                    if context and context.get('source') == 'voice':
                        return await self._process_voice_message(message, context)
                    else:
                        return await original_process_message(message, context)
                
                voice_agent.process_message = enhanced_process_message
                
                logger.info("Enhanced voice agent with integration capabilities")
            
        except Exception as e:
            logger.error(f"Error registering voice agent: {e}")
    
    async def start_voice_interface(self):
        """Start the voice interface"""
        try:
            if self.voice_interface:
                await self.voice_interface.start_listening()
                logger.info("Voice interface started")
            
        except Exception as e:
            logger.error(f"Error starting voice interface: {e}")
            raise
    
    async def stop_voice_interface(self):
        """Stop the voice interface"""
        try:
            if self.voice_interface:
                await self.voice_interface.stop_listening()
                logger.info("Voice interface stopped")
                
        except Exception as e:
            logger.error(f"Error stopping voice interface: {e}")
    
    async def _on_wake_word_detected(self, detection: WakeWordDetection):
        """Handle wake word detection"""
        try:
            logger.info(f"Wake word '{detection.word}' detected (confidence: {detection.confidence:.3f})")
            
            # Notify orchestrator about wake word
            if self.orchestrator and hasattr(self.orchestrator, 'message_bus'):
                wake_message = {
                    'type': 'wake_word_detected',
                    'word': detection.word,
                    'confidence': detection.confidence,
                    'timestamp': detection.timestamp.isoformat()
                }
                await self.orchestrator.message_bus.put(wake_message)
            
        except Exception as e:
            logger.error(f"Error handling wake word detection: {e}")
    
    async def _on_voice_command(self, command_data: Dict[str, Any]):
        """Handle voice command from interface"""
        try:
            logger.info(f"Processing voice command: '{command_data['text']}'")
            
            start_time = datetime.now()
            
            # Create context for voice command
            context = {
                'source': 'voice',
                'confidence': command_data['confidence'],
                'language': command_data['language'],
                'timestamp': command_data['timestamp'],
                'audio_duration': command_data['audio_duration']
            }
            
            # Route to appropriate agent or process directly
            if self.orchestrator:
                # Process through orchestrator
                response = await self.orchestrator.process_user_input(
                    command_data['text'], 
                    context
                )
                
                # Synthesize and play response if auto-response enabled
                if self.auto_response_enabled and response:
                    await self._provide_voice_response(response)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["voice_commands_processed"] += 1
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["voice_commands_processed"] - 1) + processing_time) /
                self.stats["voice_commands_processed"]
            )
            
        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            self.stats["integration_errors"] += 1
            
            # Provide error response
            if self.auto_response_enabled:
                await self._provide_voice_response("I'm sorry, I encountered an error processing that request.")
    
    async def _on_voice_error(self, error_data: Dict[str, Any]):
        """Handle voice interface errors"""
        try:
            logger.error(f"Voice interface error: {error_data['error']}")
            self.stats["integration_errors"] += 1
            
            # Optionally provide audio feedback for errors
            if self.auto_response_enabled:
                await self._provide_voice_response("I'm having trouble with my voice processing. Please try again.")
                
        except Exception as e:
            logger.error(f"Error handling voice error: {e}")
    
    async def _process_voice_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process voice message with enhanced context"""
        try:
            # Add voice-specific processing
            enhanced_context = {
                **context,
                'processing_mode': 'voice',
                'response_format': 'conversational',
                'max_response_length': 200  # Keep voice responses concise
            }
            
            # Process through orchestrator
            if self.orchestrator:
                response = await self.orchestrator.process_user_input(message, enhanced_context)
                return response
            
            return "I'm not connected to the main system right now."
            
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            return "I encountered an error processing your request."
    
    async def _provide_voice_response(self, response_text: str):
        """Provide voice response through TTS"""
        try:
            if self.voice_interface:
                voice_response = await self.voice_interface.provide_response(response_text)
                
                if voice_response:
                    self.stats["agent_responses_synthesized"] += 1
                    logger.info(f"Provided voice response: '{response_text[:50]}...'")
                    return voice_response
            
            return None
            
        except Exception as e:
            logger.error(f"Error providing voice response: {e}")
            self.stats["integration_errors"] += 1
    
    async def send_voice_notification(self, message: str, priority: str = "normal"):
        """Send voice notification (for external notifications)"""
        try:
            if self.voice_interface:
                # Interrupt current activity for high priority notifications
                if priority == "high":
                    # Could implement interruption logic here
                    pass
                
                # Synthesize and play notification
                await self._provide_voice_response(f"Notification: {message}")
                
                logger.info(f"Sent voice notification: '{message}' (priority: {priority})")
            
        except Exception as e:
            logger.error(f"Error sending voice notification: {e}")
    
    async def process_text_to_speech(self, text: str, voice_model: str = None) -> bytes:
        """Process text-to-speech request"""
        try:
            if self.voice_interface and self.voice_interface.tts:
                synthesis_result = await self.voice_interface.tts.synthesize_text(
                    text, 
                    voice_model=voice_model
                )
                
                audio_bytes = await self.voice_interface.tts.get_audio_bytes(
                    synthesis_result, 
                    format="wav"
                )
                
                return audio_bytes
            
            return None
            
        except Exception as e:
            logger.error(f"Error in text-to-speech processing: {e}")
            return None
    
    async def process_speech_to_text(self, audio_data: bytes) -> Optional[str]:
        """Process speech-to-text request"""
        try:
            if self.voice_interface and self.voice_interface.stt:
                transcription = await self.voice_interface.stt.transcribe_audio(audio_data)
                return transcription.text
            
            return None
            
        except Exception as e:
            logger.error(f"Error in speech-to-text processing: {e}")
            return None
    
    async def set_voice_configuration(self, config_updates: Dict[str, Any]):
        """Update voice configuration"""
        try:
            if 'wake_word' in config_updates and self.voice_interface:
                await self.voice_interface.set_wake_word(config_updates['wake_word'])
            
            if 'voice_model' in config_updates and self.voice_interface:
                await self.voice_interface.set_voice_model(config_updates['voice_model'])
            
            if 'auto_response' in config_updates:
                self.auto_response_enabled = config_updates['auto_response']
            
            logger.info(f"Updated voice configuration: {config_updates}")
            
        except Exception as e:
            logger.error(f"Error updating voice configuration: {e}")
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get comprehensive voice system status"""
        try:
            status = {
                "integration_stats": self.stats,
                "auto_response_enabled": self.auto_response_enabled,
                "active_session": self.active_voice_session is not None
            }
            
            if self.voice_interface:
                status["voice_interface"] = self.voice_interface.get_stats()
            
            if self.wake_word_detector:
                status["wake_word_detector"] = self.wake_word_detector.get_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting voice status: {e}")
            return {"error": str(e)}
    
    async def train_wake_word(self, wake_word: str, audio_samples: List[bytes]):
        """Train custom wake word"""
        try:
            if self.wake_word_detector:
                # Convert audio samples to numpy arrays
                import numpy as np
                import soundfile as sf
                import io
                
                audio_arrays = []
                for sample in audio_samples:
                    audio_array, sample_rate = sf.read(io.BytesIO(sample))
                    
                    # Resample if needed
                    if sample_rate != 16000:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    
                    audio_arrays.append(audio_array)
                
                # Train the wake word
                await self.wake_word_detector.train_wake_word(wake_word, audio_arrays)
                
                logger.info(f"Successfully trained wake word: '{wake_word}'")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error training wake word: {e}")
            return False
    
    async def cleanup(self):
        """Clean up voice-agent integration"""
        try:
            logger.info("Cleaning up voice-agent integration...")
            
            if self.voice_interface:
                await self.voice_interface.shutdown()
            
            if self.wake_word_detector:
                await self.wake_word_detector.cleanup()
            
            self.pending_responses.clear()
            
            logger.info("Voice-agent integration cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during voice-agent cleanup: {e}")

# Factory function for creating voice-agent bridge
async def create_voice_agent_bridge(orchestrator, config: Dict[str, Any]) -> VoiceAgentBridge:
    """Create and initialize voice-agent bridge"""
    bridge = VoiceAgentBridge(orchestrator, config)
    await bridge.initialize()
    return bridge