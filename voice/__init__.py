# Voice Interface Module
# Speech-to-text and text-to-speech functionality

from .stt import WhisperSTT, TranscriptionResult, create_whisper_stt
from .tts import PiperTTS, SynthesisResult, create_piper_tts
from .audio_processor import AudioProcessor, AudioChunk, AudioConfig, create_audio_processor
from .voice_interface import VoiceInterface, VoiceCommand, VoiceResponse, create_voice_interface
from .wake_word import WakeWordDetector, WakeWordDetection, create_wake_word_detector
from .voice_agent_integration import VoiceAgentBridge, create_voice_agent_bridge

__all__ = [
    # STT
    'WhisperSTT', 'TranscriptionResult', 'create_whisper_stt',
    # TTS
    'PiperTTS', 'SynthesisResult', 'create_piper_tts',
    # Audio Processing
    'AudioProcessor', 'AudioChunk', 'AudioConfig', 'create_audio_processor',
    # Voice Interface
    'VoiceInterface', 'VoiceCommand', 'VoiceResponse', 'create_voice_interface',
    # Wake Word
    'WakeWordDetector', 'WakeWordDetection', 'create_wake_word_detector',
    # Integration
    'VoiceAgentBridge', 'create_voice_agent_bridge'
]