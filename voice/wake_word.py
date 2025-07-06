"""
Wake Word Detection System
Advanced wake word detection using multiple methods for accuracy
"""

import os
import logging
import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import queue
import json

# Audio processing and ML imports
import librosa
from scipy import signal
from scipy.spatial.distance import cosine
import soundfile as sf

logger = logging.getLogger(__name__)

@dataclass
class WakeWordTemplate:
    """Wake word template for matching"""
    word: str
    features: np.ndarray
    duration: float
    sample_rate: int
    confidence_threshold: float = 0.7
    created_at: datetime = None

@dataclass
class WakeWordDetection:
    """Wake word detection result"""
    word: str
    confidence: float
    timestamp: datetime
    audio_segment: np.ndarray
    detection_method: str
    processing_time: float

class WakeWordDetector:
    """Advanced wake word detection system"""
    
    def __init__(self, 
                 wake_words: List[str] = None,
                 models_path: str = "/opt/archie/data/models",
                 sample_rate: int = 16000):
        
        self.wake_words = wake_words or ["archie"]
        self.models_path = models_path
        self.sample_rate = sample_rate
        
        # Detection methods
        self.use_mfcc_matching = True
        self.use_spectral_matching = True
        self.use_template_matching = True
        self.use_neural_detection = False  # Placeholder for future neural models
        
        # Detection parameters
        self.detection_window_size = 2.0  # seconds
        self.hop_length = 0.1  # seconds
        self.confidence_threshold = 0.7
        self.confirmation_threshold = 0.8  # Higher threshold for confirmed detection
        
        # Audio preprocessing
        self.preemphasis_coeff = 0.97
        self.frame_length = int(0.025 * sample_rate)  # 25ms frames
        self.frame_step = int(0.010 * sample_rate)    # 10ms step
        
        # MFCC parameters
        self.n_mfcc = 13
        self.n_fft = 512
        self.hop_length_mfcc = 160  # 10ms at 16kHz
        self.n_mels = 40
        
        # Templates and models
        self.wake_word_templates = {}
        self.spectral_templates = {}
        self.background_noise_profile = None
        
        # Detection state
        self.detection_buffer = []
        self.last_detection_time = None
        self.detection_cooldown = 2.0  # seconds between detections
        
        # Statistics
        self.stats = {
            "total_detections": 0,
            "confirmed_detections": 0,
            "false_positives": 0,
            "processing_time_total": 0.0,
            "average_processing_time": 0.0,
            "templates_loaded": 0
        }
        
        # Callbacks
        self.callbacks = []
        
        # Ensure models directory exists
        os.makedirs(self.models_path, exist_ok=True)
    
    async def initialize(self):
        """Initialize wake word detection system"""
        try:
            logger.info("Initializing wake word detector...")
            
            # Load or create wake word templates
            await self._load_wake_word_templates()
            
            # Initialize background noise profiling
            self._initialize_noise_profiling()
            
            logger.info(f"Wake word detector initialized for words: {self.wake_words}")
            
        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")
            raise
    
    async def _load_wake_word_templates(self):
        """Load or create templates for wake words"""
        try:
            for wake_word in self.wake_words:
                template_file = os.path.join(self.models_path, f"wake_word_{wake_word}.json")
                
                if os.path.exists(template_file):
                    # Load existing template
                    await self._load_template_from_file(wake_word, template_file)
                else:
                    # Create default template (would typically be trained)
                    await self._create_default_template(wake_word)
                
                self.stats["templates_loaded"] += 1
            
            logger.info(f"Loaded {len(self.wake_word_templates)} wake word templates")
            
        except Exception as e:
            logger.error(f"Error loading wake word templates: {e}")
            raise
    
    async def _load_template_from_file(self, wake_word: str, template_file: str):
        """Load wake word template from file"""
        try:
            with open(template_file, 'r') as f:
                template_data = json.load(f)
            
            template = WakeWordTemplate(
                word=wake_word,
                features=np.array(template_data['features']),
                duration=template_data['duration'],
                sample_rate=template_data['sample_rate'],
                confidence_threshold=template_data.get('confidence_threshold', 0.7),
                created_at=datetime.fromisoformat(template_data['created_at'])
            )
            
            self.wake_word_templates[wake_word] = template
            logger.info(f"Loaded template for '{wake_word}' from {template_file}")
            
        except Exception as e:
            logger.error(f"Error loading template from {template_file}: {e}")
            # Fall back to creating default template
            await self._create_default_template(wake_word)
    
    async def _create_default_template(self, wake_word: str):
        """Create a default template for wake word"""
        try:
            logger.info(f"Creating default template for '{wake_word}'")
            
            # Generate synthetic features based on phonetic properties
            # In production, this would be trained from real audio samples
            
            # Estimate duration based on syllables (rough approximation)
            syllable_count = max(1, len(wake_word) // 2)
            duration = syllable_count * 0.3  # ~300ms per syllable
            
            # Create synthetic MFCC features
            n_frames = int(duration * self.sample_rate / self.hop_length_mfcc)
            features = self._generate_synthetic_mfcc(wake_word, n_frames)
            
            template = WakeWordTemplate(
                word=wake_word,
                features=features,
                duration=duration,
                sample_rate=self.sample_rate,
                confidence_threshold=0.6,  # Lower threshold for synthetic templates
                created_at=datetime.now()
            )
            
            self.wake_word_templates[wake_word] = template
            
            # Save template to file
            await self._save_template_to_file(template)
            
        except Exception as e:
            logger.error(f"Error creating default template for '{wake_word}': {e}")
    
    def _generate_synthetic_mfcc(self, wake_word: str, n_frames: int) -> np.ndarray:
        """Generate synthetic MFCC features for wake word"""
        try:
            # Very basic synthetic feature generation
            # In production, use trained templates or phonetic models
            
            # Create varying MFCC coefficients based on word characteristics
            features = np.random.normal(0, 1, (self.n_mfcc, n_frames))
            
            # Add some structure based on word length and vowels
            vowels = sum(1 for c in wake_word.lower() if c in 'aeiou')
            consonants = len(wake_word) - vowels
            
            # Emphasize certain coefficients based on phonetic content
            if vowels > consonants:
                features[1:4] *= 1.5  # Emphasize formant-related coefficients
            else:
                features[6:9] *= 1.5  # Emphasize higher frequency coefficients
            
            # Apply smoothing
            for i in range(features.shape[0]):
                features[i] = signal.savgol_filter(features[i], 
                                                 min(5, n_frames), 
                                                 min(3, n_frames-1))
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating synthetic MFCC: {e}")
            return np.zeros((self.n_mfcc, n_frames))
    
    async def _save_template_to_file(self, template: WakeWordTemplate):
        """Save wake word template to file"""
        try:
            template_file = os.path.join(self.models_path, f"wake_word_{template.word}.json")
            
            template_data = {
                'word': template.word,
                'features': template.features.tolist(),
                'duration': template.duration,
                'sample_rate': template.sample_rate,
                'confidence_threshold': template.confidence_threshold,
                'created_at': template.created_at.isoformat()
            }
            
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            logger.info(f"Saved template for '{template.word}' to {template_file}")
            
        except Exception as e:
            logger.error(f"Error saving template: {e}")
    
    def _initialize_noise_profiling(self):
        """Initialize background noise profiling"""
        try:
            # Initialize with default noise profile
            self.background_noise_profile = np.ones(self.n_mfcc) * 0.1
            logger.info("Background noise profiling initialized")
            
        except Exception as e:
            logger.error(f"Error initializing noise profiling: {e}")
    
    async def detect_wake_word(self, audio_chunk: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word in audio chunk"""
        try:
            start_time = datetime.now()
            
            # Check detection cooldown
            if (self.last_detection_time and 
                (start_time - self.last_detection_time).total_seconds() < self.detection_cooldown):
                return None
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_chunk)
            
            # Add to detection buffer
            self.detection_buffer.append(processed_audio)
            
            # Maintain buffer size (keep last N seconds)
            max_buffer_length = int(self.detection_window_size * self.sample_rate)
            if len(self.detection_buffer) > max_buffer_length:
                self.detection_buffer = self.detection_buffer[-max_buffer_length:]
            
            # Only detect if we have enough audio
            current_buffer = np.concatenate(self.detection_buffer)
            if len(current_buffer) < int(0.5 * self.sample_rate):  # Need at least 0.5 seconds
                return None
            
            # Run detection methods
            detections = []
            
            if self.use_mfcc_matching:
                mfcc_detection = await self._detect_with_mfcc(current_buffer)
                if mfcc_detection:
                    detections.append(mfcc_detection)
            
            if self.use_spectral_matching:
                spectral_detection = await self._detect_with_spectral_matching(current_buffer)
                if spectral_detection:
                    detections.append(spectral_detection)
            
            if self.use_template_matching:
                template_detection = await self._detect_with_template_matching(current_buffer)
                if template_detection:
                    detections.append(template_detection)
            
            # Combine detection results
            best_detection = self._combine_detections(detections)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if best_detection:
                best_detection.processing_time = processing_time
                self.last_detection_time = start_time
                self.stats["total_detections"] += 1
                
                # Check if this meets confirmation threshold
                if best_detection.confidence >= self.confirmation_threshold:
                    self.stats["confirmed_detections"] += 1
                    logger.info(f"Wake word '{best_detection.word}' detected with confidence {best_detection.confidence:.3f}")
                    
                    # Notify callbacks
                    await self._notify_callbacks(best_detection)
                
                return best_detection
            
            # Update processing time stats
            self.stats["processing_time_total"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["processing_time_total"] / max(1, self.stats["total_detections"])
            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in wake word detection: {e}")
            return None
    
    def _preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for wake word detection"""
        try:
            # Apply preemphasis
            if self.preemphasis_coeff > 0:
                audio = np.append(audio[0], audio[1:] - self.preemphasis_coeff * audio[:-1])
            
            # Normalize amplitude
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio
    
    async def _detect_with_mfcc(self, audio: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using MFCC feature matching"""
        try:
            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length_mfcc,
                n_mels=self.n_mels
            )
            
            # Normalize features
            mfcc_features = (mfcc_features - np.mean(mfcc_features, axis=1, keepdims=True)) / \
                           (np.std(mfcc_features, axis=1, keepdims=True) + 1e-8)
            
            best_match = None
            best_confidence = 0.0
            
            # Compare with each wake word template
            for wake_word, template in self.wake_word_templates.items():
                confidence = self._compare_mfcc_features(mfcc_features, template.features)
                
                if confidence > template.confidence_threshold and confidence > best_confidence:
                    best_confidence = confidence
                    best_match = wake_word
            
            if best_match:
                return WakeWordDetection(
                    word=best_match,
                    confidence=best_confidence,
                    timestamp=datetime.now(),
                    audio_segment=audio,
                    detection_method="mfcc",
                    processing_time=0.0  # Will be set by caller
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MFCC detection: {e}")
            return None
    
    def _compare_mfcc_features(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compare two MFCC feature sets using DTW-like alignment"""
        try:
            # Simple correlation-based matching
            # In production, would use Dynamic Time Warping (DTW)
            
            # Resize features to same length using interpolation
            min_length = min(features1.shape[1], features2.shape[1])
            max_length = max(features1.shape[1], features2.shape[1])
            
            if features1.shape[1] != features2.shape[1]:
                if features1.shape[1] > features2.shape[1]:
                    # Resize features2 to match features1
                    indices = np.linspace(0, features2.shape[1] - 1, features1.shape[1])
                    features2_resized = np.array([
                        np.interp(indices, np.arange(features2.shape[1]), features2[i])
                        for i in range(features2.shape[0])
                    ])
                    features2 = features2_resized
                else:
                    # Resize features1 to match features2
                    indices = np.linspace(0, features1.shape[1] - 1, features2.shape[1])
                    features1_resized = np.array([
                        np.interp(indices, np.arange(features1.shape[1]), features1[i])
                        for i in range(features1.shape[0])
                    ])
                    features1 = features1_resized
            
            # Calculate cosine similarity for each MFCC coefficient
            similarities = []
            for i in range(features1.shape[0]):
                if np.std(features1[i]) > 0 and np.std(features2[i]) > 0:
                    correlation = np.corrcoef(features1[i], features2[i])[0, 1]
                    if not np.isnan(correlation):
                        similarities.append(abs(correlation))
            
            # Return average similarity
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error comparing MFCC features: {e}")
            return 0.0
    
    async def _detect_with_spectral_matching(self, audio: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using spectral pattern matching"""
        try:
            # Compute spectrogram
            n_fft = 512
            hop_length = 160
            
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(stft)
            
            # Convert to log scale
            log_magnitude = np.log(magnitude + 1e-8)
            
            # Simple spectral matching (placeholder)
            # In production, would use more sophisticated spectral templates
            
            # Look for energy patterns typical of wake words
            energy_profile = np.mean(log_magnitude, axis=0)
            
            # Check for energy peaks that might indicate speech
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(energy_profile, height=np.mean(energy_profile) + np.std(energy_profile))
            
            if len(peaks) >= 2:  # Need at least some energy variation
                confidence = min(1.0, len(peaks) / 5.0)  # Normalize by expected peak count
                
                if confidence > 0.4:  # Lower threshold for spectral method
                    return WakeWordDetection(
                        word=self.wake_words[0],  # Default to first wake word
                        confidence=confidence,
                        timestamp=datetime.now(),
                        audio_segment=audio,
                        detection_method="spectral",
                        processing_time=0.0
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in spectral detection: {e}")
            return None
    
    async def _detect_with_template_matching(self, audio: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using cross-correlation template matching"""
        try:
            # Placeholder for template matching
            # Would use stored audio templates for cross-correlation
            
            # For now, just check energy and duration patterns
            rms_energy = np.sqrt(np.mean(audio ** 2))
            duration = len(audio) / self.sample_rate
            
            # Simple heuristic: check if energy and duration are reasonable for wake word
            for wake_word, template in self.wake_word_templates.items():
                expected_duration = template.duration
                duration_match = abs(duration - expected_duration) / expected_duration < 0.5
                energy_match = rms_energy > 0.01  # Minimum energy threshold
                
                if duration_match and energy_match:
                    confidence = 0.5 * (1.0 - abs(duration - expected_duration) / expected_duration)
                    
                    if confidence > 0.3:
                        return WakeWordDetection(
                            word=wake_word,
                            confidence=confidence,
                            timestamp=datetime.now(),
                            audio_segment=audio,
                            detection_method="template",
                            processing_time=0.0
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in template detection: {e}")
            return None
    
    def _combine_detections(self, detections: List[WakeWordDetection]) -> Optional[WakeWordDetection]:
        """Combine multiple detection results"""
        try:
            if not detections:
                return None
            
            # Group detections by word
            word_detections = {}
            for detection in detections:
                if detection.word not in word_detections:
                    word_detections[detection.word] = []
                word_detections[detection.word].append(detection)
            
            # Find best combined detection
            best_detection = None
            best_combined_confidence = 0.0
            
            for word, word_detections_list in word_detections.items():
                # Combine confidences (weighted average)
                method_weights = {
                    "mfcc": 0.5,
                    "spectral": 0.3,
                    "template": 0.2
                }
                
                total_weight = 0.0
                weighted_confidence = 0.0
                
                for detection in word_detections_list:
                    weight = method_weights.get(detection.detection_method, 0.1)
                    weighted_confidence += detection.confidence * weight
                    total_weight += weight
                
                if total_weight > 0:
                    combined_confidence = weighted_confidence / total_weight
                    
                    # Boost confidence if multiple methods agree
                    if len(word_detections_list) > 1:
                        combined_confidence *= 1.2
                    
                    combined_confidence = min(1.0, combined_confidence)
                    
                    if combined_confidence > best_combined_confidence:
                        best_combined_confidence = combined_confidence
                        
                        # Use the detection with highest individual confidence
                        best_individual = max(word_detections_list, key=lambda d: d.confidence)
                        best_detection = WakeWordDetection(
                            word=word,
                            confidence=combined_confidence,
                            timestamp=best_individual.timestamp,
                            audio_segment=best_individual.audio_segment,
                            detection_method="combined",
                            processing_time=0.0
                        )
            
            return best_detection
            
        except Exception as e:
            logger.error(f"Error combining detections: {e}")
            return detections[0] if detections else None
    
    async def train_wake_word(self, wake_word: str, audio_samples: List[np.ndarray]):
        """Train wake word template from audio samples"""
        try:
            logger.info(f"Training wake word template for '{wake_word}' with {len(audio_samples)} samples")
            
            # Extract features from all samples
            all_features = []
            durations = []
            
            for sample in audio_samples:
                # Preprocess
                processed = self._preprocess_audio(sample)
                durations.append(len(processed) / self.sample_rate)
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(
                    y=processed,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length_mfcc,
                    n_mels=self.n_mels
                )
                
                # Normalize
                mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
                       (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
                
                all_features.append(mfcc)
            
            # Create average template
            # Align all features to same length (using median length)
            median_length = int(np.median([f.shape[1] for f in all_features]))
            
            aligned_features = []
            for features in all_features:
                if features.shape[1] != median_length:
                    # Resize using interpolation
                    indices = np.linspace(0, features.shape[1] - 1, median_length)
                    resized = np.array([
                        np.interp(indices, np.arange(features.shape[1]), features[i])
                        for i in range(features.shape[0])
                    ])
                    aligned_features.append(resized)
                else:
                    aligned_features.append(features)
            
            # Average features
            template_features = np.mean(aligned_features, axis=0)
            template_duration = np.mean(durations)
            
            # Create template
            template = WakeWordTemplate(
                word=wake_word,
                features=template_features,
                duration=template_duration,
                sample_rate=self.sample_rate,
                confidence_threshold=0.7,
                created_at=datetime.now()
            )
            
            # Store template
            self.wake_word_templates[wake_word] = template
            
            # Save to file
            await self._save_template_to_file(template)
            
            logger.info(f"Successfully trained wake word template for '{wake_word}'")
            
        except Exception as e:
            logger.error(f"Error training wake word: {e}")
            raise
    
    async def _notify_callbacks(self, detection: WakeWordDetection):
        """Notify registered callbacks of wake word detection"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(detection)
                else:
                    callback(detection)
            except Exception as e:
                logger.error(f"Error in wake word callback: {e}")
    
    def register_callback(self, callback: Callable):
        """Register callback for wake word detections"""
        self.callbacks.append(callback)
        logger.info("Registered wake word detection callback")
    
    def unregister_callback(self, callback: Callable):
        """Unregister callback"""
        try:
            self.callbacks.remove(callback)
            logger.info("Unregistered wake word detection callback")
        except ValueError:
            logger.warning("Callback not found for unregistration")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wake word detection statistics"""
        return {
            **self.stats,
            "wake_words": self.wake_words,
            "templates_loaded": len(self.wake_word_templates),
            "detection_window_size": self.detection_window_size,
            "confidence_threshold": self.confidence_threshold,
            "last_detection": self.last_detection_time.isoformat() if self.last_detection_time else None
        }
    
    async def cleanup(self):
        """Clean up wake word detector resources"""
        try:
            self.detection_buffer.clear()
            self.callbacks.clear()
            logger.info("Wake word detector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during wake word detector cleanup: {e}")

# Factory function for creating wake word detector
async def create_wake_word_detector(config: Dict[str, Any]) -> WakeWordDetector:
    """Create and initialize wake word detector"""
    voice_config = config.get('voice', {})
    wake_word_config = voice_config.get('wake_word', {})
    
    wake_words = [wake_word_config.get('word', voice_config.get('wake_word', 'archie'))]
    models_path = config.get('data_path', '/opt/archie/data') + '/models'
    sample_rate = voice_config.get('audio', {}).get('sample_rate', 16000)
    
    detector = WakeWordDetector(
        wake_words=wake_words,
        models_path=models_path,
        sample_rate=sample_rate
    )
    
    await detector.initialize()
    return detector