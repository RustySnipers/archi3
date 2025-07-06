"""
Multi-Modal Processing System for Archie
Handles voice, image, video, text, and other data modalities
"""

import os
import io
import logging
import asyncio
import json
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of data modalities"""
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SENSOR = "sensor"
    DOCUMENT = "document"
    STRUCTURED = "structured"

class ProcessingState(Enum):
    """Processing states"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

@dataclass
class ModalInput:
    """Multi-modal input structure"""
    id: str
    modality: ModalityType
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    source: str
    format: str
    size_bytes: int
    confidence: float = 1.0

@dataclass
class ProcessingResult:
    """Result from multi-modal processing"""
    input_id: str
    modality: ModalityType
    state: ProcessingState
    extracted_features: Dict[str, Any]
    interpreted_content: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    cached: bool = False

@dataclass
class MultiModalContext:
    """Context combining multiple modalities"""
    id: str
    inputs: List[ModalInput]
    results: List[ProcessingResult]
    combined_interpretation: str
    confidence: float
    created_at: datetime
    processing_time: float

class ImageProcessor:
    """Handles image processing and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        self.max_image_size = config.get('max_image_size', 1920)
        
    async def process_image(self, image_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process image and extract features"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Basic image properties
            features = {
                "format": image.format,
                "size": image.size,
                "mode": image.mode,
                "has_transparency": image.mode in ('RGBA', 'LA'),
                "estimated_colors": len(image.getcolors(maxcolors=256)) if image.getcolors(maxcolors=256) else 256
            }
            
            # Resize if too large
            if max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                features["resized"] = True
                features["original_size"] = features["size"]
                features["size"] = new_size
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract visual features
            visual_features = await self._extract_visual_features(image)
            features.update(visual_features)
            
            # Analyze content
            content_analysis = await self._analyze_image_content(image)
            features.update(content_analysis)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    async def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features from image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Color analysis
            mean_color = np.mean(img_array, axis=(0, 1))
            dominant_color = self._get_dominant_color(img_array)
            
            # Brightness and contrast
            brightness = np.mean(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
            contrast = np.std(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
            
            # Edge detection for complexity estimation
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                "mean_color": mean_color.tolist(),
                "dominant_color": dominant_color,
                "brightness": float(brightness),
                "contrast": float(contrast),
                "edge_density": float(edge_density),
                "complexity": "high" if edge_density > 0.1 else "medium" if edge_density > 0.05 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
    
    def _get_dominant_color(self, img_array: np.ndarray) -> List[int]:
        """Get dominant color from image"""
        try:
            # Reshape to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Use k-means clustering to find dominant color
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(pixels)
            
            return kmeans.cluster_centers_[0].astype(int).tolist()
            
        except Exception as e:
            logger.error(f"Error getting dominant color: {e}")
            return [128, 128, 128]  # Default gray
    
    async def _analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image content (objects, text, etc.)"""
        try:
            content = {
                "detected_objects": [],
                "text_regions": [],
                "scene_type": "unknown",
                "estimated_content": ""
            }
            
            # Simple heuristic analysis
            img_array = np.array(image)
            
            # Detect if image is predominantly text
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Simple text detection using edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.15:
                content["scene_type"] = "text_document"
                content["estimated_content"] = "Document or text-heavy image"
            elif edge_density > 0.08:
                content["scene_type"] = "complex_scene"
                content["estimated_content"] = "Complex scene with multiple objects"
            else:
                content["scene_type"] = "simple_scene"
                content["estimated_content"] = "Simple scene or portrait"
            
            # Detect potential faces using simple skin color detection
            skin_pixels = self._detect_skin_pixels(img_array)
            if skin_pixels > 0.05:  # More than 5% skin-colored pixels
                content["detected_objects"].append("potential_person")
            
            return content
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {e}")
            return {"error": str(e)}
    
    def _detect_skin_pixels(self, img_array: np.ndarray) -> float:
        """Simple skin color detection"""
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Define skin color range in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Calculate percentage of skin pixels
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.shape[0] * skin_mask.shape[1])
            
            return skin_ratio
            
        except Exception as e:
            logger.error(f"Error detecting skin pixels: {e}")
            return 0.0

class VideoProcessor:
    """Handles video processing and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        self.max_duration = config.get('max_video_duration', 300)  # 5 minutes
        self.frame_sample_rate = config.get('frame_sample_rate', 30)  # Every 30 frames
        
    async def process_video(self, video_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process video and extract features"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            features = {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "resolution": (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            }
            
            # Check duration limit
            if duration > self.max_duration:
                features["warning"] = f"Video duration ({duration:.1f}s) exceeds limit ({self.max_duration}s)"
                return features
            
            # Sample frames for analysis
            sampled_frames = await self._sample_frames(cap, frame_count)
            
            # Analyze frames
            frame_analysis = await self._analyze_frames(sampled_frames)
            features.update(frame_analysis)
            
            cap.release()
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {"error": str(e)}
    
    async def _sample_frames(self, cap: cv2.VideoCapture, frame_count: int) -> List[np.ndarray]:
        """Sample frames from video"""
        try:
            frames = []
            sample_interval = max(1, frame_count // 10)  # Sample up to 10 frames
            
            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                
                if len(frames) >= 10:  # Limit to 10 frames
                    break
            
            return frames
            
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
            return []
    
    async def _analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze sampled frames"""
        try:
            if not frames:
                return {"frame_analysis": "No frames to analyze"}
            
            analysis = {
                "sampled_frames": len(frames),
                "scene_changes": 0,
                "motion_level": "unknown",
                "color_variation": 0.0
            }
            
            # Analyze frame differences
            if len(frames) > 1:
                differences = []
                for i in range(1, len(frames)):
                    diff = cv2.absdiff(
                        cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                    )
                    differences.append(np.mean(diff))
                
                avg_difference = np.mean(differences)
                analysis["scene_changes"] = sum(1 for d in differences if d > avg_difference * 2)
                
                if avg_difference > 50:
                    analysis["motion_level"] = "high"
                elif avg_difference > 20:
                    analysis["motion_level"] = "medium"
                else:
                    analysis["motion_level"] = "low"
            
            # Color analysis across frames
            colors = []
            for frame in frames:
                mean_color = np.mean(frame, axis=(0, 1))
                colors.append(mean_color)
            
            if colors:
                color_std = np.std(colors, axis=0)
                analysis["color_variation"] = float(np.mean(color_std))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {e}")
            return {"error": str(e)}

class DocumentProcessor:
    """Handles document processing and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = ['.pdf', '.txt', '.docx', '.html', '.md']
        
    async def process_document(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process document and extract content"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt':
                return await self._process_text_file(file_path)
            elif file_ext == '.pdf':
                return await self._process_pdf_file(file_path)
            elif file_ext == '.html':
                return await self._process_html_file(file_path)
            else:
                return {"error": f"Unsupported document format: {file_ext}"}
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"error": str(e)}
    
    async def _process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "content": content,
                "word_count": len(content.split()),
                "character_count": len(content),
                "line_count": len(content.splitlines()),
                "encoding": "utf-8"
            }
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            return {"error": str(e)}
    
    async def _process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file"""
        try:
            # This would require PyPDF2 or similar library
            # For now, return basic info
            file_size = os.path.getsize(file_path)
            
            return {
                "format": "pdf",
                "file_size": file_size,
                "content": "PDF processing not fully implemented",
                "pages": "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"error": str(e)}
    
    async def _process_html_file(self, file_path: str) -> Dict[str, Any]:
        """Process HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic HTML analysis
            import re
            
            # Extract text content (simple approach)
            text_content = re.sub(r'<[^>]+>', '', content)
            
            # Count elements
            tags = re.findall(r'<(\w+)', content)
            
            return {
                "content": text_content,
                "html_content": content,
                "word_count": len(text_content.split()),
                "tag_count": len(tags),
                "unique_tags": len(set(tags)),
                "has_images": 'img' in tags,
                "has_links": 'a' in tags
            }
            
        except Exception as e:
            logger.error(f"Error processing HTML: {e}")
            return {"error": str(e)}

class MultiModalProcessor:
    """Main multi-modal processing system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize specialized processors
        self.image_processor = ImageProcessor(config.get('image', {}))
        self.video_processor = VideoProcessor(config.get('video', {}))
        self.document_processor = DocumentProcessor(config.get('document', {}))
        
        # Processing cache
        self.processing_cache = {}
        self.cache_enabled = config.get('cache_enabled', True)
        self.max_cache_size = config.get('max_cache_size', 1000)
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "by_modality": {modality.value: 0 for modality in ModalityType},
            "cache_hits": 0,
            "processing_time_total": 0.0,
            "average_processing_time": 0.0
        }
    
    async def process_input(self, modal_input: ModalInput) -> ProcessingResult:
        """Process multi-modal input"""
        try:
            start_time = datetime.now()
            
            # Check cache first
            if self.cache_enabled:
                cache_key = self._generate_cache_key(modal_input)
                if cache_key in self.processing_cache:
                    self.stats["cache_hits"] += 1
                    result = self.processing_cache[cache_key]
                    result.cached = True
                    return result
            
            # Process based on modality
            if modal_input.modality == ModalityType.IMAGE:
                features = await self.image_processor.process_image(
                    modal_input.data, modal_input.metadata
                )
                interpretation = self._interpret_image_features(features)
                confidence = self._calculate_image_confidence(features)
                
            elif modal_input.modality == ModalityType.VIDEO:
                features = await self.video_processor.process_video(
                    modal_input.data, modal_input.metadata
                )
                interpretation = self._interpret_video_features(features)
                confidence = self._calculate_video_confidence(features)
                
            elif modal_input.modality == ModalityType.DOCUMENT:
                features = await self.document_processor.process_document(
                    modal_input.data, modal_input.metadata
                )
                interpretation = self._interpret_document_features(features)
                confidence = self._calculate_document_confidence(features)
                
            elif modal_input.modality == ModalityType.TEXT:
                features = {"content": modal_input.data, "length": len(str(modal_input.data))}
                interpretation = str(modal_input.data)
                confidence = 1.0
                
            elif modal_input.modality == ModalityType.VOICE:
                # Voice processing would be handled by voice interface
                features = modal_input.metadata
                interpretation = modal_input.metadata.get("transcription", "Voice input")
                confidence = modal_input.metadata.get("confidence", 0.8)
                
            else:
                features = {"raw_data": "Unsupported modality"}
                interpretation = f"Unsupported modality: {modal_input.modality.value}"
                confidence = 0.0
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ProcessingResult(
                input_id=modal_input.id,
                modality=modal_input.modality,
                state=ProcessingState.COMPLETED if "error" not in features else ProcessingState.FAILED,
                extracted_features=features,
                interpreted_content=interpretation,
                confidence=confidence,
                processing_time=processing_time,
                error_message=features.get("error"),
                cached=False
            )
            
            # Cache result
            if self.cache_enabled and result.state == ProcessingState.COMPLETED:
                self._cache_result(cache_key, result)
            
            # Update statistics
            self.stats["total_processed"] += 1
            self.stats["by_modality"][modal_input.modality.value] += 1
            self.stats["processing_time_total"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["processing_time_total"] / self.stats["total_processed"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input {modal_input.id}: {e}")
            return ProcessingResult(
                input_id=modal_input.id,
                modality=modal_input.modality,
                state=ProcessingState.FAILED,
                extracted_features={},
                interpreted_content="",
                confidence=0.0,
                processing_time=0.0,
                error_message=str(e)
            )
    
    async def process_multi_modal_context(self, inputs: List[ModalInput]) -> MultiModalContext:
        """Process multiple inputs together for combined context"""
        try:
            start_time = datetime.now()
            
            # Process each input
            results = []
            for modal_input in inputs:
                result = await self.process_input(modal_input)
                results.append(result)
            
            # Combine interpretations
            combined_interpretation = await self._combine_interpretations(results)
            
            # Calculate overall confidence
            confidences = [r.confidence for r in results if r.state == ProcessingState.COMPLETED]
            overall_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create context
            context = MultiModalContext(
                id=f"context_{len(inputs)}_{int(datetime.now().timestamp())}",
                inputs=inputs,
                results=results,
                combined_interpretation=combined_interpretation,
                confidence=overall_confidence,
                created_at=start_time,
                processing_time=processing_time
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Error processing multi-modal context: {e}")
            return MultiModalContext(
                id="error_context",
                inputs=inputs,
                results=[],
                combined_interpretation=f"Error: {str(e)}",
                confidence=0.0,
                created_at=datetime.now(),
                processing_time=0.0
            )
    
    def _interpret_image_features(self, features: Dict[str, Any]) -> str:
        """Interpret image features into human-readable description"""
        if "error" in features:
            return f"Image processing error: {features['error']}"
        
        description = []
        
        # Size and format
        if "size" in features:
            description.append(f"Image is {features['size'][0]}x{features['size'][1]} pixels")
        
        # Scene type
        if "scene_type" in features:
            scene_type = features["scene_type"]
            if scene_type == "text_document":
                description.append("appears to be a document or text")
            elif scene_type == "complex_scene":
                description.append("shows a complex scene with multiple elements")
            else:
                description.append("shows a simple scene")
        
        # Objects
        if "detected_objects" in features and features["detected_objects"]:
            objects = features["detected_objects"]
            if "potential_person" in objects:
                description.append("may contain a person")
        
        # Color information
        if "brightness" in features:
            brightness = features["brightness"]
            if brightness > 200:
                description.append("is very bright")
            elif brightness > 100:
                description.append("has moderate brightness")
            else:
                description.append("is quite dark")
        
        return "This image " + ", ".join(description) if description else "Image processed successfully"
    
    def _interpret_video_features(self, features: Dict[str, Any]) -> str:
        """Interpret video features into human-readable description"""
        if "error" in features:
            return f"Video processing error: {features['error']}"
        
        description = []
        
        # Duration
        if "duration" in features:
            duration = features["duration"]
            description.append(f"Video is {duration:.1f} seconds long")
        
        # Motion level
        if "motion_level" in features:
            motion = features["motion_level"]
            if motion == "high":
                description.append("with high motion/activity")
            elif motion == "medium":
                description.append("with moderate motion")
            else:
                description.append("with minimal motion")
        
        # Scene changes
        if "scene_changes" in features:
            changes = features["scene_changes"]
            if changes > 5:
                description.append("and multiple scene changes")
            elif changes > 0:
                description.append("and some scene changes")
        
        return " ".join(description) if description else "Video processed successfully"
    
    def _interpret_document_features(self, features: Dict[str, Any]) -> str:
        """Interpret document features into human-readable description"""
        if "error" in features:
            return f"Document processing error: {features['error']}"
        
        if "word_count" in features:
            word_count = features["word_count"]
            if word_count > 1000:
                length_desc = "long document"
            elif word_count > 100:
                length_desc = "medium-length document"
            else:
                length_desc = "short document"
            
            return f"This is a {length_desc} with {word_count} words"
        
        return "Document processed successfully"
    
    def _calculate_image_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for image processing"""
        if "error" in features:
            return 0.0
        
        confidence = 0.8  # Base confidence
        
        # Boost confidence for successful feature extraction
        if "scene_type" in features:
            confidence += 0.1
        
        if "detected_objects" in features and features["detected_objects"]:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_video_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for video processing"""
        if "error" in features:
            return 0.0
        
        confidence = 0.7  # Base confidence (lower than image)
        
        if "motion_level" in features and features["motion_level"] != "unknown":
            confidence += 0.2
        
        if "sampled_frames" in features and features["sampled_frames"] > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_document_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence score for document processing"""
        if "error" in features:
            return 0.0
        
        if "content" in features and features["content"]:
            return 0.9  # High confidence for successful text extraction
        
        return 0.5
    
    async def _combine_interpretations(self, results: List[ProcessingResult]) -> str:
        """Combine multiple modal interpretations"""
        try:
            successful_results = [r for r in results if r.state == ProcessingState.COMPLETED]
            
            if not successful_results:
                return "No successful processing results to combine"
            
            # Group by modality
            by_modality = {}
            for result in successful_results:
                modality = result.modality.value
                if modality not in by_modality:
                    by_modality[modality] = []
                by_modality[modality].append(result.interpreted_content)
            
            # Create combined description
            combined = []
            
            for modality, interpretations in by_modality.items():
                if len(interpretations) == 1:
                    combined.append(f"{modality.title()}: {interpretations[0]}")
                else:
                    combined.append(f"{modality.title()}: {len(interpretations)} items processed")
            
            return "; ".join(combined)
            
        except Exception as e:
            logger.error(f"Error combining interpretations: {e}")
            return "Error combining multi-modal interpretations"
    
    def _generate_cache_key(self, modal_input: ModalInput) -> str:
        """Generate cache key for input"""
        try:
            # Create hash of input data and metadata
            import hashlib
            
            # For large data, use size and metadata
            if modal_input.size_bytes > 1024 * 1024:  # 1MB
                key_data = f"{modal_input.modality.value}:{modal_input.size_bytes}:{modal_input.format}"
            else:
                key_data = f"{modal_input.modality.value}:{str(modal_input.data)[:1000]}"
            
            return hashlib.md5(key_data.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"cache_error_{modal_input.id}"
    
    def _cache_result(self, cache_key: str, result: ProcessingResult):
        """Cache processing result"""
        try:
            # Remove oldest entries if cache is full
            if len(self.processing_cache) >= self.max_cache_size:
                # Remove 20% of oldest entries
                remove_count = self.max_cache_size // 5
                for _ in range(remove_count):
                    if self.processing_cache:
                        oldest_key = next(iter(self.processing_cache))
                        del self.processing_cache[oldest_key]
            
            self.processing_cache[cache_key] = result
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            "cache_size": len(self.processing_cache),
            "cache_enabled": self.cache_enabled
        }
    
    def clear_cache(self):
        """Clear processing cache"""
        self.processing_cache.clear()
        logger.info("Multi-modal processing cache cleared")

# Factory function
async def create_multimodal_processor(config: Dict[str, Any]) -> MultiModalProcessor:
    """Create and initialize multi-modal processor"""
    processor = MultiModalProcessor(config.get('multimodal', {}))
    return processor

# Utility functions
def create_modal_input_from_file(file_path: str, source: str = "file") -> Optional[ModalInput]:
    """Create ModalInput from file path"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        # Determine modality from file extension
        ext = path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            modality = ModalityType.IMAGE
            with open(file_path, 'rb') as f:
                data = f.read()
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            modality = ModalityType.VIDEO
            data = str(path.absolute())  # Store path for video
        elif ext in ['.pdf', '.txt', '.docx', '.html', '.md']:
            modality = ModalityType.DOCUMENT
            data = str(path.absolute())  # Store path for document
        elif ext in ['.wav', '.mp3', '.flac', '.ogg']:
            modality = ModalityType.AUDIO
            data = str(path.absolute())  # Store path for audio
        else:
            modality = ModalityType.TEXT
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
        
        return ModalInput(
            id=f"file_{path.stem}_{int(datetime.now().timestamp())}",
            modality=modality,
            data=data,
            metadata={
                "filename": path.name,
                "file_path": str(path.absolute()),
                "file_size": path.stat().st_size,
                "modification_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            },
            timestamp=datetime.now(),
            source=source,
            format=ext,
            size_bytes=path.stat().st_size
        )
        
    except Exception as e:
        logger.error(f"Error creating modal input from file {file_path}: {e}")
        return None

def create_modal_input_from_text(text: str, source: str = "user") -> ModalInput:
    """Create ModalInput from text"""
    return ModalInput(
        id=f"text_{int(datetime.now().timestamp())}",
        modality=ModalityType.TEXT,
        data=text,
        metadata={},
        timestamp=datetime.now(),
        source=source,
        format="text",
        size_bytes=len(text.encode('utf-8'))
    )

def create_modal_input_from_voice(transcription: str, 
                                audio_data: bytes, 
                                confidence: float,
                                source: str = "voice") -> ModalInput:
    """Create ModalInput from voice transcription"""
    return ModalInput(
        id=f"voice_{int(datetime.now().timestamp())}",
        modality=ModalityType.VOICE,
        data=audio_data,
        metadata={
            "transcription": transcription,
            "confidence": confidence
        },
        timestamp=datetime.now(),
        source=source,
        format="audio",
        size_bytes=len(audio_data),
        confidence=confidence
    )