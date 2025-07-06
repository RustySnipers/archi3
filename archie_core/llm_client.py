"""
LocalAI client for Archie
Handles communication with LocalAI LLM backend
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    size: str
    description: str
    loaded: bool = False
    last_used: Optional[datetime] = None

@dataclass
class CompletionRequest:
    """Completion request structure"""
    model: str
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None

@dataclass
class CompletionResponse:
    """Completion response structure"""
    id: str
    model: str
    content: str
    usage: Dict[str, int]
    finish_reason: str
    tool_calls: Optional[List[Dict]] = None
    created: Optional[datetime] = None

class LocalAIClient:
    """LocalAI client for LLM operations"""
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.models = {}
        self.default_model = "gpt-4"
        self.model_cache = {}
        
        # Configure session headers
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Archie/1.0.0"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the client session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for model loading
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers
            )
            
            # Load available models
            await self._load_available_models()
            logger.info(f"LocalAI client initialized with {len(self.models)} models")
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _load_available_models(self):
        """Load list of available models from LocalAI"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    for model_data in data.get('data', []):
                        model_info = ModelInfo(
                            name=model_data.get('id', ''),
                            size=model_data.get('object', 'unknown'),
                            description=model_data.get('description', ''),
                            loaded=True
                        )
                        self.models[model_info.name] = model_info
                else:
                    logger.warning(f"Failed to load models: HTTP {response.status}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def list_models(self) -> List[ModelInfo]:
        """List all available models"""
        if not self.models:
            await self._load_available_models()
        return list(self.models.values())
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            # Check if model is already loaded
            if model_name in self.models and self.models[model_name].loaded:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Attempt to load model by making a test request
            test_request = CompletionRequest(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            
            response = await self.complete(test_request)
            if response:
                if model_name not in self.models:
                    self.models[model_name] = ModelInfo(
                        name=model_name,
                        size="unknown",
                        description="Loaded dynamically",
                        loaded=True
                    )
                else:
                    self.models[model_name].loaded = True
                    
                logger.info(f"Model {model_name} loaded successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
        
        return False
    
    async def complete(self, request: CompletionRequest) -> Optional[CompletionResponse]:
        """Complete a chat completion request"""
        if not self.session:
            await self.initialize()
        
        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream
            }
            
            # Add tools if provided
            if request.tools:
                payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
            
            # Make request
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse response
                    choice = data.get('choices', [{}])[0]
                    message = choice.get('message', {})
                    
                    completion_response = CompletionResponse(
                        id=data.get('id', ''),
                        model=data.get('model', request.model),
                        content=message.get('content', ''),
                        usage=data.get('usage', {}),
                        finish_reason=choice.get('finish_reason', 'unknown'),
                        tool_calls=message.get('tool_calls'),
                        created=datetime.now()
                    )
                    
                    # Update model cache
                    if request.model in self.models:
                        self.models[request.model].last_used = datetime.now()
                    
                    return completion_response
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Completion failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in completion: {e}")
        
        return None
    
    async def complete_streaming(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        """Stream completion response"""
        if not self.session:
            await self.initialize()
        
        try:
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }
            
            # Add tools if provided
            if request.tools:
                payload["tools"] = request.tools
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice
            
            # Make streaming request
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                choice = data.get('choices', [{}])[0]
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    yield content
                                    
                            except json.JSONDecodeError:
                                continue
                                
                else:
                    error_text = await response.text()
                    logger.error(f"Streaming completion failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error in streaming completion: {e}")
    
    async def generate_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> Optional[List[List[float]]]:
        """Generate embeddings for texts"""
        if not self.session:
            await self.initialize()
        
        try:
            payload = {
                "model": model,
                "input": texts
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    embeddings = []
                    
                    for item in data.get('data', []):
                        embeddings.append(item.get('embedding', []))
                    
                    return embeddings
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Embedding generation failed: HTTP {response.status} - {error_text}")
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LocalAI health status"""
        if not self.session:
            await self.initialize()
        
        try:
            async with self.session.get(f"{self.base_url}/readyz") as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "models_loaded": len([m for m in self.models.values() if m.loaded]),
                        "total_models": len(self.models),
                        "last_check": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status}",
                        "last_check": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        if model_name in self.models:
            return self.models[model_name]
        
        # Try to load model info
        await self._load_available_models()
        return self.models.get(model_name)
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        try:
            # LocalAI doesn't have a direct unload endpoint
            # This would typically involve model management
            if model_name in self.models:
                self.models[model_name].loaded = False
                logger.info(f"Model {model_name} marked as unloaded")
                return True
                
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
        
        return False

# Factory function for creating client instances
async def create_localai_client(config: Dict[str, Any]) -> LocalAIClient:
    """Create and initialize LocalAI client"""
    base_url = config.get('localai', {}).get('url', 'http://localhost:8080')
    api_key = config.get('localai', {}).get('api_key')
    
    client = LocalAIClient(base_url=base_url, api_key=api_key)
    await client.initialize()
    return client