# Archie API Documentation

This document provides comprehensive API documentation for the Archie Personal AI Assistant system.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core API Endpoints](#core-api-endpoints)
4. [Agent APIs](#agent-apis)
5. [Voice Interface API](#voice-interface-api)
6. [Memory Management API](#memory-management-api)
7. [Automation API](#automation-api)
8. [Multi-Modal API](#multi-modal-api)
9. [Administration API](#administration-api)
10. [WebSocket APIs](#websocket-apis)
11. [Error Codes](#error-codes)
12. [Rate Limiting](#rate-limiting)
13. [SDKs and Examples](#sdks-and-examples)

## 🔍 Overview

### Base URL
```
Production: https://api.archie.yourdomain.com
Development: http://localhost:8080/api
```

### API Version
Current API version: `v1`

All API endpoints are prefixed with `/api/v1/`

### Response Format
All API responses follow this standard format:

```json
{
  "success": boolean,
  "data": object | array | null,
  "error": {
    "code": "string",
    "message": "string",
    "details": object
  } | null,
  "metadata": {
    "timestamp": "2025-07-06T12:00:00Z",
    "request_id": "uuid",
    "version": "v1"
  }
}
```

## 🔐 Authentication

### JWT Authentication

Archie uses JWT (JSON Web Tokens) for authentication.

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "expires_in": 3600,
    "token_type": "Bearer"
  }
}
```

#### Using Tokens
Include the JWT token in the Authorization header:

```http
Authorization: Bearer <access_token>
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "string"
}
```

### API Keys
For service-to-service communication, use API keys:

```http
X-API-Key: your_api_key_here
```

## 🎯 Core API Endpoints

### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 86400,
    "components": {
      "database": "healthy",
      "memory": "healthy",
      "voice": "healthy",
      "automation": "healthy"
    }
  }
}
```

### System Status
```http
GET /api/v1/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 62.1,
      "disk_usage": 34.8
    },
    "agents": {
      "voice_agent": "active",
      "automation_agent": "active",
      "memory_agent": "active",
      "communication_agent": "active"
    },
    "services": {
      "localai": "running",
      "chromadb": "running",
      "home_assistant": "connected"
    }
  }
}
```

## 🤖 Agent APIs

### Send Message to Agent
```http
POST /api/v1/agents/{agent_name}/message
Content-Type: application/json

{
  "message": "Turn on the living room lights",
  "context": {
    "user_id": "user123",
    "location": "home",
    "time": "2025-07-06T20:30:00Z"
  },
  "modal_inputs": [
    {
      "type": "text",
      "data": "additional context"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "I've turned on the living room lights for you.",
    "agent": "automation_agent",
    "processing_time": 1.23,
    "actions_taken": [
      {
        "type": "device_control",
        "entity_id": "light.living_room",
        "action": "turn_on"
      }
    ]
  }
}
```

### List Agents
```http
GET /api/v1/agents
```

### Agent Status
```http
GET /api/v1/agents/{agent_name}/status
```

### Agent Capabilities
```http
GET /api/v1/agents/{agent_name}/capabilities
```

## 🎤 Voice Interface API

### Voice Configuration
```http
GET /api/v1/voice/config
```

```http
PUT /api/v1/voice/config
Content-Type: application/json

{
  "wake_word": "archie",
  "stt_model": "whisper-base",
  "tts_model": "en_US-jenny-medium",
  "voice_detection_threshold": 0.6
}
```

### Start Listening
```http
POST /api/v1/voice/listen
Content-Type: application/json

{
  "duration": 30,
  "wake_word_required": true
}
```

### Stop Listening
```http
POST /api/v1/voice/stop
```

### Voice Command Processing
```http
POST /api/v1/voice/process
Content-Type: multipart/form-data

audio_file: <audio file>
format: "wav"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "transcription": "Turn on the bedroom lights",
    "confidence": 0.95,
    "language": "en",
    "processing_time": 2.1,
    "agent_response": "I've turned on the bedroom lights.",
    "audio_response": "base64_encoded_audio"
  }
}
```

### Text-to-Speech
```http
POST /api/v1/voice/synthesize
Content-Type: application/json

{
  "text": "Hello, how can I help you today?",
  "voice_model": "en_US-jenny-medium",
  "speed": 1.0
}
```

### Voice History
```http
GET /api/v1/voice/history?limit=50&offset=0
```

## 🧠 Memory Management API

### Store Memory
```http
POST /api/v1/memory
Content-Type: application/json

{
  "content": "User prefers morning briefings at 8 AM",
  "memory_type": "preference",
  "metadata": {
    "user_id": "user123",
    "category": "schedule"
  },
  "importance": 0.8,
  "tags": ["schedule", "preference", "morning"]
}
```

### Query Memories
```http
POST /api/v1/memory/query
Content-Type: application/json

{
  "query_text": "morning routine preferences",
  "memory_types": ["preference", "routine"],
  "limit": 10,
  "similarity_threshold": 0.7,
  "include_context": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "memories": [
      {
        "id": "mem_abc123",
        "content": "User prefers morning briefings at 8 AM",
        "similarity_score": 0.92,
        "importance": 0.8,
        "created_at": "2025-07-01T08:00:00Z",
        "context_memories": [
          {
            "id": "mem_def456",
            "content": "User wakes up at 7:30 AM on weekdays"
          }
        ]
      }
    ],
    "total_found": 5
  }
}
```

### Memory Statistics
```http
GET /api/v1/memory/stats
```

### Memory Clusters
```http
GET /api/v1/memory/clusters
```

### Memory Insights
```http
GET /api/v1/memory/insights
```

## 🔧 Automation API

### Create Automation Rule
```http
POST /api/v1/automation/rules
Content-Type: application/json

{
  "name": "Evening Light Routine",
  "description": "Turn on living room lights when motion detected after sunset",
  "natural_language": "when motion detected in living room after sunset turn on lights",
  "enabled": true,
  "user_context": {
    "user_id": "user123",
    "location": "home"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "rule_id": "rule_xyz789",
    "name": "Evening Light Routine",
    "rule_type": "conditional",
    "priority": "medium",
    "conditions": [
      {
        "type": "motion_detected",
        "entity_id": "binary_sensor.living_room_motion"
      },
      {
        "type": "time_condition",
        "condition": "after_sunset"
      }
    ],
    "actions": [
      {
        "type": "device_control",
        "entity_id": "light.living_room",
        "action": "turn_on"
      }
    ]
  }
}
```

### List Automation Rules
```http
GET /api/v1/automation/rules?limit=20&enabled=true
```

### Execute Rule
```http
POST /api/v1/automation/rules/{rule_id}/execute
Content-Type: application/json

{
  "context": {
    "trigger_source": "manual",
    "user_id": "user123"
  }
}
```

### Rule Statistics
```http
GET /api/v1/automation/rules/{rule_id}/stats
```

### Workflow Generation
```http
POST /api/v1/automation/workflows
Content-Type: application/json

{
  "description": "Send notification when door opens and nobody is home",
  "name": "Security Alert Workflow"
}
```

## 🎨 Multi-Modal API

### Process Image
```http
POST /api/v1/multimodal/image
Content-Type: multipart/form-data

image: <image file>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "input_id": "img_abc123",
    "extracted_features": {
      "size": [1920, 1080],
      "format": "JPEG",
      "scene_type": "indoor",
      "detected_objects": ["person", "furniture"],
      "dominant_color": [120, 150, 180],
      "brightness": 142.5
    },
    "interpreted_content": "Indoor scene with a person sitting on furniture in moderate lighting",
    "confidence": 0.87,
    "processing_time": 2.3
  }
}
```

### Process Video
```http
POST /api/v1/multimodal/video
Content-Type: multipart/form-data

video: <video file>
```

### Process Document
```http
POST /api/v1/multimodal/document
Content-Type: multipart/form-data

document: <document file>
```

### Multi-Modal Context
```http
POST /api/v1/multimodal/context
Content-Type: application/json

{
  "inputs": [
    {
      "type": "text",
      "data": "What do you see in this image?"
    },
    {
      "type": "image",
      "file_id": "img_abc123"
    }
  ]
}
```

## 🛠️ Administration API

### System Configuration
```http
GET /api/v1/admin/config
```

```http
PUT /api/v1/admin/config
Content-Type: application/json

{
  "logging_level": "INFO",
  "performance_optimization": "balanced",
  "cache_settings": {
    "enabled": true,
    "max_memory_mb": 1024
  }
}
```

### Performance Report
```http
GET /api/v1/admin/performance
```

### Cache Management
```http
POST /api/v1/admin/cache/clear
```

```http
GET /api/v1/admin/cache/stats
```

### User Management
```http
GET /api/v1/admin/users
```

```http
POST /api/v1/admin/users
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "role": "user"
}
```

### Backup Operations
```http
POST /api/v1/admin/backup
Content-Type: application/json

{
  "type": "full",
  "encryption": true
}
```

## 🔌 WebSocket APIs

### Real-time Events
```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws/events');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Event:', data);
};
```

**Event Types:**
- `agent_message` - Agent responses
- `automation_triggered` - Automation executions
- `voice_command` - Voice commands processed
- `system_alert` - System alerts and warnings

### Voice Streaming
```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws/voice');

// Stream audio data
ws.send(audioBuffer);

ws.onmessage = function(event) {
  const response = JSON.parse(event.data);
  if (response.type === 'transcription') {
    console.log('Transcribed:', response.text);
  }
};
```

## ❌ Error Codes

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error

### Application Error Codes
```json
{
  "error": {
    "code": "AGENT_NOT_FOUND",
    "message": "The specified agent was not found",
    "details": {
      "agent_name": "invalid_agent",
      "available_agents": ["voice_agent", "automation_agent"]
    }
  }
}
```

**Common Error Codes:**
- `INVALID_TOKEN` - Authentication token invalid
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `AGENT_BUSY` - Agent currently processing another request
- `VOICE_PROCESSING_FAILED` - Voice processing error
- `MEMORY_STORAGE_FAILED` - Memory storage error
- `AUTOMATION_EXECUTION_FAILED` - Automation execution error

## 🚥 Rate Limiting

### Default Limits
- **API Requests**: 60 requests per minute per IP
- **Voice Processing**: 20 requests per minute per user
- **File Uploads**: 10 uploads per minute per user
- **WebSocket Connections**: 5 concurrent connections per user

### Rate Limit Headers
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1625097600
```

### Rate Limit Response
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again later.",
    "details": {
      "limit": 60,
      "reset_time": "2025-07-06T12:15:00Z"
    }
  }
}
```

## 📚 SDKs and Examples

### Python SDK
```python
from archie_client import ArchieClient

client = ArchieClient(
    base_url="http://localhost:8080/api",
    api_key="your_api_key"
)

# Send message to agent
response = client.agents.send_message(
    agent_name="voice_agent",
    message="What's the weather like?"
)

# Process voice command
audio_response = client.voice.process_audio("path/to/audio.wav")

# Create automation rule
rule = client.automation.create_rule(
    name="Morning Routine",
    description="Turn on lights and play news at 7 AM"
)
```

### JavaScript SDK
```javascript
import { ArchieClient } from '@archie/client';

const client = new ArchieClient({
  baseURL: 'http://localhost:8080/api',
  apiKey: 'your_api_key'
});

// Send message
const response = await client.agents.sendMessage('voice_agent', {
  message: 'Turn on the lights',
  context: { location: 'living_room' }
});

// WebSocket connection
const ws = client.createWebSocket('/events');
ws.on('agent_message', (data) => {
  console.log('Agent response:', data);
});
```

### cURL Examples

#### Basic Authentication
```bash
# Login
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Use token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v1/agents
```

#### Send Voice Command
```bash
curl -X POST http://localhost:8080/api/v1/voice/process \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio_file=@command.wav" \
  -F "format=wav"
```

#### Create Automation
```bash
curl -X POST http://localhost:8080/api/v1/automation/rules \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Security Alert",
    "description": "Send alert when door opens at night",
    "natural_language": "when front door opens between 10pm and 6am send notification"
  }'
```

## 📖 Additional Resources

- [OpenAPI Specification](openapi.yaml) - Complete API specification
- [Postman Collection](archie-api.postman_collection.json) - Ready-to-use API collection
- [SDKs Repository](https://github.com/your-org/archie-sdks) - Official SDKs
- [API Examples](examples/) - Code examples and tutorials

---

For questions about the API, please refer to our [FAQ](../faq/README.md) or create an issue in the [GitHub repository](https://github.com/your-org/archie/issues).