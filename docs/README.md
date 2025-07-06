# Archie - Personal AI Assistant

## Overview

Archie is an advanced personal AI assistant that combines voice interaction, home automation, workflow generation, and intelligent learning capabilities. Built with a modular architecture, Archie provides a comprehensive platform for managing smart homes, automating tasks, and providing intelligent assistance through multiple modalities.

## 🌟 Key Features

### Core AI Capabilities
- **Multi-Agent Orchestration**: Specialized agents for different domains (voice, automation, memory, communication)
- **Advanced Memory System**: Semantic clustering, contextual retrieval, and adaptive importance scoring
- **Learning & Adaptation**: User preference learning, behavior pattern recognition, and adaptive responses
- **Context-Aware Processing**: Conversation pattern analysis and intelligent routing

### Voice & Multi-Modal Processing
- **Voice Interface**: Wake word detection, speech-to-text (Whisper), text-to-speech (Piper)
- **Multi-Modal Support**: Image, video, document, and audio processing
- **Intelligent Audio Processing**: Voice activity detection and audio enhancement

### Home Automation & Integration
- **Home Assistant Integration**: Complete device control and state management
- **Dynamic Workflow Generation**: Natural language to n8n workflow conversion
- **Advanced Automation Rules**: Intelligent rule creation with temporal and conditional logic
- **Device Control**: Lights, switches, sensors, and IoT device management

### Communication & Notifications
- **Multi-Channel Notifications**: Telegram, email, SMS backup
- **Workflow Integration**: Automated notification workflows
- **Context-Aware Messaging**: Intelligent message routing and formatting

### Performance & Monitoring
- **Resource Optimization**: Automatic memory and CPU optimization
- **Performance Monitoring**: Real-time system metrics and alerts
- **Intelligent Caching**: Multi-level caching with adaptive algorithms
- **Health Monitoring**: System health checks and automated recovery

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Archie Orchestrator                     │
├─────────────────────────────────────────────────────────────┤
│  Voice Agent  │  Automation  │  Memory Agent │  Comm Agent │
│               │    Agent     │               │             │
├─────────────────────────────────────────────────────────────┤
│           Core Systems & Services                          │
│ • LLM Client     • Memory Manager    • Performance Opt    │
│ • MCP Client     • Learning System   • Multi-Modal Proc   │
│ • Tool Manager   • Workflow Gen      • Resource Monitor   │
├─────────────────────────────────────────────────────────────┤
│                 External Integrations                      │
│ • LocalAI       • Home Assistant     • ChromaDB          │
│ • n8n           • Telegram Bot       • Mosquitto (MQTT)   │
│ • Whisper STT   • Piper TTS          • Nginx (Reverse)    │
└─────────────────────────────────────────────────────────────┘
```

### System Components

#### 1. **Archie Core (`archie_core/`)**
- `agent.py` - Multi-agent orchestration and management
- `memory.py` - Advanced memory system with semantic clustering
- `learning.py` - Learning and adaptation engine
- `multimodal.py` - Multi-modal processing system
- `performance.py` - Performance optimization and monitoring
- `llm_client.py` - LocalAI integration for language models
- `mcp_client.py` - Model Context Protocol client
- `tools.py` - Tool management and execution

#### 2. **Voice Processing (`voice/`)**
- `voice_interface.py` - Voice interface coordinator
- `stt.py` - Speech-to-text with Whisper
- `tts.py` - Text-to-speech with Piper
- `audio_processor.py` - Audio processing and VAD
- `wake_word.py` - Wake word detection
- `voice_agent_integration.py` - Agent system integration

#### 3. **Integrations (`integrations/`)**
- `home_assistant.py` - Home Assistant API client
- `n8n_client.py` - n8n workflow management
- `workflow_generator.py` - Dynamic workflow creation
- `advanced_automation.py` - Advanced automation engine
- `telegram_client.py` - Telegram bot integration
- `notification_manager.py` - Multi-channel notifications

#### 4. **MCP Servers (`mcp_servers/`)**
- `home_assistant_mcp.py` - HA MCP server
- `n8n_mcp.py` - n8n MCP server

#### 5. **Infrastructure (`docker/`, `configs/`)**
- Docker containers for all services
- Configuration management
- Security and networking setup

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- 8GB+ RAM recommended
- NVIDIA GPU (optional, for enhanced AI features)

### Basic Setup
1. Clone the repository
2. Copy `config_template.yaml` to `configs/archie_config.yaml`
3. Configure your settings
4. Run `docker-compose up -d`
5. Access the interface at `http://localhost:8080`

For detailed setup instructions, see [Deployment Guide](deployment/README.md).

## 📚 Documentation Structure

### For Users
- [**User Manual**](user-guide/README.md) - Complete user guide
- [**Quick Start**](user-guide/quick-start.md) - Get started quickly
- [**Voice Commands**](user-guide/voice-commands.md) - Voice interaction guide
- [**Automation Guide**](user-guide/automation.md) - Creating automations

### For Developers
- [**API Documentation**](api/README.md) - Complete API reference
- [**Development Guide**](development/README.md) - Development setup
- [**Architecture Guide**](architecture/README.md) - System architecture
- [**Integration Guide**](integrations/README.md) - Adding integrations

### For Administrators
- [**Deployment Guide**](deployment/README.md) - Production deployment
- [**Configuration Guide**](configuration/README.md) - System configuration
- [**Monitoring Guide**](monitoring/README.md) - System monitoring
- [**Troubleshooting**](troubleshooting/README.md) - Common issues

## 🔧 Configuration

Archie uses YAML configuration files located in the `configs/` directory:

- `archie_config.yaml` - Main configuration
- `logging.yaml` - Logging configuration
- `security.yaml` - Security settings

Key configuration sections:
- **LLM Settings** - LocalAI model configuration
- **Voice Settings** - STT/TTS model selection
- **Integration Settings** - External service configuration
- **Performance Settings** - Optimization parameters

## 🔌 Integrations

### Supported Platforms
- **Home Assistant** - Complete smart home integration
- **n8n** - Workflow automation platform
- **Telegram** - Messaging and notifications
- **LocalAI** - Local language model inference
- **ChromaDB** - Vector database for memory
- **Whisper** - Speech recognition
- **Piper** - Speech synthesis

### Third-Party Services
- **Weather APIs** - Weather data integration
- **Calendar Services** - Schedule integration
- **Email Services** - Email notifications
- **SMS Gateways** - SMS backup notifications

## 🛡️ Security

- **API Authentication** - JWT-based authentication
- **Encrypted Communications** - TLS/SSL encryption
- **Access Control** - Role-based permissions
- **Data Privacy** - Local data processing
- **Security Monitoring** - Real-time security alerts

## 📊 Monitoring & Performance

- **Real-time Metrics** - System performance monitoring
- **Health Checks** - Automated health monitoring
- **Alert System** - Performance and error alerts
- **Resource Optimization** - Automatic optimization
- **Logging** - Comprehensive logging system

## 🆘 Support

### Community
- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Community discussions and help
- **Wiki** - Community-maintained documentation

### Documentation
- [**Troubleshooting Guide**](troubleshooting/README.md)
- [**FAQ**](faq/README.md)
- [**Known Issues**](known-issues/README.md)

## 📋 Requirements

### Minimum System Requirements
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: Stable internet connection
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows with WSL2

### Recommended System Requirements
- **CPU**: 8 cores, 3.0GHz
- **RAM**: 16GB
- **Storage**: 100GB NVMe SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)
- **Network**: High-speed internet connection

## 🔄 Version History

### v1.0.0 (Current)
- Complete multi-agent architecture
- Advanced memory and learning systems
- Multi-modal processing capabilities
- Performance optimization
- Comprehensive integrations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** - Whisper speech recognition
- **Piper** - Text-to-speech synthesis
- **Home Assistant** - Smart home platform
- **n8n** - Workflow automation
- **ChromaDB** - Vector database
- **LocalAI** - Local LLM inference

---

**Archie** - Your intelligent personal assistant for the modern smart home.