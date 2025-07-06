# Archie - Personal AI Assistant

🤖 **A sophisticated, zero-cost, local-first AI assistant with voice interface, smart home integration, and workflow automation**

## Overview

Archie is a comprehensive personal AI assistant designed to run entirely on your local infrastructure, ensuring privacy and eliminating ongoing costs. It combines advanced AI capabilities with smart home integration, workflow automation, and voice interaction.

### Key Features

- 🎙️ **Voice Interface**: Wake word detection, speech-to-text, and natural text-to-speech
- 🏠 **Smart Home Integration**: Complete Home Assistant integration with device control
- 🔄 **Workflow Automation**: AI-powered workflow generation using n8n
- 💬 **Communication**: Telegram bot integration with notification system
- 🧠 **Memory System**: Persistent memory with ChromaDB vector storage
- 📊 **Monitoring**: Full observability with Prometheus and Grafana
- 🔒 **Security**: Local-first processing with enterprise-grade security
- 🐳 **Easy Deployment**: Complete Docker Compose setup

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Input   │    │   Smart Home    │    │   Workflows     │
│   (Whisper)     │    │ (Home Assistant)│    │     (n8n)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Archie Core   │
                    │   (AI Agent)    │
                    └─────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Memory        │    │   LocalAI       │    │ Communications  │
│  (ChromaDB)     │    │   (LLM)         │    │   (Telegram)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- **Hardware**: 4-core CPU, 16GB RAM, 64GB storage (minimum)
- **Software**: Docker, Docker Compose, Linux/WSL2
- **Network**: Internet connection for initial setup

### Installation

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd archie
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Configure Environment**:
   ```bash
   # Edit .env file with your settings
   cp .env.example .env
   nano .env
   ```

3. **Start Services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify Installation**:
   ```bash
   python3 scripts/health_check.py
   ```

### First Run

1. **Access Home Assistant**: http://localhost:8123
2. **Setup n8n**: http://localhost:5678
3. **Configure Telegram Bot**: Add your bot token to .env
4. **Test Voice Interface**: Say "Archie" followed by a command

## Configuration

### Core Configuration

Edit `configs/archie_config.yaml`:

```yaml
# Voice settings
voice:
  enabled: true
  wake_word: "archie"
  
# Home Assistant integration
homeassistant:
  url: "http://homeassistant:8123"
  token: "your-token-here"
  
# Telegram bot
telegram:
  bot_token: "your-bot-token"
  chat_id: "your-chat-id"
```

### Environment Variables

Key environment variables in `.env`:

```bash
# Core settings
ARCHIE_LOG_LEVEL=INFO
ARCHIE_ENABLE_VOICE=true
ARCHIE_ENABLE_AUTOMATION=true

# Integration tokens
HOMEASSISTANT_TOKEN=your-ha-token
TELEGRAM_BOT_TOKEN=your-bot-token
N8N_API_KEY=your-n8n-key

# Security
JWT_SECRET=your-jwt-secret
SECRET_KEY=your-secret-key
```

## Usage

### Voice Commands

- **"Archie, what's the weather?"** - Get weather information
- **"Archie, turn on the lights"** - Control smart home devices
- **"Archie, create a morning routine"** - Generate automation workflows
- **"Archie, remind me to..."** - Set reminders and notifications

### API Endpoints

- **Health Check**: `GET /health`
- **Voice Transcription**: `POST /api/voice/transcribe`
- **Device Control**: `POST /api/devices/control`
- **Workflow Creation**: `POST /api/workflows/create`
- **Memory Query**: `GET /api/memory/query`

### Telegram Bot

Send messages to your Telegram bot:
- `/status` - Check Archie status
- `/devices` - List available devices
- `/workflows` - Show active workflows
- Text messages for natural language processing

## Services

### Core Services

| Service | Port | Purpose |
|---------|------|---------|
| Archie Core | 8000 | Main AI agent |
| LocalAI | 8080 | LLM backend |
| Home Assistant | 8123 | Smart home hub |
| n8n | 5678 | Workflow engine |
| ChromaDB | 8000 | Vector database |
| Mosquitto | 1883 | MQTT broker |

### Monitoring

| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboard |
| Nginx | 80/443 | Reverse proxy |

## Development

### Project Structure

```
archie/
├── archie_core/          # Main AI agent code
├── voice/                # Voice processing
├── integrations/         # External integrations
├── mcp_servers/          # MCP protocol servers
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── docker/               # Docker configurations
├── data/                 # Persistent data
├── docs/                 # Documentation
└── tests/                # Test files
```

### Adding New Integrations

1. Create integration module in `integrations/`
2. Add MCP server in `mcp_servers/`
3. Update configuration in `configs/`
4. Add tests in `tests/`

### Custom Workflows

Create custom n8n workflows:
1. Access n8n at http://localhost:5678
2. Create workflow using visual editor
3. Add webhook triggers for Archie integration
4. Test and deploy

## Troubleshooting

### Common Issues

**Voice not working**:
```bash
# Check audio devices
docker exec archie_core python -c "import sounddevice; print(sounddevice.query_devices())"

# Test microphone
docker exec archie_core python -c "import pyaudio; print('Audio OK')"
```

**Home Assistant connection**:
```bash
# Check HA status
curl http://localhost:8123/api/
```

**Service health**:
```bash
# Run health check
python3 scripts/health_check.py

# Check logs
docker-compose logs archie_core
```

### Log Files

- **Main logs**: `data/logs/archie.log`
- **Voice logs**: `data/logs/voice.log`
- **Error logs**: `data/logs/archie_errors.log`
- **Security logs**: `data/logs/security_audit.log`

## Backup and Recovery

### Create Backup

```bash
# Full backup
python3 scripts/backup.py create full

# Configuration only
python3 scripts/backup.py create config
```

### Restore Backup

```bash
# List available backups
python3 scripts/backup.py list

# Restore specific backup
python3 scripts/backup.py restore backup_name.tar.gz
```

## Security

### Security Features

- **Local Processing**: All AI processing happens locally
- **Encryption**: Data at rest and in transit encryption
- **Authentication**: JWT-based API authentication
- **Rate Limiting**: Request rate limiting and throttling
- **Audit Logging**: Complete audit trail
- **Network Isolation**: Docker network segmentation

### Security Best Practices

1. **Change Default Passwords**: Update all default credentials
2. **Enable Firewall**: Configure host firewall rules
3. **Regular Updates**: Keep Docker images updated
4. **Monitor Logs**: Review security audit logs
5. **Backup Encryption**: Encrypt backup files

## Performance

### Optimization

- **Model Selection**: Choose appropriate model sizes
- **Resource Limits**: Set Docker resource constraints
- **Caching**: Enable Redis caching for frequently accessed data
- **Load Balancing**: Use nginx for load balancing

### Monitoring

Access Grafana dashboard at http://localhost:3000:
- System resource usage
- Service health metrics
- Response time monitoring
- Error rate tracking

## API Reference

### Voice API

```bash
# Transcribe audio
curl -X POST -F "audio=@recording.wav" http://localhost:8000/api/voice/transcribe

# Text to speech
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  http://localhost:8000/api/voice/synthesize
```

### Device Control API

```bash
# List devices
curl http://localhost:8000/api/devices

# Control device
curl -X POST -H "Content-Type: application/json" \
  -d '{"device_id": "light.living_room", "action": "turn_on"}' \
  http://localhost:8000/api/devices/control
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: Check `docs/` directory
- **Issues**: Report bugs and feature requests
- **Community**: Join our discussion forums
- **Commercial Support**: Available for enterprise deployments

## Roadmap

### Phase 1 ✅
- [x] Project structure and Docker setup
- [x] Basic service orchestration
- [x] Configuration management
- [x] Monitoring and logging

### Phase 2 🔄
- [ ] LLM integration with LocalAI
- [ ] MCP protocol implementation
- [ ] Agent orchestration system
- [ ] Memory management

### Phase 3 📋
- [ ] Voice interface (Whisper + Piper)
- [ ] Audio processing pipeline
- [ ] Wake word detection
- [ ] Voice command processing

### Phase 4 📋
- [ ] Home Assistant integration
- [ ] n8n workflow automation
- [ ] Telegram bot implementation
- [ ] Device control interfaces

### Phase 5 📋
- [ ] Advanced AI features
- [ ] Natural language to workflow
- [ ] Context-aware automation
- [ ] Advanced memory system

### Phase 6 📋
- [ ] Complete documentation
- [ ] Testing framework
- [ ] Deployment automation
- [ ] Performance optimization

---

🚀 **Start your journey with Archie today!**

For detailed installation and configuration instructions, see the `docs/` directory.