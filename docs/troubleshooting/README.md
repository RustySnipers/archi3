# Archie Troubleshooting Guide

This comprehensive troubleshooting guide will help you resolve common issues with your Archie personal assistant.

## 📋 Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Voice Interface Issues](#voice-interface-issues)
3. [Smart Home Control Problems](#smart-home-control-problems)
4. [Automation & Workflow Issues](#automation--workflow-issues)
5. [Performance Problems](#performance-problems)
6. [Network & Connectivity Issues](#network--connectivity-issues)
7. [System Errors](#system-errors)
8. [Data & Memory Issues](#data--memory-issues)
9. [Integration Problems](#integration-problems)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

## 🔍 Quick Diagnostics

### System Health Check

Before diving into specific issues, run these quick checks:

#### 1. Check System Status
- **Web Interface**: Visit `/health` endpoint
- **Voice Command**: "Archie, what's your status?"
- **API**: `GET /api/v1/health`

#### 2. Verify Core Services
```bash
# Check Docker containers
docker-compose ps

# Check service logs
docker-compose logs archie
```

#### 3. Test Basic Functionality
- Say "Archie, hello" (voice test)
- Send a simple text message (agent test)
- Check device connectivity (integration test)

### Common Status Indicators

| Status | Meaning | Action Needed |
|--------|---------|---------------|
| 🟢 Healthy | All systems operational | None |
| 🟡 Warning | Minor issues detected | Monitor closely |
| 🔴 Error | Critical issue | Immediate attention |
| ⚪ Unknown | Status unavailable | Check connectivity |

## 🎤 Voice Interface Issues

### Archie Doesn't Respond to Wake Word

#### Symptoms
- No response when saying "Archie"
- Wake word detection seems inactive
- Voice interface appears offline

#### Solutions

**1. Check Microphone**
```bash
# Test microphone on Linux
arecord -d 5 test.wav && aplay test.wav

# Check audio devices
pactl list sources short
```

**2. Verify Wake Word Settings**
- Check current wake word: "Archie, what's your wake word?"
- Reset to default: "Archie, reset wake word to default"
- Adjust sensitivity in web interface

**3. Restart Voice Services**
```bash
# Restart voice container
docker-compose restart archie_voice

# Or restart all services
docker-compose restart
```

**4. Check Voice Model Status**
- Voice command: "Archie, check voice model status"
- Web interface: Settings → Voice → Model Status

### Poor Voice Recognition Accuracy

#### Symptoms
- Frequent misunderstanding of commands
- Low confidence scores
- Incorrect transcriptions

#### Solutions

**1. Improve Audio Quality**
- Reduce background noise
- Move closer to microphone (3-6 feet optimal)
- Use a higher quality microphone
- Avoid speaking too fast or too slow

**2. Recalibrate Voice System**
```bash
# Voice recalibration
curl -X POST http://localhost:8080/api/v1/voice/calibrate
```

**3. Check Language Settings**
- Verify language model matches your accent
- Update to latest speech recognition model
- Adjust pronunciation settings

**4. Environmental Factors**
- Minimize echo (add soft furnishings)
- Reduce competing audio sources
- Ensure stable room temperature (affects hardware)

### Text-to-Speech Issues

#### Symptoms
- No audio output from Archie
- Distorted or robotic voice
- Audio cutting out

#### Solutions

**1. Check Audio Output**
```bash
# Test speakers on Linux
speaker-test -t wav -c 2

# Check audio devices
pactl list sinks short
```

**2. Adjust Voice Settings**
- Change voice model: "Archie, change voice to [model name]"
- Adjust speech speed: "Archie, speak slower/faster"
- Reset voice settings to default

**3. Audio Driver Issues**
```bash
# Restart audio services
sudo systemctl restart pulseaudio

# Check ALSA configuration
cat /proc/asound/cards
```

## 🏠 Smart Home Control Problems

### Devices Not Responding

#### Symptoms
- "Device not found" errors
- Commands accepted but no action taken
- Partial device control working

#### Solutions

**1. Verify Device Connection**
```bash
# Check Home Assistant connection
curl http://localhost:8123/api/states

# Test device status
curl -H "Authorization: Bearer $HA_TOKEN" \
  http://localhost:8123/api/states/light.living_room
```

**2. Refresh Device List**
- Voice: "Archie, refresh my devices"
- API: `POST /api/v1/integrations/home_assistant/refresh`
- Web interface: Integrations → Home Assistant → Refresh

**3. Check Device Names**
- List all devices: "Archie, what devices do you know about?"
- Verify exact naming in Home Assistant
- Use entity IDs if friendly names don't work

**4. Network Connectivity**
```bash
# Test Home Assistant connectivity
ping ha.local

# Check network routes
traceroute ha.local
```

### Incorrect Device Actions

#### Symptoms
- Wrong device responds to command
- Unexpected device behavior
- Multiple devices responding to single command

#### Solutions

**1. Improve Command Specificity**
- Use full device names: "bedroom lamp" not "lamp"
- Include room names: "living room lights"
- Use unique identifiers when possible

**2. Check Device Grouping**
- Review Home Assistant groups and areas
- Verify device area assignments
- Adjust entity naming for clarity

**3. Update Device Mapping**
```yaml
# In archie_config.yaml
integrations:
  home_assistant:
    device_mapping:
      "main light": "light.living_room_main"
      "bedroom lamp": "light.bedroom_table_lamp"
```

### Scene and Routine Issues

#### Symptoms
- Scenes don't activate completely
- Some devices in scene don't respond
- Scene activation takes too long

#### Solutions

**1. Test Individual Components**
```bash
# Test each device in the scene individually
curl -X POST http://localhost:8123/api/services/light/turn_on \
  -H "Authorization: Bearer $HA_TOKEN" \
  -d '{"entity_id": "light.living_room"}'
```

**2. Check Scene Definition**
- Verify all entities exist in Home Assistant
- Check for entity ID changes
- Test scene in Home Assistant directly

**3. Adjust Timing**
```yaml
# Add delays between actions
scenes:
  movie_night:
    actions:
      - service: light.turn_off
        entity_id: light.living_room
      - delay: 2  # Wait 2 seconds
      - service: media_player.turn_on
        entity_id: media_player.tv
```

## 🔧 Automation & Workflow Issues

### Automations Not Triggering

#### Symptoms
- Expected automations don't run
- Manual trigger works but automatic doesn't
- Intermittent automation execution

#### Solutions

**1. Check Trigger Conditions**
- Review automation logs: "Archie, show automation logs"
- Verify all conditions are met
- Test with simplified conditions

**2. Debug Automation**
```bash
# Check automation status
curl http://localhost:8080/api/v1/automation/rules/{rule_id}/status

# View execution history
curl http://localhost:8080/api/v1/automation/rules/{rule_id}/history
```

**3. Common Trigger Issues**
- **Time-based**: Check timezone settings
- **Sensor-based**: Verify sensor connectivity
- **State-based**: Confirm entity state changes
- **Event-based**: Check event subscription

**4. Increase Logging**
```yaml
# In logging.yaml
loggers:
  automation:
    level: DEBUG
    handlers: [file]
```

### Workflow Generation Failures

#### Symptoms
- "Unable to create workflow" errors
- Incomplete workflow generation
- n8n integration not working

#### Solutions

**1. Check n8n Connection**
```bash
# Test n8n API
curl http://localhost:5678/rest/workflows

# Check authentication
curl -H "X-N8N-API-KEY: $N8N_API_KEY" \
  http://localhost:5678/rest/workflows
```

**2. Simplify Workflow Description**
- Use clear, simple language
- Break complex workflows into steps
- Avoid ambiguous references

**3. Verify Template Availability**
- Check workflow templates are loaded
- Update workflow generation rules
- Test with known working examples

### Automation Performance Issues

#### Symptoms
- Slow automation execution
- Timeouts during automation
- High system resource usage

#### Solutions

**1. Optimize Automation Logic**
```yaml
# Reduce complexity
automations:
  - name: "Simple Light Control"
    trigger: motion_detected
    condition: after_sunset
    action: turn_on_lights
    # Avoid nested conditions and multiple triggers
```

**2. Add Delays and Timeouts**
```yaml
# Add appropriate delays
automations:
  - actions:
    - service: light.turn_on
    - delay: 1  # Wait between actions
    - service: media_player.turn_on
```

**3. Monitor Resource Usage**
```bash
# Check system resources
docker stats

# Monitor automation performance
curl http://localhost:8080/api/v1/admin/performance
```

## ⚡ Performance Problems

### Slow Response Times

#### Symptoms
- Long delays between command and response
- Timeouts on voice commands
- Sluggish web interface

#### Solutions

**1. Check System Resources**
```bash
# Monitor CPU and memory
htop

# Check disk space
df -h

# Monitor Docker containers
docker stats
```

**2. Optimize Configuration**
```yaml
# In archie_config.yaml
performance:
  optimization_level: aggressive
  cache_enabled: true
  max_cache_memory_mb: 2048
  
  workers: 4  # Adjust based on CPU cores
  worker_connections: 1000
```

**3. Clear Caches**
```bash
# Clear all caches
curl -X POST http://localhost:8080/api/v1/admin/cache/clear

# Or via voice
"Archie, clear your cache"
```

**4. Database Optimization**
```bash
# Run database maintenance
docker exec archie_db vacuumdb -U archie archie

# Check database size
docker exec archie_db psql -U archie -c "SELECT pg_size_pretty(pg_database_size('archie'));"
```

### High Memory Usage

#### Symptoms
- System becoming unresponsive
- Out of memory errors
- Container restarts

#### Solutions

**1. Monitor Memory Usage**
```bash
# Check memory by container
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# System memory
free -h
```

**2. Adjust Memory Limits**
```yaml
# In docker-compose.yml
services:
  archie:
    mem_limit: 4g
    memswap_limit: 4g
```

**3. Optimize Memory Settings**
```yaml
# In archie_config.yaml
memory:
  max_cache_size: 1000
  cleanup_interval: 300  # Clean up every 5 minutes
  
performance:
  garbage_collection:
    enabled: true
    frequency: 60  # Run every minute
```

### CPU Performance Issues

#### Symptoms
- High CPU usage
- System overheating
- Slow processing

#### Solutions

**1. Identify CPU-Heavy Processes**
```bash
# Monitor CPU usage
top -p $(docker inspect --format='{{.State.Pid}}' archie_app)

# Check process breakdown
docker exec archie_app ps aux --sort=-%cpu
```

**2. Optimize Processing**
```yaml
# Reduce processing intensity
voice:
  stt_model: "whisper-tiny"  # Use smaller model
  processing_threads: 2

multimodal:
  image_max_size: 1024  # Reduce max image size
  video_sample_rate: 10  # Sample fewer frames
```

**3. Scale Processing**
```yaml
# Distribute load
services:
  archie_worker:
    image: archie
    command: worker
    deploy:
      replicas: 3
```

## 🌐 Network & Connectivity Issues

### Internet Connectivity Problems

#### Symptoms
- "No internet connection" errors
- External service integrations failing
- API timeouts

#### Solutions

**1. Test Basic Connectivity**
```bash
# Test internet connection
ping 8.8.8.8

# Test DNS resolution
nslookup google.com

# Test specific services
curl -I https://api.openai.com
```

**2. Check Proxy Settings**
```yaml
# In archie_config.yaml if behind proxy
network:
  proxy:
    http: "http://proxy.company.com:8080"
    https: "https://proxy.company.com:8080"
    no_proxy: "localhost,127.0.0.1"
```

**3. Firewall Issues**
```bash
# Check firewall rules
sudo ufw status

# Open required ports
sudo ufw allow 8080
sudo ufw allow 5678  # n8n
sudo ufw allow 8123  # Home Assistant
```

### Local Network Issues

#### Symptoms
- Can't reach Home Assistant
- Device discovery failures
- MQTT connection problems

#### Solutions

**1. Network Connectivity**
```bash
# Test local network
ping 192.168.1.1

# Check routing
ip route

# Test specific ports
telnet ha.local 8123
```

**2. Docker Network Issues**
```bash
# Check Docker networks
docker network ls

# Inspect network configuration
docker network inspect archie_default

# Restart Docker networking
sudo systemctl restart docker
```

**3. DNS Resolution**
```bash
# Check DNS in containers
docker exec archie_app nslookup ha.local

# Update DNS settings
docker-compose down
docker-compose up -d
```

## 🚨 System Errors

### Application Crashes

#### Symptoms
- Archie stops responding
- Container restarts frequently
- Critical error messages

#### Solutions

**1. Check Error Logs**
```bash
# Application logs
docker-compose logs --tail=100 archie

# System logs
journalctl -u docker -n 50

# Container inspection
docker inspect archie_app
```

**2. Increase Resource Limits**
```yaml
# In docker-compose.yml
services:
  archie:
    mem_limit: 8g
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
```

**3. Recovery Procedures**
```bash
# Graceful restart
docker-compose restart archie

# Full restart
docker-compose down
docker-compose up -d

# Clean restart
docker-compose down
docker system prune -f
docker-compose up -d
```

### Database Connection Issues

#### Symptoms
- "Database connection failed" errors
- Data not persisting
- Transaction failures

#### Solutions

**1. Check Database Status**
```bash
# Check PostgreSQL container
docker-compose logs postgres

# Test connection
docker exec archie_db psql -U archie -c "SELECT 1;"
```

**2. Connection Pool Issues**
```yaml
# In archie_config.yaml
database:
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

**3. Database Recovery**
```bash
# Restart database
docker-compose restart postgres

# Check database integrity
docker exec archie_db pg_dump -U archie archie > backup.sql
```

### Authentication Errors

#### Symptoms
- "Unauthorized" API responses
- Login failures
- Token expiration issues

#### Solutions

**1. Reset Authentication**
```bash
# Generate new JWT secret
openssl rand -hex 32

# Update configuration
# Restart services
docker-compose restart
```

**2. Check Token Validity**
```bash
# Decode JWT token
echo $TOKEN | cut -d. -f2 | base64 -d

# Test API with token
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v1/health
```

## 💾 Data & Memory Issues

### Memory Management Problems

#### Symptoms
- Memory usage growing over time
- Cache not clearing properly
- Old data not being cleaned up

#### Solutions

**1. Manual Memory Cleanup**
```bash
# Clear caches
curl -X POST http://localhost:8080/api/v1/admin/cache/clear

# Force garbage collection
curl -X POST http://localhost:8080/api/v1/admin/memory/cleanup
```

**2. Automated Cleanup Configuration**
```yaml
# In archie_config.yaml
memory:
  cleanup_schedule: "0 */6 * * *"  # Every 6 hours
  max_memory_age_days: 30
  cache_cleanup_threshold: 80  # Clean when 80% full
```

**3. Database Maintenance**
```bash
# Clean old records
docker exec archie_db psql -U archie -c "
  DELETE FROM memories WHERE created_at < NOW() - INTERVAL '30 days';
  DELETE FROM events WHERE created_at < NOW() - INTERVAL '7 days';
  VACUUM ANALYZE;
"
```

### Data Corruption Issues

#### Symptoms
- Inconsistent data responses
- Database errors
- Missing user preferences

#### Solutions

**1. Data Integrity Check**
```bash
# Check database integrity
docker exec archie_db pg_dump -U archie archie --schema-only > schema.sql

# Validate data
docker exec archie_db psql -U archie -c "
  SELECT table_name, column_name 
  FROM information_schema.columns 
  WHERE table_schema = 'public';
"
```

**2. Restore from Backup**
```bash
# List available backups
ls /backups/archie/

# Restore database
docker exec -i archie_db psql -U archie archie < backup_20250706.sql

# Restart services
docker-compose restart
```

## 🔌 Integration Problems

### Home Assistant Integration

#### Symptoms
- Device commands not working
- State updates not received
- Authentication failures

#### Solutions

**1. Check Home Assistant API**
```bash
# Test HA API
curl -H "Authorization: Bearer $HA_TOKEN" \
  http://localhost:8123/api/

# Check entity states
curl -H "Authorization: Bearer $HA_TOKEN" \
  http://localhost:8123/api/states
```

**2. Update Integration Settings**
```yaml
# In archie_config.yaml
integrations:
  home_assistant:
    url: "http://homeassistant:8123"
    token: "your_long_lived_access_token"
    ssl_verify: false  # For local instances
    timeout: 30
```

**3. Refresh Integration**
```bash
# Restart integration
curl -X POST http://localhost:8080/api/v1/integrations/home_assistant/restart

# Reload devices
curl -X POST http://localhost:8080/api/v1/integrations/home_assistant/reload
```

### Telegram Bot Issues

#### Symptoms
- Notifications not sent
- Bot not responding
- "Unauthorized" errors

#### Solutions

**1. Verify Bot Token**
```bash
# Test bot token
curl https://api.telegram.org/bot$BOT_TOKEN/getMe

# Check bot status
curl https://api.telegram.org/bot$BOT_TOKEN/getUpdates
```

**2. Update Bot Configuration**
```yaml
# In archie_config.yaml
integrations:
  telegram:
    bot_token: "your_bot_token"
    chat_ids: ["your_chat_id"]
    parse_mode: "HTML"
```

### n8n Workflow Integration

#### Symptoms
- Workflows not executing
- n8n connection errors
- Workflow creation failures

#### Solutions

**1. Check n8n Status**
```bash
# Test n8n connection
curl http://localhost:5678/healthz

# Check workflows
curl -H "X-N8N-API-KEY: $N8N_KEY" \
  http://localhost:5678/rest/workflows
```

**2. Restart n8n Integration**
```bash
# Restart n8n container
docker-compose restart n8n

# Reload workflows
curl -X POST http://localhost:8080/api/v1/integrations/n8n/reload
```

## 🔧 Advanced Troubleshooting

### Debug Mode

Enable debug mode for detailed logging:

```yaml
# In archie_config.yaml
logging:
  level: DEBUG
  handlers:
    - console
    - file
  
debug:
  enabled: true
  profiling: true
  trace_requests: true
```

### Performance Profiling

```bash
# Generate performance report
curl http://localhost:8080/api/v1/admin/performance/report

# Profile memory usage
curl http://localhost:8080/api/v1/admin/performance/memory

# Profile CPU usage
curl http://localhost:8080/api/v1/admin/performance/cpu
```

### System Diagnostics

```bash
#!/bin/bash
# comprehensive_diagnostics.sh

echo "=== Archie System Diagnostics ==="
echo "Date: $(date)"
echo

echo "=== Docker Status ==="
docker-compose ps
echo

echo "=== System Resources ==="
free -h
df -h
echo

echo "=== Network Connectivity ==="
ping -c 3 8.8.8.8
echo

echo "=== Service Health ==="
curl -s http://localhost:8080/health | jq '.'
echo

echo "=== Recent Errors ==="
docker-compose logs --tail=20 archie | grep ERROR
echo

echo "=== Performance Metrics ==="
curl -s http://localhost:8080/api/v1/admin/performance | jq '.data.current_metrics'
```

### Log Analysis Tools

```bash
# Extract error patterns
docker-compose logs archie | grep ERROR | cut -d' ' -f4- | sort | uniq -c | sort -nr

# Monitor real-time issues
docker-compose logs -f archie | grep -E "(ERROR|WARN|CRITICAL)"

# Performance analysis
docker-compose logs archie | grep "slow" | tail -20
```

## 📞 Getting Help

### Self-Help Resources

1. **Documentation**: Check relevant sections in `/docs`
2. **FAQ**: Review [Frequently Asked Questions](../faq/README.md)
3. **Community**: Search [Community Discussions](https://github.com/your-org/archie/discussions)

### Collecting Information for Support

When reporting issues, include:

```bash
# System information
uname -a
docker --version
docker-compose --version

# Configuration (sanitized)
cat configs/archie_config.yaml | grep -v token | grep -v password

# Recent logs
docker-compose logs --tail=50 archie

# System status
curl http://localhost:8080/health
```

### Emergency Recovery

If Archie is completely unresponsive:

```bash
# 1. Stop all services
docker-compose down

# 2. Clean Docker system
docker system prune -f

# 3. Restore from backup
tar -xzf backup_20250706.tar.gz

# 4. Start services
docker-compose up -d

# 5. Verify functionality
./scripts/health_check.py
```

---

**Still having issues?** Contact our support team with your diagnostic information and we'll help you resolve the problem quickly.