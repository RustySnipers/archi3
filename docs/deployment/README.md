# Archie Deployment Guide

This guide covers deploying Archie in various environments, from development to production.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Deployment](#quick-start-deployment)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

#### Minimum Requirements
- **Operating System**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: Stable internet connection
- **Docker**: Version 20.0+
- **Docker Compose**: Version 2.0+

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 16GB+
- **Storage**: 100GB NVMe SSD
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for enhanced AI features)
- **Network**: High-speed internet (100Mbps+)

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose-plugin curl git

# CentOS/RHEL
sudo yum install -y docker docker-compose curl git

# macOS (with Homebrew)
brew install docker docker-compose git

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Hardware Recommendations

#### Development Environment
- 8GB RAM minimum
- 4 CPU cores
- 50GB storage
- No GPU required

#### Production Environment
- 16GB+ RAM
- 8+ CPU cores
- 100GB+ NVMe SSD
- NVIDIA GPU recommended
- Load balancer for high availability

## 🚀 Quick Start Deployment

### 1. Clone Repository

```bash
git clone https://github.com/your-org/archie.git
cd archie
```

### 2. Configuration Setup

```bash
# Copy configuration template
cp config_template.yaml configs/archie_config.yaml

# Edit configuration
nano configs/archie_config.yaml
```

### 3. Environment Variables

Create `.env` file:

```bash
# Core Configuration
ARCHIE_ENV=development
ARCHIE_LOG_LEVEL=INFO

# Database Configuration
POSTGRES_DB=archie
POSTGRES_USER=archie
POSTGRES_PASSWORD=secure_password_here

# Security
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# External Services
OPENAI_API_KEY=your_openai_key_here
TELEGRAM_BOT_TOKEN=your_telegram_token_here

# Network Configuration
ARCHIE_PORT=8080
ARCHIE_HOST=0.0.0.0
```

### 4. Quick Deploy

```bash
# Pull and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f archie
```

### 5. Verify Installation

```bash
# Check API health
curl http://localhost:8080/health

# Check voice interface
curl http://localhost:8080/api/voice/status

# Check automation engine
curl http://localhost:8080/api/automation/status
```

## 🏭 Production Deployment

### 1. Infrastructure Setup

#### Option A: Single Server Deployment

```bash
# Server specifications
- CPU: 8 cores, 3.0GHz
- RAM: 16GB
- Storage: 100GB NVMe SSD
- OS: Ubuntu 22.04 LTS
```

#### Option B: Distributed Deployment

```bash
# Load Balancer + 2 App Servers + Database Server
- Load Balancer: 2 CPU, 4GB RAM
- App Servers: 8 CPU, 16GB RAM each
- Database Server: 4 CPU, 8GB RAM, 200GB SSD
```

### 2. Production Configuration

Create `configs/production.yaml`:

```yaml
environment: production

logging:
  level: INFO
  format: json
  file: /var/log/archie/archie.log
  max_size: 100MB
  backup_count: 5

database:
  host: db.archie.local
  port: 5432
  name: archie_prod
  ssl_mode: require
  pool_size: 20
  max_overflow: 30

redis:
  host: redis.archie.local
  port: 6379
  db: 0
  ssl: true

security:
  jwt_expiry: 3600
  rate_limiting:
    enabled: true
    requests_per_minute: 60
  cors:
    enabled: true
    origins: ["https://archie.yourdomain.com"]

performance:
  optimization_level: balanced
  cache_enabled: true
  max_cache_memory_mb: 2048
  monitoring:
    enabled: true
    metrics_retention_days: 30

integrations:
  home_assistant:
    url: https://ha.yourdomain.com
    ssl_verify: true
  
  n8n:
    url: https://n8n.yourdomain.com
    ssl_verify: true
```

### 3. SSL/TLS Setup

#### Using Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d archie.yourdomain.com

# Update nginx configuration
sudo nano /etc/nginx/sites-available/archie
```

#### Nginx Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name archie.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/archie.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/archie.yourdomain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/voice/stream {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 4. Database Setup

#### PostgreSQL Configuration

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres createuser archie
sudo -u postgres createdb archie
sudo -u postgres psql -c "ALTER USER archie PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE archie TO archie;"
```

#### Database Migration

```bash
# Run initial migration
docker exec archie_app python -m alembic upgrade head

# Verify tables
docker exec archie_db psql -U archie -d archie -c "\dt"
```

### 5. Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'archie'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### Grafana Dashboards

```bash
# Import pre-built dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @dashboards/archie-overview.json
```

### 6. Production Deployment Script

```bash
#!/bin/bash
# deploy_production.sh

set -e

echo "Starting Archie production deployment..."

# Update system
sudo apt update && sudo apt upgrade -y

# Pull latest code
git pull origin main

# Build production images
docker-compose -f docker-compose.prod.yml build

# Stop services gracefully
docker-compose down --timeout 30

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
sleep 30

# Run health checks
./scripts/health_check.py --production

# Update monitoring
docker-compose -f monitoring-compose.yml up -d

echo "Production deployment complete!"
```

## ⚙️ Configuration

### Environment-Specific Configurations

#### Development (`configs/development.yaml`)
```yaml
environment: development
debug: true
logging:
  level: DEBUG
  console: true
```

#### Staging (`configs/staging.yaml`)
```yaml
environment: staging
debug: false
logging:
  level: INFO
  file: /var/log/archie/staging.log
```

#### Production (`configs/production.yaml`)
```yaml
environment: production
debug: false
logging:
  level: INFO
  file: /var/log/archie/archie.log
  format: json
```

### Security Configuration

```yaml
security:
  authentication:
    method: jwt
    expiry: 3600
    refresh_enabled: true
  
  authorization:
    rbac_enabled: true
    default_role: user
  
  encryption:
    algorithm: AES-256-GCM
    key_rotation: 30d
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 10
```

### Performance Tuning

```yaml
performance:
  # Worker processes
  workers: 4
  worker_connections: 1000
  
  # Caching
  cache:
    memory_limit: 2GB
    ttl_default: 3600
    compression: true
  
  # Database
  database:
    pool_size: 20
    max_overflow: 30
    pool_timeout: 30
  
  # Optimization
  optimization_level: balanced
  auto_optimize: true
```

## 📊 Monitoring

### Health Checks

```bash
# API health check
curl http://localhost:8080/health

# Component health checks
curl http://localhost:8080/api/health/detailed

# Metrics endpoint
curl http://localhost:8080/metrics
```

### Log Monitoring

```bash
# Real-time logs
docker-compose logs -f archie

# Error logs only
docker-compose logs archie | grep ERROR

# Performance logs
docker-compose logs archie | grep "performance"
```

### Performance Metrics

Key metrics to monitor:
- **Response Time**: API response times
- **Memory Usage**: System and application memory
- **CPU Usage**: System CPU utilization
- **Error Rate**: Application error percentage
- **Active Users**: Current active sessions

## 🔄 Maintenance

### Regular Maintenance Tasks

#### Daily
```bash
# Check system health
./scripts/health_check.py

# Review error logs
tail -n 100 /var/log/archie/error.log

# Check disk space
df -h
```

#### Weekly
```bash
# Update dependencies
docker-compose pull

# Clean unused images
docker system prune -f

# Backup configuration
tar -czf backups/config-$(date +%Y%m%d).tar.gz configs/
```

#### Monthly
```bash
# Full system backup
./scripts/backup.py --full

# Security updates
sudo apt update && sudo apt upgrade

# Performance review
./scripts/performance_report.py
```

### Backup Strategy

#### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/archie"
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker exec archie_db pg_dump -U archie archie > "$BACKUP_DIR/db_$DATE.sql"

# Configuration backup
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" configs/

# Data backup
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" data/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

#### Restore Procedures

```bash
# Restore database
docker exec -i archie_db psql -U archie archie < backup.sql

# Restore configuration
tar -xzf config_backup.tar.gz

# Restart services
docker-compose restart
```

## 🔧 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs archie

# Check disk space
df -h

# Check port conflicts
netstat -tlnp | grep 8080
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Optimize cache
curl -X POST http://localhost:8080/api/admin/cache/clear

# Restart services
docker-compose restart archie
```

#### SSL Certificate Issues
```bash
# Check certificate expiry
openssl x509 -in /etc/letsencrypt/live/domain/cert.pem -noout -dates

# Renew certificate
sudo certbot renew

# Reload nginx
sudo nginx -s reload
```

### Performance Issues

#### Slow Response Times
1. Check database connections
2. Review cache hit rates
3. Monitor CPU/memory usage
4. Check network latency

#### High CPU Usage
1. Review active processes
2. Check for infinite loops
3. Optimize database queries
4. Scale horizontally

### Log Analysis

```bash
# Most common errors
grep ERROR /var/log/archie/archie.log | cut -d' ' -f5- | sort | uniq -c | sort -nr

# Performance bottlenecks
grep "slow" /var/log/archie/archie.log

# Authentication failures
grep "auth" /var/log/archie/archie.log | grep FAIL
```

## 📞 Support

### Getting Help
1. Check [Troubleshooting Guide](../troubleshooting/README.md)
2. Review [FAQ](../faq/README.md)
3. Search [GitHub Issues](https://github.com/your-org/archie/issues)
4. Create new issue with logs and configuration

### Emergency Contacts
- **Critical Issues**: emergency@archie.support
- **General Support**: support@archie.support
- **Community**: discussions@archie.support

---

For additional deployment scenarios and advanced configurations, see the [Advanced Deployment Guide](advanced-deployment.md).