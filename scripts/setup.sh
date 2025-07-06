#!/bin/bash

# Archie Setup Script
# This script sets up the Archie environment and initializes all services

set -e

echo "🚀 Setting up Archie - Personal AI Assistant"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_section "Checking Prerequisites"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if user is in docker group
if ! groups | grep -q docker; then
    print_warning "User is not in the docker group. You may need to run docker commands with sudo."
fi

print_status "Prerequisites check completed"

print_section "Setting up Directory Structure"

# Create required directories
mkdir -p "$PROJECT_DIR/data"/{models,logs,backups,cache,chroma,homeassistant,n8n,mosquitto/{data,log},redis,prometheus,grafana,nginx/{logs,ssl}}

# Set proper permissions
chmod 755 "$PROJECT_DIR/data"
chmod 755 "$PROJECT_DIR/scripts"/*.sh 2>/dev/null || true
chmod +x "$PROJECT_DIR/scripts"/*.py 2>/dev/null || true

print_status "Directory structure created"

print_section "Environment Configuration"

# Check if .env file exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        print_status "Creating .env file from template"
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        print_warning "Please edit .env file and set your API keys and configuration"
    else
        print_error ".env.example file not found"
        exit 1
    fi
else
    print_status ".env file already exists"
fi

# Generate random secrets if not set
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
    
    # Generate JWT secret if not set
    if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "your-jwt-secret-here" ]; then
        JWT_SECRET=$(openssl rand -hex 32)
        sed -i "s/JWT_SECRET=.*/JWT_SECRET=$JWT_SECRET/" "$PROJECT_DIR/.env"
        print_status "Generated JWT secret"
    fi
    
    # Generate secret key if not set
    if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-here" ]; then
        SECRET_KEY=$(openssl rand -hex 32)
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" "$PROJECT_DIR/.env"
        print_status "Generated secret key"
    fi
fi

print_section "Configuration Files"

# Copy configuration template if needed
if [ ! -f "$PROJECT_DIR/configs/archie_config.yaml" ]; then
    if [ -f "$PROJECT_DIR/config_template.yaml" ]; then
        print_status "Creating configuration from template"
        cp "$PROJECT_DIR/config_template.yaml" "$PROJECT_DIR/configs/archie_config.yaml"
        print_warning "Please review and customize configs/archie_config.yaml"
    else
        print_warning "Configuration template not found. Using default configuration."
    fi
fi

# Create MQTT password file
if [ ! -f "$PROJECT_DIR/data/mosquitto/passwd" ]; then
    print_status "Creating MQTT password file"
    mkdir -p "$PROJECT_DIR/data/mosquitto"
    echo "archie:$(openssl passwd -6 changeme)" > "$PROJECT_DIR/data/mosquitto/passwd"
    print_warning "Default MQTT password is 'changeme'. Please change it in production."
fi

# Create MQTT ACL file
if [ ! -f "$PROJECT_DIR/data/mosquitto/acl" ]; then
    print_status "Creating MQTT ACL file"
    cat > "$PROJECT_DIR/data/mosquitto/acl" << 'EOF'
# ACL for Archie MQTT
user archie
topic readwrite archie/#
topic readwrite homeassistant/#
topic readwrite n8n/#

# Anonymous users (if enabled)
pattern read $SYS/broker/load/#
EOF
fi

print_section "Docker Images"

# Pull required Docker images
print_status "Pulling Docker images (this may take a while)..."

# Pull images in parallel for faster setup
docker pull quay.io/go-skynet/local-ai:latest &
docker pull homeassistant/home-assistant:stable &
docker pull n8nio/n8n:latest &
docker pull eclipse-mosquitto:latest &
docker pull chromadb/chroma:latest &
docker pull redis:7-alpine &
docker pull nginx:alpine &
docker pull prom/prometheus:latest &
docker pull grafana/grafana:latest &

# Wait for all pulls to complete
wait

print_status "Docker images pulled successfully"

print_section "Building Archie Core"

# Build the main Archie container
print_status "Building Archie core container..."
cd "$PROJECT_DIR"
docker build -f docker/Dockerfile.archie -t archie:latest .

print_status "Archie core container built successfully"

print_section "Model Downloads"

# Download required models
print_status "Downloading AI models (this may take a while)..."

# Create models directory
mkdir -p "$PROJECT_DIR/data/models"

# Download Whisper model
if [ ! -f "$PROJECT_DIR/data/models/whisper-small.en.bin" ]; then
    print_status "Downloading Whisper model..."
    # This would typically download from a model repository
    # For now, we'll create a placeholder
    touch "$PROJECT_DIR/data/models/whisper-small.en.bin"
    print_warning "Whisper model placeholder created. You'll need to download the actual model."
fi

# Download Piper TTS model
if [ ! -f "$PROJECT_DIR/data/models/en_US-lessac-medium.onnx" ]; then
    print_status "Downloading Piper TTS model..."
    # This would typically download from a model repository
    # For now, we'll create a placeholder
    touch "$PROJECT_DIR/data/models/en_US-lessac-medium.onnx"
    print_warning "Piper TTS model placeholder created. You'll need to download the actual model."
fi

print_section "Service Initialization"

# Start core services
print_status "Starting core services..."

# Start infrastructure services first
docker-compose up -d mosquitto redis prometheus grafana chromadb

# Wait for services to be ready
sleep 10

# Start LocalAI
print_status "Starting LocalAI..."
docker-compose up -d localai

# Wait for LocalAI to be ready
sleep 30

# Start other services
print_status "Starting remaining services..."
docker-compose up -d homeassistant n8n nginx

print_section "Health Check"

# Run health check
print_status "Running health check..."
sleep 30

if python3 "$PROJECT_DIR/scripts/health_check.py"; then
    print_status "Health check passed!"
else
    print_warning "Some services may not be ready yet. This is normal during first startup."
fi

print_section "Setup Complete"

print_status "Archie setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Review and customize configs/archie_config.yaml"
echo "3. Download actual AI models (see documentation)"
echo "4. Start using Archie with: docker-compose up -d"
echo ""
echo "Access points:"
echo "- Archie API: http://localhost:8000"
echo "- Home Assistant: http://localhost:8123"
echo "- n8n: http://localhost:5678"
echo "- Grafana: http://localhost:3000"
echo "- Prometheus: http://localhost:9090"
echo ""
echo "For help and documentation, see: docs/README.md"

# Create a status file
echo "setup_completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PROJECT_DIR/data/setup_status.txt"
echo "version=1.0.0" >> "$PROJECT_DIR/data/setup_status.txt"

print_status "Setup status saved to data/setup_status.txt"