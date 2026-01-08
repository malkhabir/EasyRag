#!/bin/bash
# ==============================================================================
# EasyRag Setup Script - Linux/macOS
# ==============================================================================
# This script sets up the complete EasyRag environment:
#   1. Checks prerequisites (Docker, Python, Node.js)
#   2. Pulls and configures Docker containers
#   3. Downloads required ML models
#   4. Installs dependencies
#   5. Initializes the database
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [options]
#
# Options:
#   --full          Full setup (default)
#   --models-only   Only download/update models
#   --docker-only   Only setup Docker containers
#   --dev           Setup for development
#   --prod          Setup for production
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/models.yaml"
ENV_FILE="${SCRIPT_DIR}/.env"

# Default settings (can be overridden by config)
OLLAMA_MODEL="${OLLAMA_MODEL:-phi3}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                   ║"
    echo "║   ███████╗██╗███╗   ██╗██████╗  █████╗  ██████╗                  ║"
    echo "║   ██╔════╝██║████╗  ██║██╔══██╗██╔══██╗██╔════╝                  ║"
    echo "║   █████╗  ██║██╔██╗ ██║██████╔╝███████║██║  ███╗                 ║"
    echo "║   ██╔══╝  ██║██║╚██╗██║██╔══██╗██╔══██║██║   ██║                 ║"
    echo "║   ██║     ██║██║ ╚████║██║  ██║██║  ██║╚██████╔╝                 ║"
    echo "║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                  ║"
    echo "║                                                                   ║"
    echo "║   Financial Document RAG System - Setup Script                    ║"
    echo "║                                                                   ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# ==============================================================================
# Prerequisite Checks
# ==============================================================================

check_prerequisites() {
    log_step "Checking Prerequisites"
    
    local missing=()
    
    # Check Docker
    if check_command docker; then
        log_success "Docker: $(docker --version)"
    else
        missing+=("docker")
        log_error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if check_command docker-compose || docker compose version &> /dev/null; then
        log_success "Docker Compose: Available"
    else
        missing+=("docker-compose")
        log_error "Docker Compose is not installed"
    fi
    
    # Check Python (optional for local dev)
    if check_command python3; then
        log_success "Python: $(python3 --version)"
    elif check_command python; then
        log_success "Python: $(python --version)"
    else
        log_warning "Python not found (optional for Docker deployment)"
    fi
    
    # Check Node.js (optional for frontend dev)
    if check_command node; then
        log_success "Node.js: $(node --version)"
    else
        log_warning "Node.js not found (optional for Docker deployment)"
    fi
    
    # Check curl
    if check_command curl; then
        log_success "curl: Available"
    else
        missing+=("curl")
        log_error "curl is not installed"
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo ""
        log_error "Missing required tools: ${missing[*]}"
        echo ""
        echo "Please install the missing tools:"
        echo "  - Docker: https://docs.docker.com/get-docker/"
        echo "  - Docker Compose: https://docs.docker.com/compose/install/"
        echo "  - curl: Usually pre-installed, or use your package manager"
        exit 1
    fi
    
    log_success "All prerequisites satisfied!"
}

# ==============================================================================
# Docker Setup
# ==============================================================================

setup_docker() {
    log_step "Setting Up Docker Containers"
    
    cd "$SCRIPT_DIR"
    
    # Create necessary directories
    log_info "Creating data directories..."
    mkdir -p qdrant/storage
    mkdir -p rag-service/models
    mkdir -p rag-service/uploaded_pdfs
    mkdir -p rag-service/nltk_data
    
    # Pull latest images
    log_info "Pulling Docker images..."
    docker-compose pull
    
    # Start core services (Qdrant + Ollama)
    log_info "Starting Qdrant vector database..."
    docker-compose up -d qdrant
    
    log_info "Starting Ollama LLM service..."
    docker-compose up -d ollama
    
    # Wait for services to be ready
    log_info "Waiting for services to initialize..."
    sleep 5
    
    # Check if services are running
    if docker-compose ps | grep -q "qdrant.*Up"; then
        log_success "Qdrant is running on port 6335"
    else
        log_error "Qdrant failed to start"
        docker-compose logs qdrant
        exit 1
    fi
    
    if docker-compose ps | grep -q "ollama.*Up"; then
        log_success "Ollama is running on port 11435"
    else
        log_error "Ollama failed to start"
        docker-compose logs ollama
        exit 1
    fi
}

# ==============================================================================
# Model Setup
# ==============================================================================

setup_models() {
    log_step "Setting Up AI Models"
    
    # Pull Ollama model
    log_info "Pulling LLM model: ${OLLAMA_MODEL}..."
    
    # Try Docker exec first, fall back to direct ollama command
    if docker-compose ps | grep -q "ollama.*Up"; then
        docker-compose exec -T ollama ollama pull "${OLLAMA_MODEL}" || {
            log_warning "Docker exec failed, trying direct command..."
            if check_command ollama; then
                OLLAMA_HOST="http://localhost:11435" ollama pull "${OLLAMA_MODEL}"
            else
                log_error "Could not pull model. Please run: ollama pull ${OLLAMA_MODEL}"
            fi
        }
    else
        log_warning "Ollama not running via Docker. Trying local installation..."
        if check_command ollama; then
            ollama pull "${OLLAMA_MODEL}"
        else
            log_error "Ollama not available. Please start Docker services first."
            exit 1
        fi
    fi
    
    log_success "LLM model '${OLLAMA_MODEL}' is ready!"
    
    # Download embedding model (via Python)
    log_info "Downloading embedding model: ${EMBEDDING_MODEL}..."
    
    if check_command python3; then
        python3 -c "
from sentence_transformers import SentenceTransformer
import os

cache_dir = os.path.join('${SCRIPT_DIR}', 'rag-service', 'models')
os.makedirs(cache_dir, exist_ok=True)

print(f'Downloading ${EMBEDDING_MODEL} to {cache_dir}...')
model = SentenceTransformer('${EMBEDDING_MODEL}', cache_folder=cache_dir)
print('Model downloaded successfully!')
" 2>/dev/null || {
            log_warning "Could not download embedding model via Python."
            log_info "The model will be downloaded on first run."
        }
    else
        log_info "Python not available. Embedding model will download on first run."
    fi
    
    log_success "Model setup complete!"
}

# ==============================================================================
# Environment Setup
# ==============================================================================

setup_environment() {
    log_step "Configuring Environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating .env file from template..."
        cat > "$ENV_FILE" << EOF
# ==============================================================================
# EasyRag Environment Configuration
# ==============================================================================
# This file is auto-generated. Customize as needed.
# Sensitive values (API keys) should be set here, not in models.yaml
# ==============================================================================

# LLM Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11435
OLLAMA_MODEL=${OLLAMA_MODEL}

# Embedding Configuration
EMBEDDING_MODEL=${EMBEDDING_MODEL}

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6335

# API Keys (uncomment and set if using cloud providers)
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=your-key-here
# COHERE_API_KEY=your-key-here

# Debug Mode
DEBUG=false

# Logging
LOG_LEVEL=INFO
EOF
        log_success "Created .env file"
    else
        log_info ".env file already exists, skipping..."
    fi
    
    # Create config directory if needed
    if [ ! -d "${SCRIPT_DIR}/config" ]; then
        mkdir -p "${SCRIPT_DIR}/config"
    fi
    
    log_success "Environment configured!"
}

# ==============================================================================
# Frontend Setup
# ==============================================================================

setup_frontend() {
    log_step "Setting Up Frontend"
    
    cd "${SCRIPT_DIR}/frontend"
    
    if check_command npm; then
        log_info "Installing frontend dependencies..."
        npm install
        log_success "Frontend dependencies installed!"
    else
        log_warning "npm not found. Skipping frontend setup."
        log_info "To setup frontend manually: cd frontend && npm install"
    fi
}

# ==============================================================================
# Backend Setup
# ==============================================================================

setup_backend() {
    log_step "Setting Up Backend"
    
    cd "${SCRIPT_DIR}/rag-service"
    
    if check_command python3; then
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            log_info "Creating Python virtual environment..."
            python3 -m venv venv
        fi
        
        log_info "Installing Python dependencies..."
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        
        log_success "Backend dependencies installed!"
    else
        log_warning "Python not found. Skipping local backend setup."
        log_info "Backend will run in Docker container."
    fi
}

# ==============================================================================
# Health Check
# ==============================================================================

health_check() {
    log_step "Running Health Checks"
    
    local all_healthy=true
    
    # Check Qdrant
    log_info "Checking Qdrant..."
    if curl -s "http://localhost:6335/health" | grep -q "ok\|true"; then
        log_success "Qdrant: Healthy"
    else
        log_warning "Qdrant: Not responding (may still be initializing)"
        all_healthy=false
    fi
    
    # Check Ollama
    log_info "Checking Ollama..."
    if curl -s "http://localhost:11435/api/tags" > /dev/null 2>&1; then
        log_success "Ollama: Healthy"
        
        # List available models
        local models=$(curl -s "http://localhost:11435/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$models" ]; then
            log_info "Available models: $models"
        fi
    else
        log_warning "Ollama: Not responding (may still be initializing)"
        all_healthy=false
    fi
    
    if $all_healthy; then
        log_success "All services are healthy!"
    else
        log_warning "Some services may need more time to initialize."
        log_info "Run 'docker-compose logs -f' to monitor startup."
    fi
}

# ==============================================================================
# Print Summary
# ==============================================================================

print_summary() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete! 🎉                             ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Quick Start:${NC}"
    echo "  1. Start all services:     docker-compose up -d"
    echo "  2. Start frontend:         cd frontend && npm run dev"
    echo "  3. Start backend:          cd rag-service && ./run.sh"
    echo ""
    echo -e "${CYAN}Service URLs:${NC}"
    echo "  • Frontend:    http://localhost:5173"
    echo "  • Backend API: http://localhost:8080"
    echo "  • Qdrant:      http://localhost:6335"
    echo "  • Ollama:      http://localhost:11435"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  • Models:      config/models.yaml"
    echo "  • Environment: .env"
    echo ""
    echo -e "${CYAN}Useful Commands:${NC}"
    echo "  • View logs:          docker-compose logs -f"
    echo "  • Stop services:      docker-compose down"
    echo "  • Pull new model:     docker-compose exec ollama ollama pull <model>"
    echo "  • Swap LLM:           Edit config/models.yaml and restart"
    echo ""
    echo -e "${YELLOW}Need help? Check the README.md or open an issue on GitHub.${NC}"
    echo ""
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    print_banner
    
    # Parse arguments
    local mode="full"
    for arg in "$@"; do
        case $arg in
            --models-only)
                mode="models"
                ;;
            --docker-only)
                mode="docker"
                ;;
            --dev)
                mode="dev"
                ;;
            --prod)
                mode="prod"
                ;;
            --help)
                echo "Usage: ./setup.sh [options]"
                echo ""
                echo "Options:"
                echo "  --full          Full setup (default)"
                echo "  --models-only   Only download/update models"
                echo "  --docker-only   Only setup Docker containers"
                echo "  --dev           Setup for development"
                echo "  --prod          Setup for production"
                echo "  --help          Show this help message"
                exit 0
                ;;
        esac
    done
    
    check_prerequisites
    
    case $mode in
        "models")
            setup_docker
            setup_models
            ;;
        "docker")
            setup_docker
            ;;
        "dev")
            setup_environment
            setup_docker
            setup_models
            setup_frontend
            setup_backend
            health_check
            ;;
        "prod")
            setup_environment
            setup_docker
            setup_models
            health_check
            ;;
        *)
            setup_environment
            setup_docker
            setup_models
            setup_frontend
            setup_backend
            health_check
            ;;
    esac
    
    print_summary
}

# Run main function
main "$@"
