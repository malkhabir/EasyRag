@echo off
REM ==============================================================================
REM EasyRag Setup Script - Windows
REM ==============================================================================
REM This script sets up the complete EasyRag environment:
REM   1. Checks prerequisites (Docker, Python, Node.js)
REM   2. Pulls and configures Docker containers
REM   3. Downloads required ML models
REM   4. Installs dependencies
REM   5. Initializes the database
REM
REM Usage:
REM   setup.bat [options]
REM
REM Options:
REM   --full          Full setup (default)
REM   --models-only   Only download/update models
REM   --docker-only   Only setup Docker containers
REM   --dev           Setup for development
REM   --prod          Setup for production
REM ==============================================================================

setlocal EnableDelayedExpansion

REM Configuration
set "SCRIPT_DIR=%~dp0"
set "CONFIG_FILE=%SCRIPT_DIR%config\models.yaml"
set "ENV_FILE=%SCRIPT_DIR%.env"

REM Default model settings
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=phi3"
if not defined EMBEDDING_MODEL set "EMBEDDING_MODEL=BAAI/bge-m3"

REM Colors (Windows 10+)
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "PURPLE=[95m"
set "NC=[0m"

REM ==============================================================================
REM Main Entry Point
REM ==============================================================================

:main
call :print_banner

REM Parse arguments
set "MODE=full"
if "%1"=="--models-only" set "MODE=models"
if "%1"=="--docker-only" set "MODE=docker"
if "%1"=="--dev" set "MODE=dev"
if "%1"=="--prod" set "MODE=prod"
if "%1"=="--help" goto :show_help

call :check_prerequisites
if errorlevel 1 goto :eof

if "%MODE%"=="models" (
    call :setup_docker
    call :setup_models
    goto :print_summary
)
if "%MODE%"=="docker" (
    call :setup_docker
    goto :print_summary
)
if "%MODE%"=="dev" (
    call :setup_environment
    call :setup_docker
    call :setup_models
    call :setup_frontend
    call :setup_backend
    call :health_check
    goto :print_summary
)
if "%MODE%"=="prod" (
    call :setup_environment
    call :setup_docker
    call :setup_models
    call :health_check
    goto :print_summary
)

REM Default: full setup
call :setup_environment
call :setup_docker
call :setup_models
call :setup_frontend
call :setup_backend
call :health_check
goto :print_summary

REM ==============================================================================
REM Helper Functions
REM ==============================================================================

:print_banner
echo.
echo %PURPLE%=========================================================================%NC%
echo %PURPLE%                                                                         %NC%
echo %PURPLE%   ███████╗██╗███╗   ██╗██████╗  █████╗  ██████╗                        %NC%
echo %PURPLE%   ██╔════╝██║████╗  ██║██╔══██╗██╔══██╗██╔════╝                        %NC%
echo %PURPLE%   █████╗  ██║██╔██╗ ██║██████╔╝███████║██║  ███╗                       %NC%
echo %PURPLE%   ██╔══╝  ██║██║╚██╗██║██╔══██╗██╔══██║██║   ██║                       %NC%
echo %PURPLE%   ██║     ██║██║ ╚████║██║  ██║██║  ██║╚██████╔╝                       %NC%
echo %PURPLE%   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                        %NC%
echo %PURPLE%                                                                         %NC%
echo %PURPLE%   Financial Document RAG System - Setup Script                          %NC%
echo %PURPLE%                                                                         %NC%
echo %PURPLE%=========================================================================%NC%
echo.
exit /b 0

:log_info
echo %BLUE%[INFO]%NC% %~1
exit /b 0

:log_success
echo %GREEN%[SUCCESS]%NC% %~1
exit /b 0

:log_warning
echo %YELLOW%[WARNING]%NC% %~1
exit /b 0

:log_error
echo %RED%[ERROR]%NC% %~1
exit /b 0

:log_step
echo.
echo %CYAN%=========================================================================%NC%
echo %CYAN%  %~1%NC%
echo %CYAN%=========================================================================%NC%
echo.
exit /b 0

REM ==============================================================================
REM Prerequisite Checks
REM ==============================================================================

:check_prerequisites
call :log_step "Checking Prerequisites"

set "MISSING="

REM Check Docker
where docker >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('docker --version') do call :log_success "Docker: %%i"
) else (
    set "MISSING=!MISSING! docker"
    call :log_error "Docker is not installed"
)

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorlevel%==0 (
    call :log_success "Docker Compose: Available"
) else (
    where docker-compose >nul 2>&1
    if %errorlevel%==0 (
        call :log_success "Docker Compose: Available"
    ) else (
        set "MISSING=!MISSING! docker-compose"
        call :log_error "Docker Compose is not installed"
    )
)

REM Check Python
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('python --version') do call :log_success "Python: %%i"
) else (
    call :log_warning "Python not found (optional for Docker deployment)"
)

REM Check Node.js
where node >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('node --version') do call :log_success "Node.js: %%i"
) else (
    call :log_warning "Node.js not found (optional for Docker deployment)"
)

REM Check curl
where curl >nul 2>&1
if %errorlevel%==0 (
    call :log_success "curl: Available"
) else (
    set "MISSING=!MISSING! curl"
    call :log_error "curl is not installed"
)

if not "!MISSING!"=="" (
    echo.
    call :log_error "Missing required tools:!MISSING!"
    echo.
    echo Please install the missing tools:
    echo   - Docker: https://docs.docker.com/get-docker/
    echo   - Docker Compose: https://docs.docker.com/compose/install/
    exit /b 1
)

call :log_success "All prerequisites satisfied!"
exit /b 0

REM ==============================================================================
REM Docker Setup
REM ==============================================================================

:setup_docker
call :log_step "Setting Up Docker Containers"

cd /d "%SCRIPT_DIR%"

REM Create necessary directories
call :log_info "Creating data directories..."
if not exist "qdrant\storage" mkdir "qdrant\storage"
if not exist "rag-service\models" mkdir "rag-service\models"
if not exist "rag-service\uploaded_pdfs" mkdir "rag-service\uploaded_pdfs"
if not exist "rag-service\nltk_data" mkdir "rag-service\nltk_data"

REM Pull latest images
call :log_info "Pulling Docker images..."
docker-compose pull

REM Start core services
call :log_info "Starting Qdrant vector database..."
docker-compose up -d qdrant

call :log_info "Starting Ollama LLM service..."
docker-compose up -d ollama

REM Wait for services
call :log_info "Waiting for services to initialize..."
timeout /t 5 /nobreak >nul

REM Check services
docker-compose ps | findstr /i "qdrant.*Up" >nul
if %errorlevel%==0 (
    call :log_success "Qdrant is running on port 6335"
) else (
    call :log_error "Qdrant failed to start"
    docker-compose logs qdrant
    exit /b 1
)

docker-compose ps | findstr /i "ollama.*Up" >nul
if %errorlevel%==0 (
    call :log_success "Ollama is running on port 11435"
) else (
    call :log_error "Ollama failed to start"
    docker-compose logs ollama
    exit /b 1
)

exit /b 0

REM ==============================================================================
REM Model Setup
REM ==============================================================================

:setup_models
call :log_step "Setting Up AI Models"

REM Pull Ollama model
call :log_info "Pulling LLM model: %OLLAMA_MODEL%..."

docker-compose exec -T ollama ollama pull %OLLAMA_MODEL%
if %errorlevel%==0 (
    call :log_success "LLM model '%OLLAMA_MODEL%' is ready!"
) else (
    call :log_warning "Could not pull model via Docker."
    where ollama >nul 2>&1
    if %errorlevel%==0 (
        set "OLLAMA_HOST=http://localhost:11435"
        ollama pull %OLLAMA_MODEL%
    ) else (
        call :log_error "Please run: ollama pull %OLLAMA_MODEL%"
    )
)

REM Download embedding model
call :log_info "Downloading embedding model: %EMBEDDING_MODEL%..."

where python >nul 2>&1
if %errorlevel%==0 (
    python -c "from sentence_transformers import SentenceTransformer; import os; cache_dir = os.path.join('%SCRIPT_DIR%', 'rag-service', 'models'); os.makedirs(cache_dir, exist_ok=True); SentenceTransformer('%EMBEDDING_MODEL%', cache_folder=cache_dir); print('Model downloaded!')" 2>nul
    if %errorlevel%==0 (
        call :log_success "Embedding model downloaded!"
    ) else (
        call :log_warning "Embedding model will download on first run."
    )
) else (
    call :log_info "Python not available. Model will download on first run."
)

call :log_success "Model setup complete!"
exit /b 0

REM ==============================================================================
REM Environment Setup
REM ==============================================================================

:setup_environment
call :log_step "Configuring Environment"

REM Create config directory
if not exist "%SCRIPT_DIR%config" mkdir "%SCRIPT_DIR%config"

REM Create .env file if it doesn't exist
if not exist "%ENV_FILE%" (
    call :log_info "Creating .env file..."
    (
        echo # ==============================================================================
        echo # EasyRag Environment Configuration
        echo # ==============================================================================
        echo.
        echo # LLM Configuration
        echo OLLAMA_HOST=localhost
        echo OLLAMA_PORT=11435
        echo OLLAMA_MODEL=%OLLAMA_MODEL%
        echo.
        echo # Embedding Configuration
        echo EMBEDDING_MODEL=%EMBEDDING_MODEL%
        echo.
        echo # Vector Database
        echo QDRANT_HOST=localhost
        echo QDRANT_PORT=6335
        echo.
        echo # API Keys ^(uncomment and set if using cloud providers^)
        echo # OPENAI_API_KEY=sk-your-key-here
        echo # ANTHROPIC_API_KEY=your-key-here
        echo.
        echo # Debug Mode
        echo DEBUG=false
        echo LOG_LEVEL=INFO
    ) > "%ENV_FILE%"
    call :log_success "Created .env file"
) else (
    call :log_info ".env file already exists, skipping..."
)

call :log_success "Environment configured!"
exit /b 0

REM ==============================================================================
REM Frontend Setup
REM ==============================================================================

:setup_frontend
call :log_step "Setting Up Frontend"

cd /d "%SCRIPT_DIR%frontend"

where npm >nul 2>&1
if %errorlevel%==0 (
    call :log_info "Installing frontend dependencies..."
    npm install
    call :log_success "Frontend dependencies installed!"
) else (
    call :log_warning "npm not found. Skipping frontend setup."
    call :log_info "To setup frontend manually: cd frontend ^&^& npm install"
)

cd /d "%SCRIPT_DIR%"
exit /b 0

REM ==============================================================================
REM Backend Setup
REM ==============================================================================

:setup_backend
call :log_step "Setting Up Backend"

cd /d "%SCRIPT_DIR%rag-service"

where python >nul 2>&1
if %errorlevel%==0 (
    if not exist "venv" (
        call :log_info "Creating Python virtual environment..."
        python -m venv venv
    )
    
    call :log_info "Installing Python dependencies..."
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
    call deactivate
    
    call :log_success "Backend dependencies installed!"
) else (
    call :log_warning "Python not found. Skipping local backend setup."
    call :log_info "Backend will run in Docker container."
)

cd /d "%SCRIPT_DIR%"
exit /b 0

REM ==============================================================================
REM Health Check
REM ==============================================================================

:health_check
call :log_step "Running Health Checks"

REM Check Qdrant
call :log_info "Checking Qdrant..."
curl -s "http://localhost:6335/health" | findstr /i "ok true" >nul
if %errorlevel%==0 (
    call :log_success "Qdrant: Healthy"
) else (
    call :log_warning "Qdrant: Not responding (may still be initializing)"
)

REM Check Ollama
call :log_info "Checking Ollama..."
curl -s "http://localhost:11435/api/tags" >nul 2>&1
if %errorlevel%==0 (
    call :log_success "Ollama: Healthy"
) else (
    call :log_warning "Ollama: Not responding (may still be initializing)"
)

exit /b 0

REM ==============================================================================
REM Print Summary
REM ==============================================================================

:print_summary
echo.
echo %GREEN%=========================================================================%NC%
echo %GREEN%                    Setup Complete! 🎉                                   %NC%
echo %GREEN%=========================================================================%NC%
echo.
echo %CYAN%Quick Start:%NC%
echo   1. Start all services:     docker-compose up -d
echo   2. Start frontend:         cd frontend ^&^& npm run dev
echo   3. Start backend:          cd rag-service ^&^& run.bat
echo.
echo %CYAN%Service URLs:%NC%
echo   * Frontend:    http://localhost:5173
echo   * Backend API: http://localhost:8080
echo   * Qdrant:      http://localhost:6335
echo   * Ollama:      http://localhost:11435
echo.
echo %CYAN%Configuration:%NC%
echo   * Models:      config\models.yaml
echo   * Environment: .env
echo.
echo %CYAN%Useful Commands:%NC%
echo   * View logs:          docker-compose logs -f
echo   * Stop services:      docker-compose down
echo   * Pull new model:     docker-compose exec ollama ollama pull ^<model^>
echo   * Swap LLM:           Edit config\models.yaml and restart
echo.
echo %YELLOW%Need help? Check the README.md or open an issue on GitHub.%NC%
echo.
exit /b 0

:show_help
echo Usage: setup.bat [options]
echo.
echo Options:
echo   --full          Full setup (default)
echo   --models-only   Only download/update models
echo   --docker-only   Only setup Docker containers
echo   --dev           Setup for development
echo   --prod          Setup for production
echo   --help          Show this help message
exit /b 0

endlocal
