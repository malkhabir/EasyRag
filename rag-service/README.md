# FinRag RAG Service - Production Structure

Clean, production-ready directory structure following FAANG best practices.

## Directory Structure

```
rag-service/
├── app/                    # Application package
│   ├── api/               # API layer
│   │   └── v1/           # API version 1
│   ├── core/             # Core configuration
│   ├── db/               # Database operations
│   ├── models/           # ML models
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── docs/                 # Documentation
├── helper/              # Helper modules (detectron2, iopath)
├── models/              # Model cache directory
├── scripts/             # Build/deployment scripts
├── src/                 # Additional source (BlockDetector)
├── static/              # Static assets
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── uploaded_pdfs/       # File uploads
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
```

## Clean Architecture Benefits

- **Separation of Concerns** - Each module has single responsibility  
- **Testability** - Easy to mock and test components  
- **Scalability** - Easy to add features without breaking existing code  
- **Maintainability** - Clear structure for new developers  
- **Dependency Injection** - Loose coupling between components  

## File Organization

- **Root level** - Only essential files (main.py, requirements.txt, .env)
- **app/** - All application code in organized packages
- **tests/** - Mirror structure of app/ for test files
- **docs/** - All documentation
- **scripts/** - Deployment, build, migration scripts
- **helper/** - Third-party or legacy helper code

## Running the Application

### Using helper script (Recommended):

**Windows:**
```powershell
.\run.ps1 uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

**Linux/Mac:**
```bash
./run.sh uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Manual (with venv):

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Run server
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Access API docs at: `http://localhost:8080/docs`

## Development Setup

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development instructions.
