# FinRag Launch Instructions

## Quick Start (VS Code)

1. **Open VS Code** in this workspace
2. Go to **Run and Debug** (Ctrl+Shift+D)
3. Select **"Run Full Stack (RAG Pipeline + Frontend)"** from the dropdown
4. Click **Start Debugging** (F5)

This will launch:
- **RAG Pipeline** (FastAPI): http://localhost:8000
- **Frontend** (React/Vite): http://localhost:5173

## What Gets Launched

### RAG Pipeline (FastAPI)
- **Directory**: `rag-service/`
- **Virtual Environment**: `rag-service/venv/Scripts/python.exe`
- **Main file**: `rag-service/main.py`
- **Port**: 8000
- **Dependencies**: Already installed in venv (fastapi, uvicorn, llama-index, etc.)

### Frontend (React + Vite)
- **Directory**: `frontend/WebInterface/webinterface/`
- **Package Manager**: npm
- **Port**: 5173 (Vite default)
- **Dependencies**: Already installed in `node_modules/`

## Environment Variables (RAG Pipeline)

The RAG pipeline connects to Docker services:
- `QDRANT_HOST=localhost` on port `6335`
- `OLLAMA_HOST=localhost` on port `11435`

Make sure Docker services are running:
```powershell
docker compose up -d
```

## Manual Launch (Alternative)

### RAG Pipeline Only
```powershell
cd rag-service
.\venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Only
```powershell
cd frontend\WebInterface\webinterface
npm run dev
```

## Troubleshooting

- **RAG Pipeline won't start**: Ensure Docker containers (Qdrant, Ollama) are running: `docker compose ps`
- **Frontend won't start**: Run `npm install` in `frontend/WebInterface/webinterface/`
- **Port conflicts**: Check if ports 8000 or 5173 are already in use
