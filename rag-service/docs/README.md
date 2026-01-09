# EasyRag RAG Service

Production-grade RAG (Retrieval-Augmented Generation) service for financial document processing.

## Architecture

```
rag-service/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── dependencies.py
│   │       └── endpoints.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── qdrant.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── llm.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── file_service.py
│   │   ├── indexing_service.py
│   │   └── query_service.py
│   └── utils/
├── tests/
│   ├── unit/
│   └── integration/
├── main.py
├── pdf_utils.py
└── requirements.txt
```

## Features

- Clean architecture with separation of concerns
- Dependency injection for better testability
- Centralized configuration with Pydantic Settings
- Structured logging
- RESTful API with versioning
- Type hints throughout
- Comprehensive error handling

## Running the Application

```bash
cd rag-service
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/v1/upload` - Upload PDF file
- `GET /api/v1/files` - List all files
- `GET /api/v1/files/{filename}` - Get specific file
- `DELETE /api/v1/files/{filename}` - Delete file
- `GET /api/v1/query` - Query documents

## Configuration

Configure via environment variables or `.env` file:
- `QDRANT_HOST`
- `QDRANT_PORT`
- `OLLAMA_HOST`
- `OLLAMA_PORT`
