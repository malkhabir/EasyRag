
<h1 align="center"><img src="./finrag.png" alt="FinRag logo" style="max-width:200px; height:auto;"></h1>
<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-yellow.svg" alt="Python">
  <img src="https://img.shields.io/badge/react-18+-61DAFB.svg" alt="React">
  <img src="https://img.shields.io/badge/docker-ready-2496ED.svg" alt="Docker">
  <img src="https://img.shields.io/badge/plug%20%26%20play-models-purple.svg" alt="Plug & Play">
</p>

<p align="center">
  <strong>Enterprise-Grade Document Intelligence Platform</strong>
</p>

<p align="center">
  FinRag is a production-ready, modular RAG (Retrieval-Augmented Generation) system designed for financial document analysis.<br/>
  Extract tables, query documents with natural language, and get AI-powered answers with precise source highlighting.
</p>

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Smart Table Extraction** | AI-powered table detection using DIT/TADetect with Camelot for precise extraction |
| **Semantic Search** | BGE-M3 embeddings with 1024 dimensions for accurate document retrieval |
| **Conversational AI** | Natural language queries powered by local LLMs (Ollama) |
| **Source Highlighting** | Click any source to view the exact location in the PDF with visual highlighting |
| **Plug and Play Models** | Easily swap LLMs and embedding models via configuration |
| **Modular Architecture** | Clean separation of concerns for easy customization |

---

## Architecture

```
+------------------------------------------------------------------+
|                         Frontend (React)                          |
|  +------------+  +----------------+  +-------------------------+  |
|  | Upload     |  | Chat Window    |  | PDF Viewer + Highlight  |  |
|  +------------+  +----------------+  +-------------------------+  |
+------------------------------------------------------------------+
                              | REST API
+------------------------------------------------------------------+
|                      RAG Service (FastAPI)                        |
|  +-------------------+  +--------------+  +------------------+    |
|  | Document Builder  |  | Query Engine |  | File Service     |    |
|  | - Table Extract   |  | - Semantic   |  | - Upload/Delete  |    |
|  | - Text Extract    |  | - LLM Answer |  | - PDF Serving    |    |
|  +-------------------+  +--------------+  +------------------+    |
+------------------------------------------------------------------+
         |                      |                    |
    +----+----+           +-----+----+         +----+----+
    | Qdrant  |           | Ollama   |         | Models  |
    | Vector  |           | LLM      |         | BGE-M3  |
    | DB      |           |          |         |         |
    +---------+           +----------+         +---------+
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- 8GB+ RAM recommended

### One-Click Setup (Recommended)

**Windows:**
```powershell
.\setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

This will automatically:
- Check and validate prerequisites
- Start Docker containers (Qdrant, Ollama)
- Download AI models (LLM + Embeddings)
- Install all dependencies
- Configure environment

### Manual Setup

<details>
<summary>Click to expand manual setup instructions</summary>

#### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/FinRag.git
cd FinRag
```

#### 2. Start Infrastructure

```bash
# Start Qdrant vector database
docker-compose up -d qdrant ollama

# Pull your preferred LLM
docker-compose exec ollama ollama pull phi3
```

#### 3. Setup Backend

```bash
cd rag-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

#### 4. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

</details>

### 5. Open App

Navigate to http://localhost:5173

---

## Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | React web application |
| **Backend API** | http://localhost:8080 | FastAPI REST endpoints |
| **API Docs** | http://localhost:8080/docs | Swagger/OpenAPI documentation |
| **Qdrant Dashboard** | http://localhost:6335/dashboard | Vector database UI |
| **Ollama** | http://localhost:11435 | LLM API endpoint |

---

## Plug & Play Model Configuration

FinRag supports easy model swapping via YAML configuration. No code changes required!

### Model Configuration File

Edit `config/models.yaml` to customize your AI stack:

```yaml
# =============================================================================
# FinRag Models Configuration
# =============================================================================

# Active preset - switch between configurations instantly
active_preset: development  # Options: development, production, fast, accuracy

# -----------------------------------------------------------------------------
# LLM Models (Language Models for Q&A)
# -----------------------------------------------------------------------------
llm_providers:
  ollama:
    enabled: true
    host: localhost
    port: 11435
    models:
      phi3:           # Fast, efficient, great for tables
        context_window: 4096
        recommended_for: ["development", "fast-responses"]
      llama3:         # Better reasoning, larger context
        context_window: 8192
        recommended_for: ["production", "complex-queries"]
      mistral:        # Balanced performance
        context_window: 8192
        recommended_for: ["general-purpose"]
      mixtral:        # Highest quality, MoE architecture
        context_window: 32768
        recommended_for: ["accuracy", "long-documents"]

# -----------------------------------------------------------------------------
# Embedding Models (for semantic search)
# -----------------------------------------------------------------------------
embedding_models:
  bge-m3:            # Default: Best quality
    name: "BAAI/bge-m3"
    dimensions: 1024
    recommended_for: ["production", "accuracy"]
  minilm:            # Lightweight: Fast inference
    name: "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: 384
    recommended_for: ["development", "fast"]
  e5-large:          # Alternative: Strong performance
    name: "intfloat/e5-large-v2"
    dimensions: 1024
    recommended_for: ["accuracy"]

# -----------------------------------------------------------------------------
# Presets (Quick Configuration Profiles)
# -----------------------------------------------------------------------------
presets:
  development:
    llm: phi3
    embedding: bge-m3
    description: "Fast iteration, good quality"
  production:
    llm: llama3
    embedding: bge-m3
    description: "Optimized for quality and reliability"
  fast:
    llm: phi3
    embedding: minilm
    description: "Maximum speed, lower resource usage"
  accuracy:
    llm: mixtral
    embedding: bge-m3
    description: "Best quality, higher resource usage"
```

### Switching Models

**Option 1: Change Preset**
```yaml
# config/models.yaml
active_preset: production  # Switch from development to production
```

**Option 2: Environment Variables**
```bash
export OLLAMA_MODEL=llama3.1
export EMBEDDING_MODEL=BAAI/bge-m3
./setup.sh --models-only
```

**Option 3: Pull New Models**
```bash
# Pull a new LLM
docker-compose exec ollama ollama pull llama3.1

# Update config
# Edit config/models.yaml with the new model name
```

---

## Configuration

All configurations are centralized in `rag-service/app/core/config.py`. 
Use environment variables or edit directly.

### Model Configuration

```python
# rag-service/app/core/config.py

class Settings(BaseSettings):
    # ===========================================================
    # LLM CONFIGURATION - Swap your preferred model here
    # ===========================================================
    ollama_model: str = "phi3"          # Current: Fast, good at tables
    # ollama_model: str = "llama3"      # Alternative: Better reasoning
    # ollama_model: str = "mistral"     # Alternative: Balanced
    # ollama_model: str = "mixtral"     # Alternative: Best quality, slower
    
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_timeout: float = 120.0
    
    # ===========================================================
    # EMBEDDING CONFIGURATION - Swap your embedding model here
    # ===========================================================
    embedding_model_name: str = "BAAI/bge-m3"     # Current: 1024 dims
    # embedding_model_name: str = "BAAI/bge-large-en-v1.5"  # 1024 dims
    # embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
    
    embedding_model_cache_dir: str = "models/bge-m3"
    
    # ===========================================================
    # VECTOR DATABASE CONFIGURATION
    # ===========================================================
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "accounting_docs"
    qdrant_vector_size: int = 1024    # Must match embedding dimensions!
```

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Settings
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=phi3

# Vector DB Settings
QDRANT_HOST=localhost
QDRANT_PORT=6333

# App Settings
DEBUG=false
LOG_LEVEL=INFO
```

---

## Project Structure

```
FinRag/
|-- config/                      # Configuration files
|   +-- models.yaml              # Plug & Play model configuration
|
|-- frontend/                    # React + Vite frontend
|   |-- src/
|   |   |-- comps/              # Reusable components
|   |   |   |-- chat/           # ChatWindow, PDFViewer
|   |   |   |-- UploadWindow.jsx
|   |   |   +-- QueryForm.jsx
|   |   |-- pages/              # Page components
|   |   +-- index.css           # CSS variables and theming
|   +-- package.json
|
|-- rag-service/                 # Python FastAPI backend
|   |-- app/
|   |   |-- api/v1/             # API endpoints
|   |   |-- core/               # Config, logging
|   |   |-- db/                 # Qdrant client
|   |   |-- document_processing/ # Core extraction logic
|   |   |   |-- document_builder.py   # Orchestrator
|   |   |   |-- table_extraction.py   # Camelot + coordinates
|   |   |   |-- text_extraction.py    # Text with coordinates
|   |   |   +-- table_detection.py    # DIT/TADetect
|   |   |-- models/             # LLM and embedding wrappers
|   |   +-- services/           # Business logic
|   +-- requirements.txt
|
|-- setup.sh                    # Linux/macOS setup script
|-- setup.bat                   # Windows setup script
|-- docker-compose.yml          # Qdrant + Ollama
+-- README.md
```

---

## Roadmap

### Completed

- [x] PDF upload and processing
- [x] AI-powered table detection (DIT/TADetect)
- [x] Table extraction with precise coordinates
- [x] Text block extraction with coordinates
- [x] Semantic search with BGE-M3 embeddings
- [x] LLM-powered Q&A with Ollama
- [x] Source highlighting in PDF viewer
- [x] Multi-file navigation
- [x] CSS theming with variables
- [x] Icon system for customization
- [x] **Plug & Play Model Configuration** (YAML-based)
- [x] **One-Click Setup Scripts** (Windows + Linux/macOS)

### In Progress

- [ ] Model hot-swapping via API (runtime switching)
- [ ] Embedding model selection UI
- [ ] Batch document processing
- [ ] Config loader integration with Python backend

### Planned

- [ ] **Plugin System** - Drop-in extractors for different document types
- [ ] **Multi-tenant** - User isolation and access control
- [ ] **Cloud Deployment** - AWS/Azure/GCP templates + Terraform
- [ ] **API Keys** - Rate limiting and usage tracking
- [ ] **Webhook Support** - Notify external systems on events
- [ ] **Document Types** - Excel, Word, PowerPoint support
- [ ] **OCR Integration** - Tesseract/EasyOCR for scanned documents
- [ ] **Fine-tuning Pipeline** - Custom model training on your data
- [ ] **Analytics Dashboard** - Query patterns, usage metrics
- [ ] **Export** - PDF annotations, CSV/Excel export
- [ ] **OpenAI/Anthropic Providers** - Cloud LLM fallback
- [ ] **Model Benchmarking** - Compare model performance on your data

---

## Extending FinRag

### Adding a New LLM Provider

```python
# rag-service/app/models/llm.py

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: str) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, context: str) -> str:
        # Implement OpenAI API call
        pass

class AnthropicProvider(LLMProvider):
    def generate(self, prompt: str, context: str) -> str:
        # Implement Claude API call
        pass
```

### Adding a New Embedding Model

```python
# rag-service/app/models/embedding.py

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbeddings(EmbeddingProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Implement text-embedding-3-large
        pass
```

### Adding a New Document Type

```python
# rag-service/app/document_processing/extractors/excel_extractor.py

class ExcelExtractor(BaseExtractor):
    def extract(self, file_path: str) -> List[Document]:
        # Parse Excel sheets as tables
        pass
```

---

## Testing

```bash
cd rag-service
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

## Deployment

### Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### VS Code Tasks

Use the built-in VS Code tasks for quick actions:
- `Ctrl+Shift+P` → "Tasks: Run Task"
- Select from: Docker Start, RAG Service, Frontend Dev, etc.

### Manual

See [LAUNCH_INSTRUCTIONS.md](LAUNCH_INSTRUCTIONS.md) for detailed setup.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runtime
- [Qdrant](https://qdrant.tech/) - Vector database
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model
- [Camelot](https://camelot-py.readthedocs.io/) - PDF table extraction
- [DIT](https://github.com/microsoft/unilm/tree/master/dit) - Document layout detection
- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) - PDF rendering and low-level text extraction (uses `fitz`)

---
<p align="center">
  <a href="#-finrag">Back to top ↑</a>
</p>
