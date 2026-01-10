

<p align="center">
  <img src="./finrag.png" alt="EasyRag logo" style="max-width:220px; height:100px;">
    <p style="font-size:30px">EasyRag</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/React-19.1-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Vite-4.5-646CFF?logo=vite&logoColor=white" alt="Vite">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LlamaIndex-RAG-FF6B6B?logo=llama&logoColor=white" alt="LlamaIndex">
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/PyTorch-ML-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-BGE--M3-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-000000?logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/Anthropic-Claude-191919?logo=anthropic&logoColor=white" alt="Anthropic">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

EasyRag is a modular Retrieval-Augmented Generation (RAG) platform that extracts structured data from PDF documents with high precision. Features a pluggable provider architecture supporting any combination of local and cloud AI services. This project can serve as a proof of concept or maybe more than that. If you are looking to understand how RAG works then this is the project for you.

<div style="flex:1;min-width:320px">
  <p><strong>Web Interface - Query & Highlight</strong></p>
  <img src="./docs/example.png" alt="Query response with source highlighting" style="max-width:100%;border:1px solid #ddd;padding:4px;" />
  <p style="font-size:90%">Ask questions about your uploaded documents and get AI-powered answers with precise source highlighting. The system identifies exactly where in the PDF the answer was found, enabling quick verification and traceability.</p>
</div>

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Provider Architecture](#provider-architecture)
- [Technology Stack](#technology-stack)
- [Vector Search & Embeddings](#vector-search--embeddings)
- [Table Detection Pipeline](#table-detection-pipeline)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [GPU Setup (Ollama)](#gpu-setup-ollama)
- [Testing & Deployment](#testing--deployment)
- [Developer Guide](#developer-guide)
- [Contributing](#contributing)
- [License](#license)

[↑ Back to top](#easyrag)

---

## Overview

- **Modular Architecture**: Pluggable provider system supporting Ollama, OpenAI, Anthropic, HuggingFace
- **Document Processing**: Extracts structured tables from PDFs with precise coordinate mapping
- **Semantic Search**: Vector embeddings indexed in Qdrant for fast, accurate retrieval
- **Runtime Flexibility**: Switch between providers via API without restart
- **Source Attribution**: LLM responses include precise document locations and highlighting

---

## Quick Start

**Prerequisites**: Python 3.11+, Node.js 18+, Docker Compose, 8GB+ RAM recommended

### Automated Setup (Windows)

```powershell
.\setup.bat
```

### Manual Setup

```bash
git clone https://github.com/malkhabir/EasyRag.git
cd EasyRag

# Start infrastructure
docker compose up -d qdrant ollama

# Backend
cd rag-service
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

**Access Points**:
- Frontend: http://localhost:5173
- API Documentation: http://localhost:8080/docs

---

## Provider Architecture

EasyRag features a modular provider system that supports any combination of local and cloud AI services. Switch providers at runtime via API without restarting the application.

### Supported Providers

| Type | Provider | Models |
|------|----------|--------|
| **LLM** | Ollama (local) | `phi3`, `llama2`, `codellama` |
| | OpenAI | `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo` |
| | Anthropic | `claude-3-sonnet`, `claude-3-haiku` |
| | Azure OpenAI | Enterprise deployments |
| **Embedding** | HuggingFace (local) | `BAAI/bge-m3`, `all-MiniLM-L6-v2` |
| | OpenAI | `text-embedding-3-small`, `text-embedding-3-large` |

### Provider Configuration

Configure providers in `config/providers.yaml`:

```yaml
active_llm_provider: "local"
active_embedding_provider: "huggingface"

llm_providers:
  local:
    provider: "ollama"
    model_name: "phi3"
    host: "localhost"
    port: 11434
    
  openai:
    provider: "openai" 
    model_name: "gpt-3.5-turbo"
    # api_key: ${OPENAI_API_KEY}
```

### Runtime Provider Switching

```bash
# List available providers
curl http://localhost:8080/api/v1/providers/llm

# Switch LLM provider
curl -X POST http://localhost:8080/api/v1/providers/llm/switch \
  -H "Content-Type: application/json" \
  -d '{"provider_name": "openai", "model_name": "gpt-4"}'

# Check system status
curl http://localhost:8080/api/v1/providers/status
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | React | 19.1.1 | UI component library |
| | Vite | 4.5.0 | Build tool and dev server |
| | react-pdf | 10.0.1 | PDF rendering in browser |
| | pdfjs-dist | 5.3.31 | Mozilla's PDF.js for parsing |
| **Backend** | Python | 3.11+ | Runtime environment |
| | FastAPI | 0.104.0+ | Async REST API framework |
| | Uvicorn | latest | ASGI server |
| | Pydantic | 2.0.0+ | Data validation and settings |
| **AI/ML** | LlamaIndex | latest | RAG orchestration framework |
| | PyTorch | latest | Deep learning framework |
| | Transformers | latest | HuggingFace model hub |
| | sentence-transformers | latest | Embedding models |
| | Detectron2 | 0.6+ | Object detection (layout) |
| **Document Detection** | [DIT](https://huggingface.co/nevernever69/dit-doclaynet-segmentation) | latest | Semantic segmentation (page layout) |
| | [TADetect](https://huggingface.co/microsoft/table-transformer-detection) | latest | Object detection (table regions) |
| | [TATR](https://huggingface.co/microsoft/table-transformer-structure-recognition) | latest | Table structure recognition |
| **Vector DB** | Qdrant | latest | Vector similarity search |
| **LLM Providers** | Ollama | latest | Local LLM runtime |
| | OpenAI API | 1.0.0+ | Cloud LLM (optional) |
| | Anthropic API | 0.3.0+ | Claude models (optional) |
| **PDF Processing** | PyMuPDF (fitz) | latest | PDF rasterization |
| | pdfplumber | latest | Text extraction |
| | camelot-py | latest | Table extraction |
| | pytesseract | latest | OCR fallback |
| **Infrastructure** | Docker Compose | 3.9 | Container orchestration |

---

## Vector Search & Embeddings

EasyRag uses vector indexes to enable semantic search over documents.

### How It Works

```
Traditional Search: "revenue" → matches documents containing "revenue"
Vector Search: "how much money did we make" → matches documents about revenue, income, earnings, etc.
```

1. **Embedding Generation**: Text chunks are converted to 1024-dimensional vectors using BGE-M3
2. **Vector Storage**: Vectors are stored in Qdrant with metadata (page, coordinates, source file)
3. **Similarity Search**: Queries are embedded and compared using cosine similarity
4. **Top-K Retrieval**: Most similar chunks are retrieved and passed to the LLM

### Qdrant Configuration

| Feature | Value |
|---------|-------|
| Index Type | HNSW (Hierarchical Navigable Small World) |
| Distance Metric | Cosine Similarity |
| Vector Dimensions | 1024 (BGE-M3) |
| Storage | Persistent on disk with in-memory indexing |

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1024,           # BGE-M3 embedding dimensions
        distance=Distance.COSINE
    )
)
```

### Similarity Metrics

| Metric | Range | Best For |
|--------|-------|----------|
| **Cosine** | 0 to 2 | Text embeddings, semantic similarity |
| **Dot Product** | -∞ to +∞ | Normalized vectors, recommendations |
| **Euclidean** | 0 to +∞ | Image features, spatial data |

**Why Cosine?** Measures the angle between vectors, ignoring magnitude. This means "revenue report" and "income statement" will score as highly similar even if one document is longer than the other.

### Embedding Model: BGE-M3

| Property | Value |
|----------|-------|
| Model | BAAI/bge-m3 |
| Dimensions | 1024 |
| Max Tokens | 8192 |
| Languages | 100+ (multilingual) |
| Features | Dense + Sparse + ColBERT retrieval |

### LlamaIndex Integration

EasyRag uses [LlamaIndex](https://www.llamaindex.ai/) as the core RAG framework:

```python
from llama_index.core import Settings, VectorStoreIndex

# Configure embeddings and LLM
Settings.embed_model = embed_model.get()  # HuggingFace BGE-M3
Settings.llm = llm_model.get()            # Ollama phi3/llama2

# Create vector store and index
vector_store = QdrantVectorStore(client=qdrant_client, collection_name="documents")
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Query with similarity search
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is the total revenue?")
```

---

## Table Detection Pipeline

EasyRag uses a multi-model pipeline for accurate table extraction from PDFs. The system recursively detects nested tables using a tree-based approach, treating each document page as a hierarchy of table containers and atomic tables (leaves).

### Detection Models

The pipeline uses three primary detection models from HuggingFace. Each serves a specific purpose in the detection hierarchy.

#### Active Models

| Model | Type | How It Works | Best For | HuggingFace Link |
|-------|------|--------------|----------|------------------|
| **DIT** | Semantic Segmentation | Classifies **every pixel** into categories (table, text, figure, etc.), then finds contours | Full-page layout, leaf validation | [nevernever69/dit-doclaynet-segmentation](https://huggingface.co/nevernever69/dit-doclaynet-segmentation) |
| **TADetect** | Object Detection | Predicts **bounding boxes** with confidence scores | Fast table region detection | [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) |
| **TATR** | Object Detection | Detects table **cells, rows, columns** within a table | Table structure recognition | [microsoft/table-transformer-structure-recognition](https://huggingface.co/microsoft/table-transformer-structure-recognition) |

#### Available Models (Not Yet Integrated)

| Model | Type | Purpose | Source |
|-------|------|---------|--------|
| **Detectron2** | Object Detection | Layout detection using Faster R-CNN | [LayoutParser Model Zoo](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html) |
| **LayoutLMv3** | Token Classification | Form understanding with text + layout | [nielsr/layoutlmv3-finetuned-funsd](https://huggingface.co/nielsr/layoutlmv3-finetuned-funsd) |
| **Donut** | Vision Encoder-Decoder | End-to-end document understanding | [naver-clova-ix/donut-base-finetuned-cord-v2](https://huggingface.co/naver-clova-ix/donut-base-finetuned-cord-v2) |

### Recursive Table Detection (Tree Structure)

Documents with complex layouts often contain **nested tables** - tables within tables. EasyRag handles this by building a tree structure:

```
Page (Root)
├── Table A (Container) ──► DIT finds 2 sub-tables
│   ├── Table A.1 (Leaf) ──► DIT finds 0-1 tables, confirmed atomic
│   └── Table A.2 (Leaf) ──► DIT finds 0-1 tables, confirmed atomic
├── Table B (Leaf) ──► DIT finds 0-1 tables, already atomic
└── Text Block (ignored)
```

**Key Concepts:**
- **Container**: A table region that contains other tables inside it
- **Leaf**: An atomic table with no sub-tables (ready for text extraction)
- **Depth**: How many levels deep in the tree (0 = page level, 1 = first nesting, etc.)

**Algorithm:**
1. **Initial Detection**: Run DIT on the full page to find all table regions
2. **Recursive Descent**: For each detected table, run DIT again on the cropped region
3. **Leaf Validation**: If DIT finds 2+ tables inside, it's a container → recurse deeper
4. **Stopping Criteria**: Stop recursing when sub-tables are too small or too many are detected
5. **Output**: Only leaf tables (atomic units) are sent for text extraction

### Stopping Criteria (Tunable Constants)

Without stopping criteria, DIT would recurse forever, eventually detecting individual cells as "tables". These constants control when to stop:

| Constant | Default | Description |
|----------|---------|-------------|
| `SELF_DETECTION_RATIO` | 0.95 | Skip if sub-table is ≥95% of parent (self-detection) |
| `MIN_TABLE_WIDTH_PX` | 80px | Skip sub-tables narrower than this |
| `MIN_TABLE_HEIGHT_PX` | 50px | Skip sub-tables shorter than this |
| `MIN_TABLE_AREA_PX` | 4000px² | Skip tiny fragments |
| `MIN_SUBTABLE_AREA_RATIO` | 5% | Skip if sub-table is <5% of parent area |
| `MAX_ASPECT_RATIO` | 8.0 | Skip extreme strips (likely rows/columns, not tables) |
| `MAX_SUBTABLES_PER_REGION` | 10 | If >10 detected, it's cells not tables → stop |
| `MAX_RECURSIVE_DEPTH` | 5 | Maximum tree depth to prevent infinite recursion |

**Tuning Guide:**
- **DIT finds cells instead of tables?** → Increase `MIN_TABLE_WIDTH_PX`, `MIN_TABLE_AREA_PX`
- **DIT misses valid sub-tables?** → Decrease those values
- **Rows/columns detected as tables?** → Decrease `MAX_ASPECT_RATIO`
- **Too many fragments?** → Decrease `MAX_SUBTABLES_PER_REGION`

### Segmentation vs Object Detection

| Aspect | Segmentation (DIT) | Object Detection (TADetect) |
|--------|-------------------|---------------------------|
| **Output** | Pixel mask (each pixel gets a class label) | Bounding boxes with confidence scores |
| **Precision** | Exact boundaries at pixel level | Rectangular boxes only |
| **Context Needed** | Requires full document layout context | Works on isolated crops |
| **Speed** | Slower (processes all pixels) | Faster (sparse predictions) |
| **Use Case** | Page-level detection, leaf validation | Fast initial scanning |

**Why DIT for leaf validation?** DIT consistently finds 2 tables when a region contains nested tables, and 0-1 when it's truly atomic. This binary signal is reliable for determining container vs leaf status.

### Pipeline Flow

```
PDF Page
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Initial Detection (DIT on full page)             │
│  - Finds all table regions with full document context      │
│  - Output: List of candidate table boxes                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Recursive Nesting Detection                      │
│  - For each table, crop region and run DIT again           │
│  - If 2+ sub-tables found → mark as container, recurse     │
│  - If 0-1 sub-tables → mark as leaf, stop                  │
│  - Apply stopping criteria to prevent over-segmentation    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Leaf Validation                                  │
│  - Final DIT pass on each "leaf" to confirm it's atomic    │
│  - If DIT finds 2+ tables → split and re-validate          │
│  - Ensures no nested tables are missed                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Text Extraction & Embedding                      │
│  - Only leaf tables are processed                          │
│  - Extract text via Camelot/pdfplumber                     │
│  - Generate embeddings, store in Qdrant                    │
└─────────────────────────────────────────────────────────────┘
```

### Configuration File

All detection parameters are in `rag-service/app/document_processing/constants.py`:

```python
# Detection model selection
NESTED_DETECTION_MODEL = "dit_only"  # "dit_only", "tadetect_only", or "both"
VALIDATE_LEAVES_WITH_DIT = True      # Final validation pass

# Confidence thresholds
TADETECT_TABLE_CONF_THRESHOLD = 0.15  # Lower = more sensitive
TATR_TABLE_CONF_THRESHOLD = 0.3

# Stopping criteria
MIN_TABLE_WIDTH_PX = 80
MIN_TABLE_HEIGHT_PX = 50
MIN_TABLE_AREA_PX = 4000
MAX_ASPECT_RATIO = 8.0
MAX_SUBTABLES_PER_REGION = 10
MAX_RECURSIVE_DEPTH = 5
```

---

## API Reference

### Document Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload a PDF document |
| `GET` | `/api/v1/files` | List all uploaded files |
| `GET` | `/api/v1/files/{filename}` | Download/view a specific file |
| `DELETE` | `/api/v1/files/{filename}` | Delete a file |

### Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/query?q={query}` | Query documents with natural language |
| `GET` | `/api/v1/query?q={query}&files={file1}` | Query specific files |

### Provider Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/providers/llm` | List available LLM providers |
| `GET` | `/api/v1/providers/embedding` | List embedding providers |
| `POST` | `/api/v1/providers/llm/switch` | Switch LLM provider |
| `POST` | `/api/v1/providers/embedding/switch` | Switch embedding provider |
| `GET` | `/api/v1/providers/status` | System health status |

Full API documentation: http://localhost:8080/docs

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env`:

```bash
# Active Providers
EASYRAG_ACTIVE_LLM_PROVIDER=local
EASYRAG_ACTIVE_EMBEDDING_PROVIDER=huggingface

# API Keys (for cloud providers)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Local Ollama Configuration
EASYRAG_LLM_PROVIDERS__LOCAL__HOST=localhost
EASYRAG_LLM_PROVIDERS__LOCAL__PORT=11434
```

---

## GPU Setup (Ollama)

For GPU-accelerated local LLMs, ensure proper NVIDIA configuration.

### Prerequisites

```bash
# Verify GPU on host
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Linux Installation

```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-535

# Install Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

### Windows (WSL2)

Install the NVIDIA WSL driver, enable WSL2 backend in Docker Desktop, and test `nvidia-smi` inside WSL.

### Docker Compose Configuration

```yaml
services:
  ollama:
    image: ollama/ollama:latest-gpu
    device_requests:
      - driver: nvidia
        count: all
        capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### Troubleshooting

| Error | Solution |
|-------|----------|
| `CUDA driver version is insufficient` | Update NVIDIA driver on host |
| `could not select device driver` | Install `nvidia-docker2` or enable WSL2 GPU |
| Docker test fails but host works | Restart Docker after installing toolkit |

---

## Testing & Deployment

### Running Tests

```bash
cd rag-service
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### VS Code Tasks

Use built-in tasks via `Ctrl+Shift+P` → "Tasks: Run Task":
- Docker: Start Services
- RAG Service: Run Server
- Frontend: Run Dev Server

See [LAUNCH_INSTRUCTIONS.md](LAUNCH_INSTRUCTIONS.md) for detailed setup.

---

## Developer Guide

### Project Structure

```
rag-service/
├── app/
│   ├── api/v1/           # REST endpoints
│   ├── core/             # Config, logging
│   ├── db/               # Qdrant client
│   ├── document_processing/  # PDF extraction
│   ├── models/           # LLM & embedding wrappers
│   └── services/         # Business logic
└── tests/
```

### Adding a New LLM Provider

```python
# rag-service/app/models/providers/my_provider.py
from .base import LLMProvider

class MyProvider(LLMProvider):
    def generate(self, prompt: str, context: str) -> str:
        # Implement API call
        pass
```

### Adding a New Embedding Model

```python
# rag-service/app/models/embedding.py
class MyEmbeddings(EmbeddingProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Return embedding vectors
        pass
```

### Technical Notes

- Maintain consistent embedding dimensions (1024 for BGE-M3) across pipelines
- Use row-level granularity for table embeddings for optimal retrieval
- Store both pixel and PDF coordinates for flexible highlighting

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m "Add amazing feature"`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runtime
- [Qdrant](https://qdrant.tech/) - Vector database
- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model
- [Camelot](https://camelot-py.readthedocs.io/) - PDF table extraction
- [DIT](https://github.com/microsoft/unilm/tree/master/dit) - Document layout detection
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF rendering

---

<p align="center">
  <a href="#easyrag">↑ Back to top</a>
</p>
