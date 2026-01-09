# AI Models Guide for EasyRag

This document covers the best free models available for document understanding, table detection, and embeddings.

---

## Embedding Models (for Vector Search)

### Currently Used: BGE-M3
- **Model**: `BAAI/bge-m3`
- **Dimensions**: 1024
- **Why it's excellent**: Multi-lingual, multi-granularity, state-of-the-art retrieval
- **MTEB Benchmark**: Top-tier performance

### Top Free Alternatives (Ranked)

| Model | Dimensions | MTEB Score | Best For | HuggingFace ID |
|-------|-----------|------------|----------|----------------|
| **BGE-large-en-v1.5** | 1024 | 64.23 | English-only, fastest | `BAAI/bge-large-en-v1.5` |
| **E5-large-v2** | 1024 | 62.5 | General purpose | `intfloat/e5-large-v2` |
| **E5-mistral-7b-instruct** | 4096 | 66.6 | Highest accuracy (requires GPU) | `intfloat/e5-mistral-7b-instruct` |
| **GTE-large** | 1024 | 63.1 | Multi-task | `thenlper/gte-large` |
| **UAE-Large-V1** | 1024 | 64.64 | Angle-optimized | `WhereIsAI/UAE-Large-V1` |
| **Nomic-embed-text-v1.5** | 768 | 62.3 | Fast, good quality | `nomic-ai/nomic-embed-text-v1.5` |
| **Jina-embeddings-v2** | 768 | 60.4 | Long context (8k tokens) | `jinaai/jina-embeddings-v2-base-en` |

### For Financial Documents (Specialized)
| Model | Description | HuggingFace ID |
|-------|-------------|----------------|
| **FinBERT** | Financial sentiment/NER | `ProsusAI/finbert` |
| **SecBERT** | SEC filings specialized | `nlpaueb/sec-bert-base` |

### Recommendation
- **Keep BGE-M3** - It's already top-tier and multi-lingual
- If English-only: Consider `BGE-large-en-v1.5` (slightly faster)
- For max accuracy with GPU: `E5-mistral-7b-instruct`

---

## LLM Models (for Query Generation)

### Currently Used: LLaMA2
- **Model**: `llama2` (via Ollama)
- **Size**: ~7B parameters (~3.5GB VRAM)
- **Speed**: Slow startup (21+ seconds), good quality
- **Issue**: Model unloads from memory, causing delays

### Top Free Alternatives (Ranked by Speed)

| Model | Size | Speed | Quality | Memory | Ollama Command |
|-------|------|-------|---------|---------|----------------|
| **Phi-3.5-mini** | 3.8B | Very Fast | Excellent | 2.3GB | `ollama pull phi3.5:latest` |
| **Phi-3-mini** | 3.8B | Very Fast | Excellent | 2.3GB | `ollama pull phi3:mini` |
| **LLaMA-3.2-1B** | 1B | Ultra Fast | Good | 0.8GB | `ollama pull llama3.2:1b` |
| **LLaMA-3.2-3B** | 3B | Fast | Very Good | 1.8GB | `ollama pull llama3.2:3b` |
| **Qwen2.5-3B** | 3B | Fast | Excellent | 1.8GB | `ollama pull qwen2.5:3b` |
| **Gemma2-2B** | 2B | Very Fast | Good | 1.2GB | `ollama pull gemma2:2b` |

### Recommendation for Speed
1. **Best Balance**: `phi3.5:latest` - Microsoft's latest, very fast, excellent quality
2. **Ultra Fast**: `llama3.2:1b` - Tiny but surprisingly capable
3. **Keep Current**: Add `keep_alive: 24h` to prevent unloading

### Quick Setup for Phi-3.5

```bash
# Pull the model
ollama pull phi3.5:latest

# Set keep-alive to prevent unloading
ollama create phi3.5-persistent <<EOF
FROM phi3.5:latest
PARAMETER keep_alive 24h
PARAMETER num_ctx 8192
EOF
```

Then update your config:
```python
ollama_model: str = "phi3.5-persistent"
```

---

## Table Detection Models

### Currently Used
1. **TADETECT** (`microsoft/table-transformer-detection`) - Good for finding table boundaries
2. **DIT** (`microsoft/dit-base-finetuned-rvlcdip`) - Document layout understanding

### Top Free Alternatives (Ranked)

| Model | Best For | Speed | Accuracy | HuggingFace ID |
|-------|----------|-------|----------|----------------|
| **Table Transformer Detection** | Table boundaries | Fast | High | `microsoft/table-transformer-detection` |
| **DiT-large** | Document layout | Medium | Very High | `microsoft/dit-large-finetuned-rvlcdip` |
| **LayoutLMv3** | Document understanding | Slow | Excellent | `microsoft/layoutlmv3-large` |
| **DETR-DocLayNet** | Complex layouts | Medium | Very High | `ds4sd/doctr-detr-base` |
| **YOLOv8-DocLayout** | Fast detection | Very Fast | Good | Custom training needed |

### For Table Structure Recognition
| Model | Description | HuggingFace ID |
|-------|-------------|----------------|
| **TATR-v1.1** | Table structure (rows/cols) | `microsoft/table-transformer-structure-recognition` |
| **TATR-v1.1-FinTabNet** | Financial tables specifically | `microsoft/table-transformer-structure-recognition-v1.1-fin` |
| **TableFormer** | Complex nested tables | `microsoft/tableformer-base` (private) |

### Recommendation
- **For financial docs**: Use `TATR-v1.1-FinTabNet` for structure recognition
- **For detection**: Current DIT + TADETECT combo is good
- **Consider adding**: `LayoutLMv3-large` for complex documents

---

## Document Layout Models

### Best Free Models

| Model | Use Case | HuggingFace ID |
|-------|----------|----------------|
| **DiT-large** | General document layout | `microsoft/dit-large-finetuned-rvlcdip` |
| **LayoutLMv3-large** | Document understanding + OCR | `microsoft/layoutlmv3-large` |
| **DocTR** | End-to-end OCR | `mindee/doctr` |
| **Donut** | Document understanding (no OCR) | `naver-clova-ix/donut-base` |
| **Pix2Struct** | Visual document QA | `google/pix2struct-base` |

---

## Configuration Examples

### Upgrading Embedding Model

```python
# In config.py, change:
embedding_model_name: str = "BAAI/bge-large-en-v1.5"  # For English-only
# OR
embedding_model_name: str = "intfloat/e5-large-v2"  # General purpose

# Update vector size accordingly:
qdrant_vector_size: int = 1024
```

### Adding LayoutLMv3 for Better Detection

```python
# In models.py, add:
from transformers import AutoProcessor, AutoModelForTokenClassification

layoutlmv3_processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-large",
    apply_ocr=False  # We already have OCR
)
layoutlmv3_model = AutoModelForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-large"
)
```

### Using Financial Table Transformer

```python
# Already available in models.py as TATR
# For better financial table recognition:
tatr_fin_processor = AutoImageProcessor.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-fin"
)
tatr_fin_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-fin"
)
```

---

## 📈 Performance Comparison

### Embedding Speed (tokens/sec on GPU)
| Model | Speed | Memory |
|-------|-------|--------|
| BGE-M3 | ~1000 | 2.5GB |
| BGE-large-en | ~1500 | 1.3GB |
| E5-large-v2 | ~1200 | 1.3GB |
| E5-mistral-7b | ~200 | 14GB |

### Table Detection Accuracy (on DocLayNet)
| Model | mAP | Speed |
|-------|-----|-------|
| TADETECT | 0.82 | Fast |
| DiT-large | 0.89 | Medium |
| LayoutLMv3-large | 0.92 | Slow |

---

## Recommended Stack for FAANG-Quality

### Current (Already Excellent)
- Embedding: `BAAI/bge-m3`
- Table Detection: `TADETECT + DIT`
- Table Structure: `TATR`

### Upgrade Path (If Needed)
1. **More Accuracy**: Add `LayoutLMv3-large` for layout
2. **Financial Focus**: Switch to `TATR-v1.1-FinTabNet`
3. **Faster**: Use `BGE-large-en-v1.5` for embedding
4. **Max Quality**: Use `E5-mistral-7b-instruct` (requires 16GB VRAM)

---

## 📚 Resources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding benchmarks
- [Papers With Code - Table Detection](https://paperswithcode.com/task/table-detection)
- [DocLayNet Dataset](https://github.com/DS4SD/DocLayNet) - Document layout benchmark
- [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) - Scientific documents

---

*Last updated: January 2026*
