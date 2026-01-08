"""Main FastAPI application entry point."""

# Suppress ML library warnings before any imports
import os
import warnings
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# Completely disable NLTK downloads by redirecting stdout during import
import io
from contextlib import redirect_stdout, redirect_stderr

# Monkey patch NLTK download before any imports
import nltk

original_download = nltk.download


def silent_download(*args, **kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return original_download(*args, **kwargs)


nltk.download = silent_download

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings

from app.core import get_settings, setup_logging, get_logger
from app.api.v1 import api_router
from app.api.v1.dependencies import (
    initialize_services,
    get_embedding_model,
    get_llm_model,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting FinRag RAG Service...")

    try:
        # Initialize all services
        initialize_services()

        # Configure LlamaIndex global settings
        embed_model = get_embedding_model()
        llm_model = get_llm_model()
        Settings.embed_model = embed_model.get()
        Settings.llm = llm_model.get()

        logger.info("Application ready - Access docs at http://localhost:8080/docs")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include API router
app.include_router(api_router, prefix=f"/api/{settings.api_version}")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.api_version,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
