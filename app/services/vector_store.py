"""
VECTOR STORE SERVICE MODULE
===========================

This services builds and queries the FAISS vector index used for context retrieval.
Learning data (database/learning_data/*.txt) and past chats (database/chats_data/*.json)
are loaded at startup, split into chunks, embedded with HuggingFace, and stored in FAISS.
When the user asks a question we emded it retrieve the k most similar chunks; only
those chunks are sent to the LLM, so token usage is bounded.

LIFECYCLE:
  - create_vector_store(): Load all .txt and .json, chunk, embed, build FAISS, save to dist.
    Called once at srartup. Restart the server after adding new .txt files so they are included.
  - get_retriever(k): Return a retriever that fetches k nearest chunks for a query string.
  -save_vector_store(): Write the current FAISS index to database/vector_store/ (called after create).

Embeddings run locally (sentence-transforms); no extra API key. Groq and Realtime services
call get_retriever() for every request to get context.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstore import FAISS
from langchain_core.documents import Document

from config import (
    LEARNING_DATA_DIR,
    CHATS_DATA_DIR,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


logger = logging.getLogger("J.A.R.V.I.S")



# VECTOR STORE SERVICE CLASS


class VectorStoreService:
    """
    Builds a FAISS index from learning_data .txt files and chats_data .json files,
    and provides a retriever to fetch the k most relevant chunks for query.
    """

    def __init__(self):
        """Create the embedding model (local) and text splitter; vector_store is set in create_vector_store()."""
        # Embeddings run locally (no API key); used to convert text into vectors for similarity search.
        self.embedding = HuggingFaceEmbeddings(
            Model_name=EMBEDDING_MODEL,
            model_kwargs={"devide": "cpu"},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            
        )