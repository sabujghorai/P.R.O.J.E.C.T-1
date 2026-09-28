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
call get_retriever

"""