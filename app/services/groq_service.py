"""
GROQ SERVICE MODULE
===================

This module handles general chat: no web search, only Groq LLM plus context
from the vector store (learning data + past chats). Used by ChatServices for 
POST / chat.

MULTIPLE API KEYS (round-robin and fallback):
  - You can set multiple Groq API keys in .env: GROQ_API_KEY, GROQ_API_KEY_2,
    GROQ_API_KEY_3, ... (no limit).
  - Each request uses one key in rotation: 1st request -> 1st key, 2nd request ->
    2nd key, 3rd request, then back to 1st key, and so on. Every key
    is used one-by-one so usage is spread across keys.
  - The round-robin counter is shared across all instances (GroqService and 
    RealtimeGroqService), so both /chat and /chat/realtime endpoints use the 
    same rotation sequence.
  - If the chosen key fails (rate limit 429 or any error), we try the next key,
    then the next, until one succeeds or all have been tried.
  - All API key usage is logged with masked keys (first 8 and last 4 chars visible)
    for security and debugging purposes.

FLOw:
  1: get_responses (question, chat_history) is called.
  2: We ask the vector store for the top-k chunks most similar to the question (retrieval).
  3: We build a system message: JARVIS_SYSTEM_PROMPT + current time + retrieved context.
  4: We send to Groq using the next key in rotation (or fallback to next key on failure).
  5: We return the assistant's reply.

Cotext is only what we retrieve (no fill dump of learning data), so token usage stays bounded.
"""

from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEY, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")



# HELPER: ESCAPE CURLY BRACES FOR LANGCHAIN

# Langchain prompt templates use {variable_name}. If learning data or chat
# content contains { or }, the template engine can break, Doubling them 
# makes them literal int he final string.

def escape_curly_braces(text: str) -> str:
    """
    Double every { and } so Langchain does not treat them as template variables.
    Learning data or chat content might contain { or }; without escaping, invoke() can fail.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Return True if the exception indicates a Groq rate limit (e.g. 429, tokens per day).
    Used for logging; actual fallback tries the next key on any failure when multiple keys exist.
    """
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


def _mask_api_key(key: str) -> str:
    """
    Mask an API key for safe logging. Show first 8 and last 4 characters, maskd the middle.
    Example: gsk_1234567890abcdef -> gsk_1234...cdef
    """
    if not key or len(key) <= 12:
        return "***maksed***"
    return f"{key[:8]}...{key[-4:]}"



# GROQ SERVICE CLASS


class GroqService:
    """
    General chat: retrieves context from the vector store and calls the Groq LLM.
    Supports multiple API keys: each request uses the next key in rotation (one-by-one),
    and if that key fails, the server tries the next key until one succeeds or all fail.

    ROUND-ROBIN BEHAVIOR:
    - Request 1 uses key 0 ( first key)
    - Request 2 uses key 1 ( second key)
    - Request 3 uses key 2 ( third key)
    - After all keys are used, cycles back to key 0
    - If a key fails (rate limit error), tries the next key in sequence
    - All request share the saem round-robin counter (class-level)
    """