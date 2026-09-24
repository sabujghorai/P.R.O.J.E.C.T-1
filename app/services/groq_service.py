"""
GROQ SERVICES MODULE
====================

This module handles general chat: no web search, only the Groq LLM plus context
from the vector store (learning data + past chats). Used by ChatServices for 
POST/chats

MULTIPLE API KEYS (round-robin and fallback)
  - You can set multiple Groq API keys in .env: GROQ_APU_KEY, GROQ_API_KEY_2.
    GROQ_APU_KEY_3, ...(no limits).
    - Each request uses onw key in rotation: 1st request -> 1st key, 2nd request ->
    2nd key, 3rd request -> 3rd key, then back to 1st key, and so on. Every key
    is used one-by-one so usage is spread across keys.
    - The round-robun counter is shared across all instances (GroqServices and
      RealtimeGroqServices), so both /chat and /chat/realtime endpoint use the
      same rotation sequence.
    - If the chosen key fails (rate limit 429 or any error), we try the next key,
      then the next, until one succeeds or all have been tried.
    - All API key usage is logges with masked keys (first 8 and last 4 chars visible)
      for security and debugging purposes.

FLOW:
  1. get_responce(question, chat_history) is called.
  2. we ask the vector store for the top-k chunks most similar to the question (retrival).
  3. we build a system message: JARVIS_SYSTEM_PROMPT + current time + retrived context.
  4. we send to Groq using next key in rotation (or fallback to next key on failure).
  5. we return the assistant's reply.

Context is only what we retrive (no full dump of learning data), so token usage stays bounded.
"""

from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import logging

from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information