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