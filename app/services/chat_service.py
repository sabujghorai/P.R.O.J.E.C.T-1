"""
CHAT SERVICE MODULE
===================

This service owns all chat session and conversation logic. It is used by the
/chat and /chat/realtime endpoints. Designed for single-user use: one server
has one ChatService and one in-memory session store; the user can have many
sessions (each identified by session_id).

RESPONSIBILITIES:
    - get_or_create_session(session_id): Return existing session or create new one.
      If the user sends a session_id that was used before (e.g. before a restart),
      we try to load it from disk so the conversation continues.
    - add_message / get_chat_history: Keep messages in memory per session.
    - format_history_for_llm: Turn the message list into (user, assistant) pairs
      and trim to MAX_CHAT_HISTORY_TURNS so we don't overflow the prompt.
    - process_message / process_realtime_message: Add user message, call Groq (or
      RealtimeGroq), add assistant reply, return reply.
    - save_chat_session: Write session to database/chats_data/*.json so it persists
      and can be loaded on next startup (and used by the vector store for retrieval).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import uuid

from config import CHATS_DATA_DIR, MAX_CHAT_HISTORY_TURNS
from app.models import ChatMessage, ChatHistory
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService


logger = logging.getLogger("J.A.R.V.I.S")


# ============================================================================
# CHAT SERVICE CLASS
# ============================================================================

class ChatService:
    """
    Manages chat sessions: in-memory message lists, load/save to disk, and
    calling Groq (or Realtime) to get replies. All state for active sessions
    is in self.sessions; saving to disk is done after each message so
    conversations survive restarts.
    """

    def __init__(self,groq_service: GroqService,realtime_service: RealtimeGroqService = None):
        """ Store references to the groq and Realtime services; keep session in memory."""
        self.groq_services = groq_service
        self.realtime_services = realtime_service