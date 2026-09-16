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