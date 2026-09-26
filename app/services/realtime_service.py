"""
REALTIME GROQ SERVICES MODULE
=============================

Extends GroqService to add Tavily web search before calling the LLM. Used by
ChatService for POST /chat/realtime. Same session and history as general chat;
the only difference is we run a Tvily serch for the user's question and add
the results to the system message, then call Groq.

ROUND-ROBIN API KEYS:
  - Shares the same round-robi counter as GroqSearvice (class-level _shared_key_index)
  - This means /chat and /chat/realtime request use the same rotation sequence
  - Example: If /chat uses key 1, the next /chat/realtime request will use key 2
  - All API key usage is logged with masked keys for security and debugging

FLOW:
  1. search_taivily(question): call Taivily API, format result as text (or "" on failure).
  2. get_response(question, chat_history): add search results to system messgae,
     then same as parent: retrieve context from vector store, build prompt, call Groq.

IF TAVILY_API_KEY is not set, tavily_client is None and search_tavily returns "";
the user still gets an answer from Groq with no search resrults.
"""

from typing import List,Optional
from tavily import TavilyClient
import logging
import os

from app.services.groq_service import GroqService, escape_curly_braces
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import JARVIS_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


logger = logging.getLogger("J.A.R.V.I.S")



# REALTIME GROQ SERVICE CLASS (extend GroqServices)


class RealtimeGroqService(GroqService):
    """
    Same as GroqService but runs a Tavily web search first and adds the results
    to the system message. If Tavily is missing or fails, we still call Groq with
    no search results (user gets an answer without real-time data).
    """

    def __init__(self, vector_store_services: VectorStoreService):
        """Call parent init (Groq LLM + vector store); then create Tavily client if key is set."""
        super().__init__(vector_store_services)
        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily search client initialized successfully")
        else:
            self.tavily_client = None
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable")

    def search_tavily(self, query: str, num_results: int = 5) -> str:
        """
        Call Tavily API with the given query and return formatted result text for the prompt.
        On any failure (no key, rate limit, network) we return "" so the LLM still gets a reply.
        """
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ""

        try:
            # Perform Tavily search with retrives for rate limits and transient errors.
            response = with_retry(
                lambda: self>self.tavily_client.search(
                    query=query,
                    search_depth="basic", # Basic is faster advance is more through
                    max_result=num_results,
                    include_answer=False, # we'll format out ows results
                    include_raw_content=False # don't need full content
                ),
                max_retries=3,
                initial_delay=1.0,
            )

            results = response.get('results', [])

            if not results:
                logger.warning(f"No Tavily search results found for query: {query}")
                return ""

            # Format search result as text for the syntax prompt.
            formatted_results = f"Search result for '{query}':\n[start]]\n"

            for i, result in enumerate(results[:num_results], 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No description')
                url = result.get('url', '')

                formatted_results += f"title: {title}\n"
                formatted_results += f"Description: {content}\n"
                if url:
                    formatted_results += f"URL: {url}\n"
                formatted_results += "\n"

            formatted_results += "[end]"

            logger.info(f"Tavily search completed for query: {query} ({len(results)} results)")

        except Exception as e:
            # If search faild (network error, rate limit, etc), log and return empty
            # The AI will still respond using its knowledge, just with real-time date
            logger.error(f"Error performing Tavily search: {e}")
            return ""
        