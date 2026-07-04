from dataclasses import dataclass, Field
from typing import Any, Optional
from langgraph.graph import StateGraph, END

from index_embedd import VectorStore
from query_router import (
    Intent, QueryContext, classify_query, llm_router,
    ActiveContext, manage_active_context
)
from resolver import resolve
from data_gathering import Data, UserRequest
from query_handler import (
    handle_uploads, handle_audio, handle_image, handle_document,
    handle_multimodal, handle_general
)

from LLM import HF_LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class AgentState:
    request: UserRequest
    ctx: Optional[QueryContext] = None
    intent: Optional[Intent] = None
    target_files : Any = None
    k:int = 10
    result: Any = None
    error: Optional[str] =None
    retry_count: int = 0
    rewritten_query: Optional[str] = None


def pick_k(intent: Intent) -> int:
    if intent in (Intent.DOCUMENT_QA, Intent.SEARCH_DOCUMENT):
        return 6
    if intent in (Intent.DOCUMENT_SUMMARIZE, Intent.DOCUMENT_OVERVIEW,
                  Intent.DOCUMENT_FULL_EXPLAIN, Intent.COMPARE_DOCUMENTS):
        return 20
    return 10


def router_node(state:AgentState) -> AgentState:

    request = state.request
    has_image = (
        request.images is not None
        and len(request.images) > 0)

    has_document = (
        request.documents is not None
        and len(request.documents) > 0)
    ####equivalent:
    ## has_document = bool(request.document)
    has_audio = (
        request.audio is not None
        and len(request.audio) > 0)

    ctx = QueryContext(
        has_image=has_image,
        has_document=has_document,
        has_audio=has_audio,

        num_images=len(request.images) if request.images else 0,
        num_documents=len(request.documents) if request.documents else 0,
        num_audio=len(request.audio) if request.audio else 0,

        has_stored_documents=(len(Data.docs) > 0),
        has_stored_images=( VectorStore.image_index is not None
            and VectorStore.image_index.ntotal > 0),
        has_stored_audio=(
            any(
                getattr(doc, "doc_type", None) == "audio"
                for doc in Data.docs.values())
        )
    )

    intent = classify_query(request.query, ctx)

    if intent == Intent.UNKNOWN:
        intent = llm_router(request.query,ctx)

    state.intent = intent
    state.ctx = ctx
    state.k = pick_k(intent)
    
    return state


def context_node(state:AgentState) -> AgentState:

    manage_active_context(state.request)
    handle_uploads(state.request)
    return state


def resolver_node(state: AgentState) -> AgentState:
    state.target_files = resolve(state.request)
    return state


def dispatcher_node(state: AgentState) -> AgentState:
    return state


def dispatch_branch(state: AgentState) -> str:

    intent = state.intent
    request = state.request

    INTENT_ROUTE = {
        Intent.IMAGE_SEARCH: "image",
        Intent.IMAGE_UNDERSTANDING: "image",

        Intent.AUDIO_QA: "audio",
        Intent.AUDIO_SUMMARIZE: "audio",
        Intent.AUDIO_TRANSCRIBE: "audio",
        Intent.AUDIO_OVERVIEW: "audio",

        Intent.GENERAL_CHAT: "general",
        Intent.UNKNOWN: "general",
    }

    route = INTENT_ROUTE.get(intent)

    if route:
        return route

    if request.audio or request.images:
        return "multimodal"

    return "document"