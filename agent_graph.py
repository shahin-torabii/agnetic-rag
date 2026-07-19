from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Literal
from langgraph.graph import StateGraph, END

from index_embedd import VectorStore
from query_router import (
    Intent, QueryContext, classify_query, llm_router,
    ActiveContext, build_active_context, build_session_context
)
from resolver import resolve
from data_gathering import Data, UserRequest
from query_handler import (
    handle_uploads, handle_audio, handle_image, handle_document,
    handle_multimodal, handle_general)

from LLM import HF_LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import UserDocument



@dataclass
class AgentState:
    request: UserRequest
    user_id: str
    session_id: str
    db: Any = None
    ctx: Optional[QueryContext] = None
    intent: Optional[Intent] = None
    active_ctx: Any = None
    session_ctx: Any = None
    target_files: Any = None
    k: int = 10
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    rewritten_query: Optional[str] = None


MAX_RETRIES = 3


Route = Literal[
    "document",
    "image",
    "audio",
    "multimodal",
    "general",
]


EVALUATION_PROMPT = """
You are evaluating the quality of an AI response.

Determine whether the response adequately satisfies the request.

Return only one word.
Return ONLY one of:

PASS
FAIL
"""

@lru_cache(maxsize=1)
def get_evaluator_chain():
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATION_PROMPT),
        ("human", "User Request:{query} \n\n User Intent:{intent}\n\n Generated Response:{response}")
    ])
    evaluator_chain = eval_prompt | HF_LLM.fast_llm.bind(max_tokens=30, temprature=0) | StrOutputParser()
    #

    return evaluator_chain


@lru_cache(maxsize=1)
def get_rewrite_chain():
    REWRITER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Rewrite unclear original queries to be broader, stronger and clearer for the specific intent the user wants 
                   Return ONLY the rewritten query."""),
        ("human", "Original query: {query} \n\n Intent:{intent}")
    ])

    rewrite_chain = REWRITER_PROMPT | HF_LLM.fast_llm.bind(max_tokens=70) | StrOutputParser()

    return rewrite_chain

def pick_k(intent: Intent) -> int:
    if intent in (Intent.DOCUMENT_QA, Intent.SEARCH_DOCUMENT):
        return 6
    if intent in (Intent.DOCUMENT_SUMMARIZE, Intent.DOCUMENT_OVERVIEW,
                  Intent.DOCUMENT_FULL_EXPLAIN, Intent.COMPARE_DOCUMENTS):
        return 20
    return 10


def get_user_doc_kinds(user_id: str, db) -> set[str]:
    rows = db.query(UserDocument.kind).filter(UserDocument.user_id == user_id).distinct().all()
    return {r[0] for r in rows}


def split_resolved_kinds(target_files):
    docs, audio = [], []
    for doc_id in target_files.documents:
        meta = Data.docs.get(doc_id)
        if meta is not None and getattr(meta, "doc_type", None) == "audio":
            audio.append(doc_id)
        else:
            docs.append(doc_id)
    return docs, audio


def context_node(state: AgentState) -> AgentState:
    state.active_ctx = build_active_context(state.request, state.user_id)
    handle_uploads(state.request, state.active_ctx, state.session_id, state.user_id, state.db)
    state.session_ctx = build_session_context(state.session_id, state.user_id, state.db)
    return state


def resolver_node(state: AgentState) -> AgentState:
    state.target_files = resolve(
        state.request, state.active_ctx, state.session_ctx, state.user_id, state.db
    )
    return state


def router_node(state:AgentState) -> AgentState:

    request = state.request
    target_files = state.target_files
    resolved_images = target_files.images
    resolved_docs, resolved_audio = split_resolved_kinds(target_files)

    if not(resolved_images or target_files.documents) and not state.active_ctx.has_file:
        state.intent = Intent.GENERAL_CHAT
        state.ctx = QueryContext()
        state.k = pick_k(state.intent)
        return state

    ctx = QueryContext(
        has_image=bool(resolved_images),
        has_document=bool(resolved_docs),
        has_audio=bool(resolved_audio),
        num_images=len(resolved_images),
        num_documents=len(resolved_docs),
        num_audio=len(resolved_audio),
        has_stored_documents=bool(resolved_docs),
        has_stored_images=bool(resolved_images),
        has_stored_audio=bool(resolved_audio),
    )

    intent = classify_query(request.query, ctx)

    if intent == Intent.UNKNOWN:
        intent = llm_router(request.query,ctx)

    state.intent = intent
    state.ctx = ctx
    state.k = pick_k(intent)

    return state


def dispatcher_node(state: AgentState) -> AgentState:
    return state


def dispatch_branch(state: AgentState) -> Route:

    intent = state.intent
    request = state.request


    intent_route = {
        Intent.IMAGE_SEARCH: "image",
        Intent.IMAGE_UNDERSTANDING: "image",

        Intent.AUDIO_QA: "audio",
        Intent.AUDIO_SUMMARIZE: "audio",
        Intent.AUDIO_TRANSCRIBE: "audio",
        Intent.AUDIO_OVERVIEW: "audio",

        Intent.GENERAL_CHAT: "general",
        Intent.UNKNOWN: "general",
    }

    route = intent_route.get(intent)

    if route:
        return route

    if request.audio or request.images or request.documents:
        return "multimodal"

    return "document"


def document_node(state: AgentState) -> AgentState:
    print("enter document")
    try:
        state.result = handle_document(state.intent, state.request, state.target_files, k=state.k)
        print(state.result)
        state.error = None
    except Exception as e:
        print(e)
        state.error = str(e)
    return state


def image_node(state: AgentState) -> AgentState:
    try:
        state.result = handle_image(state.intent, state.request, state.target_files)
    except Exception as e:
        state.error = str(e)
    return state


def audio_node(state: AgentState) -> AgentState:
    try:
        state.result = handle_audio(state.intent, state.request, state.target_files)
    except Exception as e:
        state.error = str(e)
    return state


def multimodal_node(state: AgentState) -> AgentState:
    try:
        state.result = handle_multimodal(state.intent, state.request, state.target_files, k=state.k)
        state.error = None
    except Exception as e:
        state.error = str(e)
    return state


def general_node(state: AgentState) -> AgentState:
    try:
        state.result = handle_general(state.intent, state.request)
    except Exception as e:
        state.error = str(e)
    return state


REFLECTABLE_INTENTS = (
    Intent.DOCUMENT_QA,
    Intent.SEARCH_DOCUMENT,
    Intent.AUDIO_QA,
    Intent.IMAGE_SEARCH,
)

def is_result_weak(intent: Intent, result: Any, query: str) -> bool:
    if intent not in REFLECTABLE_INTENTS:
        return False

    if result is None:
        return True
    if isinstance(result, (list, tuple, str)) and len(result) == 0:
        return True

    verdict = get_evaluator_chain().invoke({
        "query": query,
        "intent": intent.value,
        "response": result,
    })
    return verdict.lower().strip() == "fail"

def reflect_node(state: AgentState) -> AgentState:

    state.rewritten_query = None
    if (state.retry_count < MAX_RETRIES
       and (state.error or is_result_weak(state.intent, state.result, state.request.query))):
        state.retry_count += 1
        state.rewritten_query = get_rewrite_chain().invoke({"query": state.request.query, "intent": state.intent})
        state.request.query = state.rewritten_query

    return state


def reflect_branch(state: AgentState) -> str:
    if state.rewritten_query:
        return  "retry"

    return "done"

#################### Build Graph  ############################
def build_graph():

    graph = StateGraph(AgentState)


    graph.add_node("context", context_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("router", router_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("document", document_node)
    graph.add_node("image", image_node)
    graph.add_node("audio", audio_node)
    graph.add_node("multimodal", multimodal_node)
    graph.add_node("general", general_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("context")
    graph.add_edge("context", "resolver")
    graph.add_edge("resolver", "router")  # reordered
    graph.add_edge("router", "dispatcher")

    graph.add_conditional_edges("dispatcher", dispatch_branch, {
        "document": "document",
        "image": "image",
        "audio": "audio",
        "multimodal": "multimodal",
        "general": "general",
    })

    graph.add_edge("document", "reflect")
    graph.add_edge("multimodal", "reflect")
    graph.add_edge("image", "reflect")
    graph.add_edge("audio", "reflect")
    graph.add_edge("general", "reflect")

    graph.add_conditional_edges("reflect", reflect_branch,{
        "retry": "resolver",
        "done":END
    })

    return graph.compile()


def run_agent(request: UserRequest, user_id: str, session_id: str, db=None):
    app = build_graph()
    initial_state = AgentState(request=request, user_id=user_id, session_id=session_id, db = db)
    final_state = app.invoke(initial_state)
    return final_state["result"] if isinstance(final_state, dict) else final_state.result