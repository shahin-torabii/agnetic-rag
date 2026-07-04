from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Literal
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


def dispatch_branch(state: AgentState) -> Route:

    intent = state.intent
    request = state.request
    print("intent is:" , intent)

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

    if request.audio or request.images:
        return "multimodal"

    return "document"


def document_node(state: AgentState) -> AgentState:
    try:
        state.result = handle_document(state.intent, state.request, state.target_files, k=state.k)
        state.error = None
    except Exception as e:
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


def is_result_weak(intent: Intent, result:Any, query:str)-> bool:

    if result is None:
        return True

    if isinstance(result, (list, tuple, str)) and len(result) == 0:
        return True

    verdict = get_evaluator_chain().invoke({"query": query, "intent":intent, "response": result})

    return verdict.lower().strip() == "fail"


def reflect_node(state: AgentState) -> AgentState:
    if (state.retry_count < MAX_RETRIES
       and (state.error or is_result_weak(state.intent, state.result, state.request.query))):
        state.retry_count += 1
        state.rewritten_query = get_rewrite_chain().invoke({"query": state.request.query, "intent": state.intent})

    return state


def reflect_branch(state: AgentState) -> str:
    if state.rewritten_query:
        state.rewritten_query =None
        return  "retry"

    return "done"

#################### Build Graph  ############################
def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("context", context_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("document", document_node)
    graph.add_node("image", image_node)
    graph.add_node("audio", audio_node)
    graph.add_node("multimodal", multimodal_node)
    graph.add_node("general", general_node)
    graph.add_node("reflect", reflect_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "context")
    graph.add_edge("context", "resolver")
    graph.add_edge("resolver", "dispatcher")

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


def run_agent(request: UserRequest):
    app = build_graph()
    final_state = app.invoke(AgentState(request=request))
    result = final_state["result"] if isinstance(final_state, dict) else final_state.result
    return result