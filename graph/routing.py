from functools import lru_cache
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.manager import get_config
from core.constants import MAX_RETRIES
from core.types import Intent
from graph.state import AgentState
from llm.client import HF_LLM
from llm.prompts import EVALUATION_PROMPT

Route = Literal[
    "document",
    "image",
    "audio",
    "multimodal",
    "general",
]


REFLECTABLE_INTENTS = (
    Intent.DOCUMENT_QA,
    Intent.SEARCH_DOCUMENT,
    Intent.AUDIO_QA,
    Intent.IMAGE_SEARCH,
)


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


def reflect_branch(state: AgentState) -> str:
    if state.rewritten_query:
        return "retry"

    return "done"


def is_result_weak(intent: Intent, result: Any, query: str) -> bool:
    if intent not in REFLECTABLE_INTENTS:
        return False

    if result is None:
        return True
    if isinstance(result, (list, tuple, str)) and len(result) == 0:
        return True

    verdict = get_evaluator_chain().invoke(
        {
            "query": query,
            "intent": intent.value,
            "response": result,
        }
    )
    return verdict.lower().strip() == "fail"


@lru_cache(maxsize=1)
def get_evaluator_chain():
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVALUATION_PROMPT),
            (
                "human",
                "User Request:{query} \n\n User Intent:{intent}\n\n Generated Response:{response}",
            ),
        ]
    )
    evaluator_chain = (
        eval_prompt
        | HF_LLM.fast_llm.bind(max_tokens=30, temprature=0)
        | StrOutputParser()
    )
    #

    return evaluator_chain


@lru_cache(maxsize=1)
def get_rewrite_chain():
    REWRITER_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rewrite unclear original queries to be broader, stronger and clearer for the specific intent the user wants
                   Return ONLY the rewritten query.""",
            ),
            ("human", "Original query: {query} \n\n Intent:{intent}"),
        ]
    )

    rewrite_chain = (
        REWRITER_PROMPT | HF_LLM.fast_llm.bind(max_tokens=70) | StrOutputParser()
    )

    return rewrite_chain


def pick_k(intent: Intent) -> int:
    cfg = get_config().retrieval
    if intent in (Intent.DOCUMENT_QA, Intent.SEARCH_DOCUMENT):
        return cfg.top_k
    if intent in (
        Intent.DOCUMENT_SUMMARIZE,
        Intent.DOCUMENT_OVERVIEW,
        Intent.DOCUMENT_FULL_EXPLAIN,
        Intent.COMPARE_DOCUMENTS,
    ):
        return cfg.text_k
    return cfg.top_k


def split_resolved_kinds(target_files):
    from core.types import Data

    docs, audio = [], []
    for doc_id in target_files.documents:
        meta = Data.docs.get(doc_id)
        if meta is not None and getattr(meta, "doc_type", None) == "audio":
            audio.append(doc_id)
        else:
            docs.append(doc_id)
    return docs, audio
