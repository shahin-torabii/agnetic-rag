from typing import List
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.types import Data
from core.constants import STRUCTURAL_TERMS
from llm.client import HF_LLM
from llm.prompts import IMPROVE_QUERY_SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_improve_query_chain():
    improve_query_prompt = ChatPromptTemplate.from_messages([
        ("system", IMPROVE_QUERY_SYSTEM_PROMPT),
        ("human", "User Query: {query}\n\nOptimized Search Query (Output only the terms):"),
    ])

    improve_query_chain = improve_query_prompt | HF_LLM.fast_llm | StrOutputParser()

    return improve_query_chain


@lru_cache(maxsize=256)
def improve_query(query: str) -> str:
    """
    Cached — same query string won't trigger a second llm call
    within the same session.
    """
    return get_improve_query_chain().invoke({"query": query}).strip()


def is_structural_query(query: str) -> bool:
    q = query.lower()
    return any(term.lower() in q for term in STRUCTURAL_TERMS)


def _chunk_to_dict(chunk) -> dict:
    return {
        "text":         chunk.text,
        "doc_id":       chunk.doc_id,
        "chunk_index":  chunk.chunk_index,
        "chunk_type":   chunk.chunk_type,
        "section_path": chunk.section_path,
        "image_refs":   chunk.image_refs,
        "table_refs":   chunk.table_refs,
    }


def build_context(text_results:List[dict], image_results:List[dict]) ->str:
    parts = []

    for r in text_results:
        path = " > ".join(r["section_path"]) if r["section_path"] else r["doc_id"]
        parts.append(f"[{path}]\n{r['text']}")
    if image_results:
        for r in image_results:
              key = (r["doc_id"], r["image_id"])
              chunks  = Data.image_to_chunks.get(key,[])
              if chunks:
                  related = chunks[0]
                  path=" > ".join(r["section_path"] if r["section_path"] else [r["doc_id"]])
                  parts.append(f"[{path}| image_{r['image_id']}]\n context: {related.text}")

    return "\n___\n".join(parts)
