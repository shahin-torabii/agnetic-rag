from core.constants import MAX_RETRIES
from core.types import Intent, QueryContext
from graph.routing import (
    get_rewrite_chain,
    is_result_weak,
    pick_k,
    split_resolved_kinds,
)
from graph.state import AgentState
from orchestrator.handlers import (
    handle_audio,
    handle_document,
    handle_general,
    handle_image,
    handle_multimodal,
    handle_uploads,
)
from resolve.resolver import resolve
from router.classifier import (
    build_active_context,
    build_session_context,
    classify_query,
    llm_router,
)


def context_node(state: AgentState) -> AgentState:
    state.active_ctx = build_active_context(state.request, state.user_id)
    handle_uploads(
        state.request, state.active_ctx, state.session_id, state.user_id, state.db
    )
    state.session_ctx = build_session_context(state.session_id, state.user_id, state.db)
    return state


def resolver_node(state: AgentState) -> AgentState:
    state.target_files = resolve(
        state.request, state.active_ctx, state.session_ctx, state.user_id, state.db
    )
    return state


def router_node(state: AgentState) -> AgentState:

    request = state.request
    target_files = state.target_files
    resolved_images = target_files.images
    resolved_docs, resolved_audio = split_resolved_kinds(target_files)

    if (
        not (resolved_images or target_files.documents)
        and not state.active_ctx.has_file
    ):
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
        intent = llm_router(request.query, ctx)

    state.intent = intent
    state.ctx = ctx
    state.k = pick_k(intent)

    return state


def dispatcher_node(state: AgentState) -> AgentState:
    return state


def document_node(state: AgentState) -> AgentState:
    print("enter document")
    try:
        state.result = handle_document(
            state.intent, state.request, state.target_files, k=state.k
        )
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
        state.result = handle_multimodal(
            state.intent, state.request, state.target_files, k=state.k
        )
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


def reflect_node(state: AgentState) -> AgentState:

    state.rewritten_query = None
    if state.retry_count < MAX_RETRIES and (
        state.error or is_result_weak(state.intent, state.result, state.request.query)
    ):
        state.retry_count += 1
        state.rewritten_query = get_rewrite_chain().invoke(
            {"query": state.request.query, "intent": state.intent}
        )
        state.request.query = state.rewritten_query

    return state
