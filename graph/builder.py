from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    context_node, resolver_node, router_node, dispatcher_node,
    document_node, image_node, audio_node, multimodal_node,
    general_node, reflect_node,
)
from graph.routing import dispatch_branch, reflect_branch
from core.types import UserRequest


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
