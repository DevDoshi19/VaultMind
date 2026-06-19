# app/graph.py
from langgraph.graph import StateGraph, END
from app.state import RAGState
from app.nodes import (
    input_guardrail_node,
    classify_query_node,
    retrieve_node,
    context_guard_node,
    generate_node,
    output_guardrail_node,
    confidence_node,
)

def route_input(state: RAGState) -> str:
    if state.get("input_blocked", False):
        return "end"
    return "classify"

def route_query(state: RAGState) -> str:
    if state.get("question_is_relevant", False):
        return "retrieve"
    return "generate" # goes to generate with empty docs → early exit


def build_graph():
    graph = StateGraph(RAGState)

    # Register nodes / create a node 
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("classify", classify_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("context_guard", context_guard_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("confidence", confidence_node)
    
    # edges That joint the nodes .. 
    graph.set_entry_point("input_guardrail")

    graph.add_conditional_edges(
        "input_guardrail",
        route_input,
        {
            "end"     : END,
            "classify": "classify",
        }
    )

    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        }
    )

    graph.add_edge("retrieve", "context_guard")
    graph.add_edge("context_guard", "generate")
    graph.add_edge("generate", "output_guardrail")
    graph.add_edge("output_guardrail", "confidence")
    graph.add_edge("confidence", END)

    return graph.compile()


rag_graph = build_graph()