# app/graph.py
from langgraph.graph import StateGraph, END
from app.state import RAGState
from app.nodes import (
    retrieve_node, 
    generate_node, 
    context_guard_node, 
    classify_query_node,
    confidence_node
)

def route_query(state:RAGState)->str:
    if state["question_is_relevant"] :
        return "retrieve"
    return "generate" # goes to generate with empty docs → early exit


def build_graph():
    graph = StateGraph(RAGState)

    # Register nodes / create a node 
    graph.add_node("classify", classify_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("context_guard", context_guard_node)
    graph.add_node("generate", generate_node)
    graph.add_node("confidence",confidence_node)
    
    # edges That joint the nodes .. 
    graph.set_entry_point("classify")
    # conditional edge — branches based on classifier result
    graph.add_conditional_edges("classify",
                                    route_query,
                                    {
                                        "retrieve":"retrieve",
                                        "generate":"generate",
                                    }
                                )
    graph.add_edge("retrieve","context_guard")
    graph.add_edge("context_guard", "generate")
    graph.add_edge("generate","confidence")
    graph.add_edge("confidence", END)

    return graph.compile()


rag_graph = build_graph()