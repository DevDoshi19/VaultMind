# main.py
from rich.console import Console
from rich.panel import Panel
from rich import print
from app.graph import rag_graph
from app.state import RAGState
from langchain_core.tracers.context import tracing_v2_enabled

console = Console()

def run_query(question: str) -> dict:
    initial_state: RAGState = {
        "question": question,
        "question_is_relevant": False,
        "retrieved_docs": [],
        "retrieval_status": "",
        "context_token_count": 0,
        "answer": "",
        "confidence_score": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
    }

    with tracing_v2_enabled(project_name="vaultmind"):
        result = rag_graph.invoke(
            initial_state,
            config={
                "run_name": f"VaultMind | {question[:50]}",
                "tags": ["production", "resume-rag"],
                "metadata": {
                    "phase": "8",
                    "retrieval": "hybrid",
                    "llm": "gpt-4o-mini",
                }
            }
        )

    return result


def format_confidence(score) -> str:
    """Safely format confidence score — handles None for blocked/irrelevant queries."""
    if score is None:
        return "[dim]N/A[/dim]"
    elif score >= 0.7:
        return f"[green]{score:.2f} ✅[/green]"
    elif score >= 0.4:
        return f"[yellow]{score:.2f} ⚠️[/yellow]"
    else:
        return f"[red]{score:.2f} ❌[/red]"


if __name__ == "__main__":
    print("\n")
    print("-" * 40)
    print("🧠 VaultMind — Phase 4")
    print("-" * 40)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in ("exit", "quit", "bye"):
            print("👋 Bye!")
            break

        result = run_query(question)

        console.print(Panel(
            result["answer"],
            title="[bold green]VaultMind[/bold green]",
            border_style="green",
        ))

        confidence_str = format_confidence(result.get("confidence_score"))

        console.print(
            f"[dim]📊 Tokens: {result['total_tokens']} | "
            f"Cost: ${result['estimated_cost']:.6f}[/dim] | "
            f"Confidence: {confidence_str}"
        )