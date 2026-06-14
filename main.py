# main.py
from rich.console import Console
from rich.panel import Panel
from rich import print
from app.graph import rag_graph
from app.state import RAGState

console = Console()

def run_query(question: str) -> dict:
    initial_state: RAGState = {
        "question": question,
        "question_is_relevant": False, 
        "retrieved_docs": [],
        "retrieval_status":"",
        "context_token_count":0,
        "answer": "",

        "confidence_score":0.0,

        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost": 0.0,
    }

    result = rag_graph.invoke(initial_state)
    return result

if __name__ == "__main__":
    print("\n")
    print("-" * 40)
    print("🧠 VaultMind — Phase 4")
    print("-" * 40)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() == "exit" or question.lower() == "quit" or question.lower() == "bye":
            print("👋 Bye!")
            break

        result = run_query(question)
        # print(f"\nVaultMind: {result['answer']}")  # ← only this prints the answer
        console.print(Panel(
            result['answer'],
            title="[bold green]VaultMind[/bold green]",
            border_style="green"
        ))

        console.print(
                f"[dim]📊 Tokens: {result['total_tokens']} | "
                f"Cost: ${result['estimated_cost']:.6f} | "
                f"Confidence: {result['confidence_score']:.2f}[/dim]"
        )