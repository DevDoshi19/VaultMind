
# evaluation/ragas_eval.py
import os
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
from rich.console import Console
from rich.table import Table

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from evaluation.test_dataset import TEST_DATASET

load_dotenv()

console = Console()

_EVAL_CHROMA_DIR = None
_CHROMA_WARNING_SHOWN = False


def _prepare_eval_chroma_dir():
    global _EVAL_CHROMA_DIR

    if _EVAL_CHROMA_DIR:
        return _EVAL_CHROMA_DIR

    source_dir = Path("./chroma_db")
    cache_root = Path("./.ragas_eval_cache")
    target_dir = cache_root / "chroma_db"

    if not source_dir.exists():
        return str(source_dir)

    if cache_root.exists():
        shutil.rmtree(cache_root)

    shutil.copytree(source_dir, target_dir)

    _EVAL_CHROMA_DIR = str(target_dir)
    return _EVAL_CHROMA_DIR


def run_vaultmind(question: str):
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from rank_bm25 import BM25Okapi

    openai_key = os.getenv("OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_key,
    )

    semantic_docs = []

    global _CHROMA_WARNING_SHOWN

    try:
        vectorstore = Chroma(
            collection_name="vaultmind_resume",
            embedding_function=embeddings,
            persist_directory=_prepare_eval_chroma_dir(),
        )

        results = vectorstore.similarity_search_with_score(
            question,
            k=3,
        )

        semantic_docs = [
            doc.page_content
            for doc, score in results
            if score < 1.6
        ]

    except Exception as exc:
        if not _CHROMA_WARNING_SHOWN:
            console.print(
                f"[yellow]Chroma unavailable, using BM25 only: {exc}[/yellow]"
            )
            _CHROMA_WARNING_SHOWN = True

    with open("./data/chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = [item["content"] for item in data]

    tokenized = [
        doc.lower().split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(
        question.lower().split()
    )

    top_k = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    bm25_docs = [
        corpus[i]
        for i in top_k
        if scores[i] > 0
    ]

    merged = []
    seen = set()

    for doc in semantic_docs + bm25_docs:
        if doc not in seen:
            seen.add(doc)
            merged.append(doc)

    if not merged:
        return {
            "answer": "I don't have relevant information.",
            "contexts": []
        }

    context = "\n\n".join(merged)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer ONLY from the provided context."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": answer,
        "contexts": merged,
    }


def build_dataset():

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    console.print(
        "\n[cyan]Running evaluation questions...[/cyan]\n"
    )

    for idx, item in enumerate(TEST_DATASET):

        result = run_vaultmind(
            item["question"]
        )

        questions.append(
            item["question"]
        )

        answers.append(
            result["answer"]
        )

        contexts.append(
            result["contexts"]
        )

        ground_truths.append(
            item["ground_truth"]
        )

        console.print(
            f"[green]{idx+1}/{len(TEST_DATASET)} completed[/green]"
        )

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation():

    console.print(
        "\n[bold cyan]VaultMind RAG Evaluation[/bold cyan]"
    )

    dataset = build_dataset()

    openai_key = os.getenv(
        "OPENAI_API_KEY"
    )

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=openai_key,
        )
    )

    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_key,
        )
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df = results.to_pandas()

    table = Table(
        title="RAGAS Scores"
    )

    table.add_column("Metric")
    table.add_column("Score")

    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:

        if metric in df.columns:

            score = df[metric].mean()

            table.add_row(
                metric,
                f"{score:.3f}",
            )

    console.print(table)

    overall = (
        df[
            [
                c
                for c in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
                if c in df.columns
            ]
        ]
        .mean()
        .mean()
    )

    console.print(
        f"\n[bold green]Overall Score: {overall:.3f}[/bold green]"
    )

    return results


if __name__ == "__main__":
    run_evaluation()
