import tiktoken
import json
from rich.console import Console
from rich import print

from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from openai import APIError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.state import RAGState

console = Console()

MAX_CONTEXT_TOKENS = 1000
INPUT_PRICE_PER_TOKEN  = 0.150 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 0.600 / 1_000_000

def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

def load_bm25():
    with open("./data/chunks.json", "r") as f:
        data = json.load(f)

    corpus = [item["content"] for item in data]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    return bm25, corpus


def rrf_merge(semantic_docs: list, bm25_docs: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion.
    Merges two ranked lists, boosts chunks appearing in both.
    """
    scores = {}

    for rank, doc in enumerate(semantic_docs):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + k)

    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + k)

    # sort by combined score descending
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in merged]


def retrieve_node(state: RAGState) -> RAGState:
    console.print("\n[cyan]🔍 Hybrid Retrieving...[/cyan]")

    question = state["question"]

    # ── Semantic Search ────────────────────────────────────
    vectorstore = get_vectorstore()
    semantic_results = vectorstore.similarity_search_with_score(
        question,
        k=settings.retrieval_k,
    )

    # print scores for visibility
    for doc, score in semantic_results:
        console.print(f"   [dim]semantic score: {score:.4f} | {doc.page_content[:50]}[/dim]")

    # filter by threshold
    semantic_docs = [
        doc.page_content
        for doc, score in semantic_results
        if score < 1.6
    ]

    # ── BM25 Keyword Search ────────────────────────────────
    bm25, corpus = load_bm25()
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # get top-K BM25 results
    top_k_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:settings.retrieval_k]

    bm25_docs = [corpus[i] for i in top_k_indices if bm25_scores[i] > 0]

    console.print(f"   Semantic hits : [green]{len(semantic_docs)}[/green]")
    console.print(f"   BM25 hits     : [green]{len(bm25_docs)}[/green]")

    # ── RRF Merge ──────────────────────────────────────────
    merged = rrf_merge(semantic_docs, bm25_docs)

    # deduplicate
    seen = set()
    unique = []
    for doc in merged:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)

    console.print(f"   After RRF+dedup: [green]{len(unique)}[/green] chunks")

    if not unique:
        return {
            **state,
            "retrieved_docs": [],
            "retrieval_status": "empty",
        }

    return {
        **state,
        "retrieved_docs": unique,
        "retrieval_status": "ok",
    }

# ── Node 2 ────────────────────────────────────────────────
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are VaultMind, an assistant that answers questions 
about a candidate's resume. Answer ONLY from the context provided.
If the answer is not in the context, say 'I don't have that information in the resume.'""",
        ),
        (
            "human",
            """Context:{context}
                Question: {question}
                Answer:""",
        ),
    ]
)


# classidy Queary node , which is use to identify if the queary is relavent or not 
def classify_query_node(state: RAGState) -> RAGState:
    console.print("\n[cyan]🎯 Classifying query...[/cyan]")

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier for a resume assistant called VaultMind.
                    This assistant answers questions about DEV DOSHI's resume and professional background.

                    Your job is to decide if a question is related to:
                    - Dev Doshi as a person
                    - his skills, technologies, tools
                    - his projects, work experience
                    - his education, achievements
                    - anything about who he is professionally

                    Answer ONLY with 'yes' or 'no'. Nothing else.

                    Examples:
                    "Who is Dev Doshi?"              → yes
                    "Who is dev doshi?"              → yes
                    "What are his skills?"           → yes
                    "Tell me about his projects"     → yes
                    "What is attention mechanism?"   → no
                    "Capital of France?"             → no
                    "Explain transformers"           → no
        """),
            ("human", "Question: {question}"),
                        ])

    messages = classifier_prompt.format_messages(question=state["question"])
    response = llm.invoke(messages)

    is_relevant = response.content.strip().lower() == "yes"
    usage = response.response_metadata.get("token_usage", {})
    # print(f"\n\n\n{usage}\n\n\n")
    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens      = prompt_tokens + completion_tokens
    cost              = (prompt_tokens * INPUT_PRICE_PER_TOKEN) + (completion_tokens * OUTPUT_PRICE_PER_TOKEN)

    console.print(f"   Relevant : [{'green' if is_relevant else 'red'}]{is_relevant}[/{'green' if is_relevant else 'red'}]")
    console.print(f"   Tokens   : [yellow]{total_tokens}[/yellow] | Cost: [yellow]${cost:.6f}[/yellow]")

    return {
        **state,
        "question_is_relevant": is_relevant,
        "prompt_tokens"    : state["prompt_tokens"] + prompt_tokens,
        "completion_tokens": state["completion_tokens"] + completion_tokens,
        "total_tokens"     : state["total_tokens"] + total_tokens,
        "estimated_cost"   : state["estimated_cost"] + cost,
    }

# Context Length Guard node : Before calling the LLM, count the tokens in your context. If it's too long, trim it — keep only as many chunks as fit within the budget.
def count_token(text:str)->int:
    enc = tiktoken.encoding_for_model(settings.llm_model)
    return len(enc.encode(text))

def context_guard_node(state:RAGState)-> RAGState:
    console.print(f"\n[cyan]🛡️ Running Context Guard[/cyan]")  # noqa: F541

    if state["retrieval_status"] == "empty":
        return {
            **state,
            "context_token_count": 0,
        }
    allowed_chunks = []
    running_total = 0

    for chunk in state['retrieved_docs']:
        chunk_tokens = count_token(chunk)
        
        if running_total + chunk_tokens <= MAX_CONTEXT_TOKENS :
            allowed_chunks.append(chunk)
            running_total += chunk_tokens

        else:
            print(f"   ⚠️ Dropped chunk — would exceed budget ({running_total + chunk_tokens} tokens)")

    # print(f"   Chunks kept : {len(allowed_chunks)}/{len(state['retrieved_docs'])}")
    # print(f"   Token count : {running_total}/{MAX_CONTEXT_TOKENS}")

    console.print(f"   Chunks kept : [green]{len(allowed_chunks)}/{len(state['retrieved_docs'])}[/green]")
    console.print(f"   Token count : [yellow]{running_total}/{MAX_CONTEXT_TOKENS}[/yellow]")

    return {
        **state,
        "retrieved_docs": allowed_chunks,
        "context_token_count": running_total,
    }
        

# The retry logic node which will retry only the llm code , not the full chunking and all , that's why the call hase its own seprate function
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def _call_llm(llm, messages):
    return llm.invoke(messages)

# Node 3 : which is genrate The output, by using the question , it first go to prompttemplate and get the prompt with the context which is recive form the rag , then llm and then to the parser 
def generate_node(state: RAGState) -> RAGState:
    console.print("\n[cyan]🤖 Generating answer...[/cyan]")

    if not state["question_is_relevant"]:
        return {
            **state,
            "answer": "I can only answer questions about Dev's resume and professional background.",
        }

    if state["retrieval_status"] == "empty":
        return {
            **state,
            "answer": "I don't have relevant information in the resume to answer that.",
        }

    context = "\n\n".join(state["retrieved_docs"])

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    messages = PROMPT.format_messages(
        context=context,
        question=state["question"],
    )

    response = _call_llm(llm, messages)

    usage = response.response_metadata.get("token_usage", {})
    
    # print(f"\n\n\n{usage}\n\n\n")

    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens      = prompt_tokens + completion_tokens
    cost              = (prompt_tokens * INPUT_PRICE_PER_TOKEN) + \
                        (completion_tokens * OUTPUT_PRICE_PER_TOKEN)

    console.print(f"   Tokens : [yellow]{total_tokens}[/yellow] | Cost: [yellow]${cost:.6f}[/yellow]")

    return {
        **state,
        "answer"          : response.content,
        "prompt_tokens"   : state["prompt_tokens"] + prompt_tokens,
        "completion_tokens": state["completion_tokens"] + completion_tokens,
        "total_tokens"    : state["total_tokens"] + total_tokens,
        "estimated_cost"  : state["estimated_cost"] + cost,
    }


# Confidence Checker node : After generating the answer, you can add a node that checks the confidence of the response. This could be a simple heuristic (e.g., if the answer contains certain keywords) or another LLM call that evaluates the answer's quality. If confidence is low, you could trigger a fallback mechanism, like returning a generic response or asking the user to rephrase their question.

CONFIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator. Given a context, a question, and an answer,
                score how well the answer is supported by the context.

                Return ONLY a float between 0.0 and 1.0. Nothing else.

                0.0 → answer is completely unsupported or hallucinated
                0.5 → answer is partially supported
                1.0 → answer is fully supported by the context

                Examples:
                Context has full info, answer is accurate    → 0.95
                Context has partial info, answer fills gaps  → 0.50
                Context has nothing relevant, answer made up → 0.10
                """),
                    ("human", """Context:{context}

                                Question: {question}

                                Answer: {answer}

                    Score:"""),
])

def confidence_node(state: RAGState) -> RAGState:
    console.print("\n[cyan]🎯 Scoring confidence...[/cyan]")

    # skip if we never generated a real answer
    if not state["question_is_relevant"] or state["retrieval_status"] == "empty":
        return {
            **state,
            "confidence_score": 0.0,
        }

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    context = "\n\n".join(state["retrieved_docs"])

    messages = CONFIDENCE_PROMPT.format_messages(
        context = context,
        question =state["question"],
        answer = state["answer"]
    )

    response = _call_llm(llm,messages)

    # parse float safely
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))   # clamp between 0 and 1
    except ValueError:
        score = 0.5   # default if LLM returns something unexpected

    # track tokens 

    usage = response.response_metadata.get("token_usage", {})

    prompt_tokens     = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens      = prompt_tokens + completion_tokens
    cost              = (prompt_tokens * INPUT_PRICE_PER_TOKEN) + (completion_tokens * OUTPUT_PRICE_PER_TOKEN)

    # confidence label
    if score >= 0.7:
        label = f"[green]{score:.2f} ✅[/green]"
    elif score >= 0.4:
        label = f"[yellow]{score:.2f} ⚠️[/yellow]"
    else:
        label = f"[red]{score:.2f} ❌[/red]"

    console.print(f"   Score : {label}")
    console.print(f"   Tokens: [yellow]{total_tokens}[/yellow] | Cost: [yellow]${cost:.6f}[/yellow]")

    # append warning to answer if low confidence
    answer = state["answer"]

    if score < 0.7:
        answer += "\n\n⚠️ Low confidence — answer may not be fully supported by resume."

    return {
        **state,
        "answer"          : answer,
        "confidence_score": score,
        "prompt_tokens"   : state["prompt_tokens"] + prompt_tokens,
        "completion_tokens": state["completion_tokens"] + completion_tokens,
        "total_tokens"    : state["total_tokens"] + total_tokens,
        "estimated_cost"  : state["estimated_cost"] + cost,
    }