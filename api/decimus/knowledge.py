"""RAG knowledge base for orchestration rules.

Queries the ChromaDB database of Rimsky-Korsakov's 'Principles of Orchestration'
to inform instrument assignments, doublings, and texture decisions.
"""

import os
from dataclasses import dataclass

_collection = None
_collection_unavailable = False
_DB_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'knowledge_base', 'chromadb'
)
_COLLECTION_NAME = 'orchestration_rules'


def is_available() -> bool:
    """True if the ChromaDB store has been built and is loadable."""
    return os.path.isdir(_DB_PATH) and not _collection_unavailable


def _get_collection():
    """Lazy-load the ChromaDB collection. Returns None if unavailable."""
    global _collection, _collection_unavailable
    if _collection is not None:
        return _collection
    if _collection_unavailable:
        return None
    if not os.path.isdir(_DB_PATH):
        _collection_unavailable = True
        return None
    try:
        import chromadb
        client = chromadb.PersistentClient(path=os.path.abspath(_DB_PATH))
        _collection = client.get_collection(_COLLECTION_NAME)
        return _collection
    except Exception:
        _collection_unavailable = True
        return None


@dataclass
class OrchestrationAdvice:
    """A piece of orchestration advice from the knowledge base."""
    text: str
    source: str = ""
    relevance: float = 0.0


def query_rules(question: str, n_results: int = 5) -> list[OrchestrationAdvice]:
    """Query the knowledge base for orchestration advice.

    Args:
        question: Natural language query about orchestration.
        n_results: Number of results to return.

    Returns:
        List of OrchestrationAdvice sorted by relevance.
    """
    col = _get_collection()
    if col is None:
        return []
    results = col.query(query_texts=[question], n_results=n_results)

    advice_list = []
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i] if results['metadatas'] else {}
        source = metadata.get('source', '')

        advice_list.append(OrchestrationAdvice(
            text=doc,
            source=source,
            relevance=1.0 / (1.0 + distance),  # convert distance to 0-1 relevance
        ))

    return advice_list


def get_melody_doubling_advice(key: str, instrument: str) -> list[OrchestrationAdvice]:
    """Get advice on how to double a melody played by a specific instrument."""
    return query_rules(
        f"melody doubling for {instrument} in {key}",
        n_results=5,
    )


def get_bass_advice(key: str) -> list[OrchestrationAdvice]:
    """Get advice on bass line orchestration."""
    return query_rules(
        f"bass line orchestration in {key} key, cello contrabass bassoon",
        n_results=5,
    )


def get_harmony_advice(key: str, density: str = "moderate") -> list[OrchestrationAdvice]:
    """Get advice on harmonic accompaniment orchestration."""
    return query_rules(
        f"harmonic accompaniment {density} density in {key}, inner voices",
        n_results=5,
    )


def get_instrument_combination_advice(
    instruments: list[str],
) -> list[OrchestrationAdvice]:
    """Get advice on combining specific instruments."""
    inst_str = ", ".join(instruments)
    return query_rules(
        f"combining {inst_str} together, instrument blend and balance",
        n_results=5,
    )


def get_texture_advice(texture_type: str) -> list[OrchestrationAdvice]:
    """Get advice on a specific orchestral texture type."""
    return query_rules(
        f"orchestral texture: {texture_type}",
        n_results=5,
    )


def get_style_advice(style: str) -> list[OrchestrationAdvice]:
    """Get general advice for an orchestration style."""
    style_queries = {
        "romantic": "romantic orchestration full rich sound, Tchaikovsky Brahms style",
        "classical": "classical orchestration clear transparent texture, Mozart Haydn style",
        "modern": "modern orchestration bold colors unusual instrument combinations",
        "film": "dramatic orchestration power unison doublings, large orchestra",
    }
    query = style_queries.get(style, f"{style} orchestration style")
    return query_rules(query, n_results=5)


def summarize_advice(advice_list: list[OrchestrationAdvice], max_chars: int = 500) -> str:
    """Summarize a list of advice into a brief text."""
    if not advice_list:
        return "No specific advice found."

    # Take the most relevant pieces
    texts = []
    total = 0
    for a in advice_list:
        if total + len(a.text) > max_chars:
            break
        texts.append(a.text.strip())
        total += len(a.text)

    return " | ".join(texts) if texts else advice_list[0].text[:max_chars]
