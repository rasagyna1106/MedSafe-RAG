"""
Embed chunked FDA label text into a local FAISS index for MedSafe RAG retrieval.

Build once from chunks.json; load from disk for fast query-time search.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.json")
INDEX_DIR = Path("data/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

# Proposal specifies top-5 retrieval — enough context for the LLM without flooding the prompt.
DEFAULT_TOP_K = 5

METADATA_FIELDS = (
    "text",
    "drug",
    "section",
    "generic_name",
    "brand_names",
    "full_product_name",
    "is_combo_product",
)


@dataclass
class VectorStore:
    index: faiss.Index
    metadata: list[dict[str, Any]]
    model: SentenceTransformer


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _load_chunks(chunks_path: Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, Any]]:
    chunks_path = Path(chunks_path)
    with chunks_path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _metadata_entry(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("metadata") or {}
    return {
        "text": chunk.get("text", ""),
        "drug": meta.get("drug", ""),
        "section": meta.get("section", ""),
        "generic_name": meta.get("generic_name", []),
        "brand_names": meta.get("brand_names", []),
        "full_product_name": meta.get("full_product_name", ""),
        "is_combo_product": bool(meta.get("is_combo_product", False)),
    }


def _embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    show_progress_bar: bool = True,
) -> np.ndarray:
    # L2-normalize so inner product == cosine similarity (pairs with IndexFlatIP below).
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dimension = vectors.shape[1]
    # IndexFlatIP on unit vectors ranks by cosine similarity (higher score = closer match).
    # IndexFlatL2 would work on the same normalized vectors but returns distance, not similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    model: SentenceTransformer | None = None,
) -> VectorStore:
    """Embed all chunks and persist the FAISS index + sidecar metadata."""
    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    model = model or _load_embedding_model()
    # Embed stored chunk text verbatim — [Drug][Section] prefix helps drug/section-aware retrieval.
    texts = [chunk["text"] for chunk in chunks]
    metadata = [_metadata_entry(chunk) for chunk in chunks]

    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL_NAME}...")
    vectors = _embed_texts(model, texts)
    index = _build_faiss_index(vectors)

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile, indent=2, ensure_ascii=False)

    print(f"Wrote FAISS index ({index.ntotal} vectors) to {index_path}")
    print(f"Wrote metadata sidecar to {metadata_path}")

    return VectorStore(index=index, metadata=metadata, model=model)


def load_vector_store(
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
    model: SentenceTransformer | None = None,
) -> VectorStore:
    """Load a persisted index — used by --query so search does not rebuild embeddings."""
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run: python src/embed.py --build"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata sidecar not found at {metadata_path}. Run: python src/embed.py --build"
        )

    index = faiss.read_index(str(index_path))
    with metadata_path.open(encoding="utf-8") as infile:
        metadata = json.load(infile)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: {index.ntotal} vectors vs {len(metadata)} metadata rows"
        )

    model = model or _load_embedding_model()
    return VectorStore(index=index, metadata=metadata, model=model)


def retrieve(
    query: str,
    k: int = DEFAULT_TOP_K,
    store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """
    Embed the query, search the FAISS index, return the top-k chunks with metadata and scores.

    Brand-to-generic rewriting (resolve_query) is NOT done here — generate.py owns that
    wiring so retrieval always receives an already-normalized generic drug query.
    """
    store = store or load_vector_store()
    query_vector = _embed_texts(store.model, [query], show_progress_bar=False)
    scores, indices = store.index.search(query_vector, k)

    results: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
        if idx < 0:
            continue
        row = dict(store.metadata[int(idx)])
        row["score"] = float(score)
        row["rank"] = rank
        results.append(row)

    return results


def _strip_chunk_prefix(text: str) -> str:
    """Remove [Drug][Section] prefix for CLI display; stored chunk text is unchanged."""
    return re.sub(r"^\[Drug:[^\]]+\]\[Section:[^\]]+\]\s*", "", text, count=1)


def _display_snippet(text: str, max_chars: int = 150) -> str:
    compact = " ".join(_strip_chunk_prefix(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _format_full_text_for_display(text: str) -> str:
    """Break FDA label chunk text onto separate lines at each bullet (•)."""
    parts = text.split("•")
    if len(parts) == 1:
        return text.strip()

    lines: list[str] = []
    lead = parts[0].strip()
    if lead:
        lines.append(lead)
    lines.extend(f"• {part.strip()}" for part in parts[1:] if part.strip())
    return "\n".join(lines)


def _truncate_cell(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 4:
        return value[:width]
    return value[: width - 4] + "..."


def print_query_results(
    query: str,
    results: list[dict[str, Any]],
    *,
    full_text: bool = False,
) -> None:
    print(f'Query: "{query}"')
    print()
    print(f"{'Rank':<6}{'Score':<8}{'Drug':<11}{'Section':<20}{'Combo':<7}{'Product'}")
    print(f"{'----':<6}{'------':<8}{'---------':<11}{'------------------':<20}{'-----':<7}{'----------------'}")

    for row in results:
        combo = "Yes" if row.get("is_combo_product") else "No"
        product = _truncate_cell(str(row.get("full_product_name", "")), 18)
        print(
            f"{row['rank']:<6}"
            f"{row['score']:<8.4f}"
            f"{_truncate_cell(str(row['drug']), 11):<11}"
            f"{_truncate_cell(str(row['section']), 20):<20}"
            f"{combo:<7}"
            f"{product}"
        )

    if results:
        print()
        if full_text:
            print("Top result text:")
            for line in _format_full_text_for_display(results[0]["text"]).split("\n"):
                print(f"  {line}")
        else:
            print("Top result snippet:")
            print(f'  "{_display_snippet(results[0]["text"])}"')


def main() -> None:
    _configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Build and query the MedSafe FAISS vector index.")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Embed chunks.json and write data/index/faiss.index + metadata.json.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Load the saved index and print top-k retrieval results for this question.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="With --query, print the complete top-result chunk text instead of a 150-character snippet.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks.json (used only with --build).",
    )
    args = parser.parse_args()

    if args.build:
        build_index(chunks_path=args.chunks_path)
        return

    if args.query:
        store = load_vector_store()
        results = retrieve(args.query, k=args.k, store=store)
        print_query_results(args.query, results, full_text=args.full_text)
        return

    parser.error("Provide --build to create the index or --query \"...\" to search it.")


if __name__ == "__main__":
    main()
