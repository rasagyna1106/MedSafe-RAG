"""
Field-level chunking for MedSafe RAG.

Each (drug, label section) pair is chunked separately with structural prefixes
and citation-ready metadata attached to every chunk.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

LABEL_FIELDS: list[str] = [
    "warnings",
    "drug_interactions",
    "contraindications",
    "adverse_reactions",
    "dosage_and_administration",
    "indications_and_usage",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "do_not_use",
    "stop_use",
]

DEFAULT_FILTERED_PATH = Path("data/processed/drugs_filtered.json")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.json")

CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _token_length(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
    )


def _drug_display_name(record: dict[str, Any]) -> str:
    return record["matched_target"]


def _prefix_chunk_text(drug_name: str, section_name: str, field_text: str) -> str:
    return f"[Drug: {drug_name}][Section: {section_name}] {field_text}"


# Salt, hydrate, and dosage-form words stripped before deciding combo vs single-ingredient.
# e.g. "WARFARIN SODIUM" -> not a combo; "IBUPROFEN FAMOTIDINE" -> combo (famotidine remains).
_NON_INGREDIENT_TOKENS = frozenset(
    {
        "and",
        "with",
        "plus",
        "hcl",
        "hbr",
        "hydrochloride",
        "hydrobromide",
        "dihydrate",
        "monohydrate",
        "trihydrate",
        "anhydrous",
        "sodium",
        "potassium",
        "calcium",
        "magnesium",
        "sulfate",
        "phosphate",
        "citrate",
        "maleate",
        "succinate",
        "fumarate",
        "tartrate",
        "besylate",
        "mesylate",
        "granule",
        "granules",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "oral",
        "solution",
        "suspension",
        "injection",
        "extended",
        "release",
        "er",
        "xr",
        "extra",
        "strength",
    }
)

_STRENGTH_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|unit|units|%)(?:/\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL))?|\b\d+\s*mg\b",
    re.IGNORECASE,
)


def _full_product_name(generic_names: list[str]) -> str:
    return ", ".join(name.strip() for name in generic_names if name.strip())


def _remaining_ingredient_text(matched_target: str, generic_names: list[str]) -> str:
    """Return generic_name text left after removing matched_target and non-ingredient tokens."""
    joined = ", ".join(generic_names).lower()
    joined = re.sub(rf"\b{re.escape(matched_target.lower())}\b", " ", joined)
    joined = joined.replace("&", " and ")
    # Treat comma, semicolon, slash, and plus as ingredient separators (FDA uses all of these).
    joined = re.sub(r"[,/;+]", " ", joined)
    joined = _STRENGTH_PATTERN.sub(" ", joined)

    remaining_tokens: list[str] = []
    for token in re.split(r"[\s\-]+", joined):
        cleaned = re.sub(r"[^a-z0-9]", "", token)
        if not cleaned or cleaned in _NON_INGREDIENT_TOKENS:
            continue
        if cleaned.isdigit():
            continue
        remaining_tokens.append(cleaned)

    return " ".join(remaining_tokens)


def _is_combo_product(matched_target: str, generic_names: list[str]) -> bool:
    """
    True when generic_name lists matched_target plus at least one other active ingredient.
    Salt forms (e.g. warfarin sodium) and strength-only variants are not treated as combos.
    """
    if not generic_names:
        return False
    return bool(_remaining_ingredient_text(matched_target, generic_names))


def _product_metadata(record: dict[str, Any]) -> dict[str, Any]:
    openfda = record.get("openfda") or {}
    generic_names = openfda.get("generic_name") or []
    matched_target = record["matched_target"]
    return {
        "full_product_name": _full_product_name(generic_names),
        "is_combo_product": _is_combo_product(matched_target, generic_names),
    }


def chunk_record(record: dict[str, Any], splitter: RecursiveCharacterTextSplitter | None = None) -> list[Chunk]:
    """Chunk one drug record field-by-field."""
    splitter = splitter or _build_text_splitter()
    drug_name = _drug_display_name(record)
    openfda = record.get("openfda") or {}
    generic_names = openfda.get("generic_name") or []
    brand_names = openfda.get("brand_name") or []
    product_metadata = _product_metadata(record)

    chunks: list[Chunk] = []

    # Chunk each label section separately so retrieval returns focused citations
    # (e.g. only drug_interactions), not a wall of mixed warnings + dosing text.
    for section_name in LABEL_FIELDS:
        field_text = (record.get(section_name) or "").strip()
        if not field_text:
            continue

        # Split raw field text first, then prefix each piece — if we prefixed before
        # splitting, only chunk 1 would carry [Drug][Section] for citation grounding.
        split_texts = splitter.split_text(field_text)

        for piece in split_texts:
            chunk_text = _prefix_chunk_text(drug_name, section_name, piece)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "drug": drug_name,
                        "section": section_name,
                        "generic_name": generic_names,
                        "brand_names": brand_names,
                        "full_product_name": product_metadata["full_product_name"],
                        "is_combo_product": product_metadata["is_combo_product"],
                    },
                )
            )

    return chunks


def chunk_drugs(
    records: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Chunk]:
    """Chunk all drug records field-by-field."""
    splitter = splitter or _build_text_splitter()
    all_chunks: list[Chunk] = []

    for record in records:
        all_chunks.extend(chunk_record(record, splitter=splitter))

    return all_chunks


def load_and_chunk(
    filtered_path: Path = DEFAULT_FILTERED_PATH,
) -> list[Chunk]:
    filtered_path = Path(filtered_path)
    with filtered_path.open(encoding="utf-8") as infile:
        records = json.load(infile)
    return chunk_drugs(records)


def save_chunks(chunks: list[Chunk], output_path: Path = DEFAULT_CHUNKS_PATH) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks]
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, ensure_ascii=False)


def _print_inspection_chunk(chunk: Chunk, index: int, total: int, label: str) -> None:
    drug = chunk.metadata["drug"]
    section = chunk.metadata["section"]
    print(f"=== Chunk {index}/{total} for ({drug}, {section}) [{label}] ===")
    print(chunk.text, flush=True)
    print(f"Metadata: {chunk.metadata}")
    print("---")


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _filter_chunks(chunks: list[Chunk], drug: str, section: str, limit: int | None = None) -> list[Chunk]:
    matched = [
        chunk
        for chunk in chunks
        if chunk.metadata["drug"] == drug and chunk.metadata["section"] == section
    ]
    if limit is not None:
        return matched[:limit]
    return matched


def _find_multichunk_field(
    records: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    min_chunks: int = 3,
) -> tuple[str, str, list[Chunk]] | None:
    best: tuple[str, str, list[Chunk]] | None = None

    for record in records:
        record_chunks = chunk_record(record, splitter=splitter)
        grouped: dict[tuple[str, str], list[Chunk]] = {}
        for chunk in record_chunks:
            key = (chunk.metadata["drug"], chunk.metadata["section"])
            grouped.setdefault(key, []).append(chunk)

        for (drug, section), section_chunks in grouped.items():
            if len(section_chunks) < min_chunks:
                continue
            if best is None or len(section_chunks) > len(best[2]):
                best = (drug, section, section_chunks)

    return best


def _find_acetaminophen_combo_warfarin_chunk(
    records: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
) -> Chunk | None:
    for record in records:
        if record.get("matched_target") != "acetaminophen":
            continue

        generic_names = record.get("openfda", {}).get("generic_name") or []
        generic_joined = " ".join(generic_names).lower()
        if generic_joined.strip() == "acetaminophen":
            continue

        field_text = (record.get("ask_doctor_or_pharmacist") or "").strip()
        if "warfarin" not in field_text.lower():
            continue

        section_chunks = [
            chunk
            for chunk in chunk_record(record, splitter=splitter)
            if chunk.metadata["section"] == "ask_doctor_or_pharmacist"
        ]
        if not section_chunks:
            continue

        for chunk in section_chunks:
            if "warfarin" in chunk.text.lower():
                return chunk

        return section_chunks[0]

    return None


def inspect_chunks(
    chunks: list[Chunk] | None = None,
    records: list[dict[str, Any]] | None = None,
    filtered_path: Path = DEFAULT_FILTERED_PATH,
) -> None:
    """Print full chunk text for manual spot-checking."""
    _configure_utf8_stdout()
    splitter = _build_text_splitter()

    if records is None:
        filtered_path = Path(filtered_path)
        with filtered_path.open(encoding="utf-8") as infile:
            records = json.load(infile)

    if chunks is None:
        chunks = chunk_drugs(records, splitter=splitter)

    print("=== Inspection 1: warfarin / drug_interactions (first 2 chunks) ===")
    warfarin_interaction_chunks = _filter_chunks(chunks, "warfarin", "drug_interactions", limit=2)
    for index, chunk in enumerate(warfarin_interaction_chunks, start=1):
        _print_inspection_chunk(
            chunk,
            index,
            len(warfarin_interaction_chunks),
            "warfarin / drug_interactions",
        )

    print("=== Inspection 2: ibuprofen / adverse_reactions (first 2 chunks) ===")
    ibuprofen_adverse_chunks = _filter_chunks(chunks, "ibuprofen", "adverse_reactions", limit=2)
    for index, chunk in enumerate(ibuprofen_adverse_chunks, start=1):
        _print_inspection_chunk(
            chunk,
            index,
            len(ibuprofen_adverse_chunks),
            "ibuprofen / adverse_reactions",
        )

    multichunk = _find_multichunk_field(records, splitter=splitter, min_chunks=3)
    print("=== Inspection 3: multi-chunk field split (all chunks in order) ===")
    if multichunk is None:
        print("No (drug, section) field split into 3+ chunks was found.")
        print("---")
    else:
        drug, section, section_chunks = multichunk
        print(f"Selected field: ({drug}, {section}) -> {len(section_chunks)} chunks")
        for index, chunk in enumerate(section_chunks, start=1):
            _print_inspection_chunk(
                chunk,
                index,
                len(section_chunks),
                f"multi-chunk example ({drug}, {section})",
            )

    print("=== Inspection 4: acetaminophen combo / ask_doctor_or_pharmacist ===")
    combo_chunk = _find_acetaminophen_combo_warfarin_chunk(records, splitter=splitter)
    if combo_chunk is None:
        print("No acetaminophen combo chunk with warfarin warning found.")
        print("---")
    else:
        _print_inspection_chunk(
            combo_chunk,
            1,
            1,
            "acetaminophen combo / ask_doctor_or_pharmacist",
        )


def _print_combo_enrichment_summary(chunks: list[Chunk]) -> None:
    per_drug: dict[str, Counter] = defaultdict(Counter)

    for chunk in chunks:
        drug = chunk.metadata["drug"]
        if chunk.metadata.get("is_combo_product"):
            per_drug[drug]["combo"] += 1
        else:
            per_drug[drug]["single"] += 1

    print("\nChunks by is_combo_product (per drug):")
    for drug in sorted(per_drug):
        counts = per_drug[drug]
        print(
            f"  {drug}: combo={counts['combo']}, single={counts['single']}, "
            f"total={counts['combo'] + counts['single']}"
        )

    print("\nSample metadata checks:")
    examples = [
        ("warfarin", False, "plain warfarin"),
        ("ibuprofen", True, "ibuprofen combo (e.g. famotidine)"),
        ("rivaroxaban", None, "rivaroxaban branded (Xarelto)"),
        ("acetaminophen", True, "acetaminophen combo"),
    ]
    for drug, expected_combo, label in examples:
        sample = next(
            (
                chunk
                for chunk in chunks
                if chunk.metadata["drug"] == drug
                and (
                    expected_combo is None
                    or chunk.metadata.get("is_combo_product") is expected_combo
                )
            ),
            None,
        )
        if sample is None:
            print(f"  [{label}] no matching chunk found")
            continue
        print(f"  [{label}] {sample.metadata}")


def _print_chunk_summary(chunks: list[Chunk]) -> None:
    print(f"Total chunks: {len(chunks)}")

    by_drug = Counter(chunk.metadata["drug"] for chunk in chunks)
    by_section = Counter(chunk.metadata["section"] for chunk in chunks)

    print("\nChunks by drug (top 10):")
    for drug, count in by_drug.most_common(10):
        print(f"  {drug}: {count}")

    print("\nChunks by section:")
    for section, count in by_section.most_common():
        print(f"  {section}: {count}")

    if chunks:
        sample = chunks[0]
        print("\nSample chunk:")
        print(f"  metadata: {sample.metadata}")
        preview = sample.text[:240].replace("\n", " ")
        print(f"  text: {preview}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk filtered FDA drug labels.")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print full chunk text for manual spot-checking.",
    )
    args = parser.parse_args()

    if args.inspect:
        inspect_chunks()
    else:
        chunks = load_and_chunk()
        save_chunks(chunks)
        print(f"Wrote {len(chunks)} chunks to {DEFAULT_CHUNKS_PATH}")
        _print_chunk_summary(chunks)
        _print_combo_enrichment_summary(chunks)
