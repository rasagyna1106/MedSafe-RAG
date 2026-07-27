"""
Stream openFDA drug labels, filter to target generics, and save a small JSON file
for all downstream pipeline steps.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ijson

# Case-insensitive substring match against openfda.generic_name values.
TARGET_DRUGS: list[str] = [
    # Blood thinners
    "warfarin",
    "apixaban",
    "rivaroxaban",
    # Diabetes
    "metformin",
    "insulin glargine",
    "glipizide",
    # Blood pressure
    "lisinopril",
    "amlodipine",
    "metoprolol",
    # Cholesterol
    "atorvastatin",
    "simvastatin",
    # Pain / inflammation
    "ibuprofen",
    "naproxen",
    "acetaminophen",
    # Antibiotics
    "amoxicillin",
    "azithromycin",
    # Mental health
    "sertraline",
    "lorazepam",
]

RAW_FILES: list[str] = [
    "data/raw/drug-label-0001-of-0013.json",
    "data/raw/drug-label-0002-of-0013.json",
    "data/raw/drug-label-0003-of-0013.json",
]

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

OPENFDA_FIELDS: list[str] = [
    "brand_name",
    "generic_name",
    "substance_name",
]

MAX_RECORDS_PER_DRUG = 20
MIN_COMBO_RECORDS_PER_DRUG = 3

# apixaban: 0 records found in openFDA Parts 1-3 of 13. Documented as a known dataset
# coverage gap (see project report, Section 8). Do not swap or download further parts.
APIXABAN_COVERAGE_NOTE = (
    "apixaban: 0 records found in openFDA Parts 1-3 of 13. Documented as a known "
    "dataset coverage gap (see project report, Section 8)."
)

DEFAULT_OUTPUT_PATH = Path("data/processed/drugs_filtered.json")
SUMMARY_CSV_PATH = Path("data/processed/drugs_filtered_summary.csv")
SUMMARY_SIZE_THRESHOLD_BYTES = 50 * 1024 * 1024

COVERAGE_CHECK_DRUGS: list[str] = ["apixaban", "insulin glargine"]


@dataclass
class IngestStats:
    per_file_counts: dict[str, int] = field(default_factory=dict)
    total_before_dedup: int = 0
    unique_set_ids: int = 0
    total_after_dedup: int = 0
    per_drug_before_cap: dict[str, int] = field(default_factory=dict)
    per_drug_after_cap: dict[str, int] = field(default_factory=dict)
    total_after_cap: int = 0
    output_path: Path = DEFAULT_OUTPUT_PATH
    output_size_bytes: int = 0
    summary_csv_path: Path | None = None
    present_targets: set[str] = field(default_factory=set)
    missing_targets: set[str] = field(default_factory=set)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _extract_text_field(record: dict[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(part).strip() for part in value if part]
        return "\n".join(parts)
    return str(value).strip()


def _matches_target_drug(generic_names: list[str]) -> str | None:
    joined = " ".join(generic_names).lower()
    for target in TARGET_DRUGS:
        if target in joined:
            return target
    return None


def _dedupe_key(record: dict[str, Any]) -> str:
    # set_id groups all versions of the same SPL label; id alone can repeat across revisions.
    set_id = str(record.get("set_id") or "").strip()
    if set_id:
        return set_id
    return str(record.get("id") or "").strip()


def _effective_time(record: dict[str, Any]) -> str:
    return str(record.get("effective_time") or "").strip()


def _effective_time_sort_key(record: dict[str, Any]) -> int:
    value = _effective_time(record)
    if value.isdigit():
        return int(value)
    return 0


def _ingredient_count(record: dict[str, Any]) -> int:
    substance_names = record.get("openfda", {}).get("substance_name") or []
    if substance_names:
        return len(substance_names)

    generic_names = record.get("openfda", {}).get("generic_name") or []
    if not generic_names:
        return 999

    joined = " ".join(generic_names)
    normalized = joined.replace(" AND ", ",").replace(" and ", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    return len(parts) if parts else 999


def _target_position_in_generic_name(record: dict[str, Any]) -> int:
    target = record["matched_target"].lower()
    generic_names = record.get("openfda", {}).get("generic_name") or []
    positions = [name.lower().find(target) for name in generic_names if target in name.lower()]
    return min(positions) if positions else 9999


def _cap_selection_rank(record: dict[str, Any]) -> tuple[int, int, int]:
    # Lower ingredient count first (prefer plain labels), then earlier target match, then recency.
    return (
        _ingredient_count(record),
        _target_position_in_generic_name(record),
        -_effective_time_sort_key(record),
    )


def _count_by_target(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {target: 0 for target in TARGET_DRUGS}
    for record in records:
        target = record["matched_target"]
        if target in counts:
            counts[target] += 1
    return counts


def _cap_per_drug_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep at most MAX_RECORDS_PER_DRUG per target. Reserve MIN_COMBO_RECORDS_PER_DRUG slots
    # for 2–3 ingredient products first so OTC combo warnings (e.g. acetaminophen + warfarin)
    # are not crowded out by hundreds of single-ingredient store-brand labels.
    grouped: dict[str, list[dict[str, Any]]] = {target: [] for target in TARGET_DRUGS}

    for record in records:
        target = record["matched_target"]
        if target in grouped:
            grouped[target].append(record)

    capped: list[dict[str, Any]] = []
    for target in TARGET_DRUGS:
        group = grouped[target]
        if len(group) <= MAX_RECORDS_PER_DRUG:
            capped.extend(group)
            continue

        singles = [record for record in group if _ingredient_count(record) == 1]
        combos = [
            record
            for record in group
            if 2 <= _ingredient_count(record) <= 3
        ]
        others = [
            record
            for record in group
            if _ingredient_count(record) not in {1, 2, 3}
        ]

        selected: list[dict[str, Any]] = []
        selected.extend(sorted(combos, key=_cap_selection_rank)[:MIN_COMBO_RECORDS_PER_DRUG])

        remaining_slots = MAX_RECORDS_PER_DRUG - len(selected)
        # Prefer fewer-ingredient singles and earlier generic_name matches, then newest labels.
        selected.extend(sorted(singles, key=_cap_selection_rank)[:remaining_slots])

        if len(selected) < MAX_RECORDS_PER_DRUG:
            already_selected = {_dedupe_key(record) for record in selected}
            filler_pool = [
                record
                for record in sorted(combos + others + singles, key=_cap_selection_rank)
                if _dedupe_key(record) not in already_selected
            ]
            selected.extend(filler_pool[: MAX_RECORDS_PER_DRUG - len(selected)])

        capped.extend(selected[:MAX_RECORDS_PER_DRUG])

    return capped


def _extract_record(record: dict[str, Any], matched_target: str) -> dict[str, Any]:
    openfda = record.get("openfda") or {}

    extracted: dict[str, Any] = {
        "matched_target": matched_target,
        "set_id": record.get("set_id", ""),
        "id": record.get("id", ""),
        "effective_time": _effective_time(record),
    }

    for field_name in LABEL_FIELDS:
        extracted[field_name] = _extract_text_field(record, field_name)

    extracted["openfda"] = {
        field_name: _as_str_list(openfda.get(field_name))
        for field_name in OPENFDA_FIELDS
    }

    return extracted


def _stream_file(raw_path: Path) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []

    # Raw openFDA parts are hundreds of MB each — stream with ijson instead of json.load().
    with raw_path.open("rb") as raw_file:
        for record in ijson.items(raw_file, "results.item"):
            openfda = record.get("openfda") or {}
            generic_names = _as_str_list(openfda.get("generic_name"))
            matched_target = _matches_target_drug(generic_names)
            if matched_target is None:
                continue

            matches.append(_extract_record(record, matched_target))

    return matches


def _dedupe_by_set_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[str, dict[str, Any]] = {}

    for record in records:
        key = _dedupe_key(record)
        if not key:
            key = f"__missing_key__:{record.get('id', '')}:{id(record)}"

        current_best = best_by_key.get(key)
        if current_best is None:
            best_by_key[key] = record
            continue

        # Same SPL can appear in multiple raw files; keep the newest effective_time revision.
        if _effective_time(record) >= _effective_time(current_best):
            best_by_key[key] = record

    return list(best_by_key.values())


def _write_summary_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["drug_name", "set_id", "generic_name", "brand_names", "matched_target"],
        )
        writer.writeheader()

        for record in records:
            generic_names = record["openfda"]["generic_name"]
            brand_names = record["openfda"]["brand_name"]
            writer.writerow(
                {
                    "drug_name": record["matched_target"],
                    "set_id": record.get("set_id", ""),
                    "generic_name": "; ".join(generic_names),
                    "brand_names": "; ".join(brand_names[:5]),
                    "matched_target": record["matched_target"],
                }
            )


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} bytes"


def ingest(
    raw_files: list[str | Path] | None = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> tuple[list[dict[str, Any]], IngestStats]:
    """Stream raw openFDA JSON files and write deduplicated filtered records to disk."""
    raw_files = [Path(path) for path in (raw_files or RAW_FILES)]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = IngestStats(output_path=output_path)
    all_matches: list[dict[str, Any]] = []
    running_total = 0

    for raw_path in raw_files:
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw input file not found: {raw_path}")

        print(f"Processing {raw_path.name}...")
        file_matches = _stream_file(raw_path)
        running_total += len(file_matches)
        stats.per_file_counts[raw_path.name] = len(file_matches)
        all_matches.extend(file_matches)
        print(f"  -> Found {len(file_matches)} matching records (running total: {running_total})")

    stats.total_before_dedup = len(all_matches)
    stats.unique_set_ids = len({_dedupe_key(record) for record in all_matches})

    deduped = _dedupe_by_set_id(all_matches)
    stats.total_after_dedup = len(deduped)
    stats.per_drug_before_cap = _count_by_target(deduped)

    capped = _cap_per_drug_records(deduped)
    stats.per_drug_after_cap = _count_by_target(capped)
    stats.total_after_cap = len(capped)

    present_targets = {record["matched_target"] for record in capped}
    stats.present_targets = present_targets
    stats.missing_targets = {
        target for target in COVERAGE_CHECK_DRUGS if target not in present_targets
    }

    print("\n=== Per-Drug Cap (max 20 per matched_target) ===")
    for target in TARGET_DRUGS:
        before = stats.per_drug_before_cap.get(target, 0)
        after = stats.per_drug_after_cap.get(target, 0)
        if before > MAX_RECORDS_PER_DRUG:
            print(f"  {target}: {before} -> {after} (capped)")
        else:
            print(f"  {target}: {before} -> {after}")

    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(capped, out_file, indent=2, ensure_ascii=False)

    stats.output_size_bytes = output_path.stat().st_size
    if stats.output_size_bytes > SUMMARY_SIZE_THRESHOLD_BYTES:
        _write_summary_csv(capped, SUMMARY_CSV_PATH)
        stats.summary_csv_path = SUMMARY_CSV_PATH

    return capped, stats


def print_ingest_summary(stats: IngestStats) -> None:
    print("\n=== Ingest Summary ===")

    for index, filename in enumerate(stats.per_file_counts, start=1):
        count = stats.per_file_counts[filename]
        print(f"Part {index} ({filename}): {count} matches")

    print(f"Total raw records before dedup: {stats.total_before_dedup}")
    print(f"Unique set_id groups after dedup: {stats.unique_set_ids}")
    print(f"Records after dedup: {stats.total_after_dedup}")
    print(f"Final records written (after per-drug cap): {stats.total_after_cap}")
    print(f"Output file: {stats.output_path}")
    print(f"Output size: {_format_bytes(stats.output_size_bytes)}")

    print("\n=== Final Per-Drug Record Counts ===")
    for target in TARGET_DRUGS:
        count = stats.per_drug_after_cap.get(target, 0)
        print(f"  {target}: {count}")

    if stats.summary_csv_path is not None:
        print(
            f"Output exceeds 50 MB — summary CSV saved to {stats.summary_csv_path}"
        )

    print("\n=== Coverage Check ===")
    for target in COVERAGE_CHECK_DRUGS:
        count = stats.per_drug_after_cap.get(target, 0)
        if target in stats.present_targets:
            print(f"  {target}: PRESENT ({count} record(s))")
        else:
            print(f"  {target}: 0 found after Parts 1-3")

    print(f"\n{APIXABAN_COVERAGE_NOTE}")

    if "insulin glargine" not in stats.present_targets:
        print(
            "WARNING: insulin glargine is missing after Parts 1-3 — unexpected if Part 3 is loaded."
        )


def _preview_field(text: str, max_chars: int = 240) -> str:
    if not text:
        return "[empty]"
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def print_sample_drugs(
    records: list[dict[str, Any]],
    sample_targets: list[str] | None = None,
    max_samples: int = 4,
) -> None:
    """Print one sample record per requested target drug for manual verification."""
    sample_targets = sample_targets or ["warfarin", "ibuprofen", "metformin", "acetaminophen"]
    seen: set[str] = set()

    print(f"\n=== Sample Records ({len(records)} total after cap) ===\n")

    for target in sample_targets:
        match = next((record for record in records if record["matched_target"] == target), None)
        if match is None:
            print(f"=== {target.upper()} ===")
            print("No matching record found in filtered output.\n")
            continue

        seen.add(target)
        generic_names = match["openfda"]["generic_name"]
        brand_names = match["openfda"]["brand_name"]

        print(f"=== {target.upper()} ===")
        print(f"matched_target: {match['matched_target']}")
        print(f"generic_name: {generic_names[:3]}")
        print(f"brand_name: {brand_names[:3]}")
        print(f"set_id: {match.get('set_id', '')}")
        print(f"effective_time: {match.get('effective_time', '')}")

        for field_name in LABEL_FIELDS:
            print(f"{field_name}: {_preview_field(match[field_name])}")

        print()

    if len(seen) < max_samples:
        extras = [
            record
            for record in records
            if record["matched_target"] not in seen
        ][: max_samples - len(seen)]
        for match in extras:
            target = match["matched_target"]
            print(f"=== {target.upper()} (extra sample) ===")
            print(f"generic_name: {match['openfda']['generic_name'][:3]}")
            print(f"brand_name: {match['openfda']['brand_name'][:3]}")
            for field_name in LABEL_FIELDS:
                print(f"{field_name}: {_preview_field(match[field_name])}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter openFDA drug labels to target generics.")
    parser.add_argument(
        "--raw-files",
        nargs="+",
        type=Path,
        default=[Path(path) for path in RAW_FILES],
        help="Paths to raw openFDA drug-label JSON files (processed in order).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the filtered JSON output.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print sample extracted records after ingestion.",
    )
    args = parser.parse_args()

    records, stats = ingest(raw_files=args.raw_files, output_path=args.output_path)
    print_ingest_summary(stats)

    if args.preview:
        print_sample_drugs(records)


if __name__ == "__main__":
    main()
