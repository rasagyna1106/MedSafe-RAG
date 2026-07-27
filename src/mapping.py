"""
Brand-to-generic query rewriting for MedSafe RAG.

Builds BRAND_TO_GENERIC from openFDA brand_name fields in drugs_filtered.json,
then rewrites caregiver queries before retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_FILTERED_PATH = Path("data/processed/drugs_filtered.json")
DEFAULT_MAPPING_PATH = Path("data/processed/brand_mapping.json")

GENERIC_PREFIX_WORDS = frozenset({"the", "lil"})
STORE_PREFIX_WORDS = frozenset(
    {"drug", "store", "stores", "pharmacy", "health", "travel", "basix", "cvp", "lil"}
)

# Tokens that are not useful standalone brand keywords for rewriting.
SKIP_BRAND_TOKENS = frozenset(
    {
        "and",
        "or",
        "with",
        "the",
        "a",
        "an",
        "of",
        "for",
        "in",
        "mg",
        "ml",
        "hcl",
        "hbr",
        "nsaid",
        "usp",
        "chewable",
        "chewables",
        "childrens",
        "children",
        "infants",
        "kids",
        "pain",
        "relief",
        "fever",
        "arthritis",
        "regular",
        "extra",
        "strength",
        "oral",
        "solution",
        "tablets",
        "tablet",
        "capsules",
        "capsule",
        "sodium",
        "calcium",
        "besylate",
        "hydrochloride",
        "dihydrate",
        "potassium",
        "phosphate",
        "famotidine",
        "pseudoephedrine",
        "clavulanate",
        "ezetimibe",
        "benazepril",
        "glyburide",
        "metformin",
        "atorvastatin",
        "amlodipine",
        "ibuprofen",
        "acetaminophen",
        "naproxen",
        "warfarin",
        "lisinopril",
        "metoprolol",
        "simvastatin",
        "amoxicillin",
        "azithromycin",
        "sertraline",
        "lorazepam",
        "glipizide",
        "rivaroxaban",
        "insulin",
        "glargine",
        "multi",
        "symptom",
        "cold",
        "flu",
        "pm",
        "good",
        "sense",
        "genexa",
        "medline",
        "granule",
        "granules",
        "succinate",
        "tartrate",
        "hydrochlorothiazide",
        "hctz",
        "er",
        "xr",
        "reliever",
        "reducer",
        "sinus",
        "severe",
        "daytime",
        "nighttime",
        "nightime",
        "dose",
        "dosing",
        "film",
        "coated",
        "suspension",
        "powder",
        "injection",
        "solostar",
        "solstar",
        "dosepak",
        "pack",
        "grape",
        "cherry",
        "berry",
        "neo",
        "lubrina",
        "hour",
        "hours",
        "plus",
        "pressure",
    }
)

# Brand keywords that belong to a different active ingredient than the record's
# matched_target (common in combo OTC labels). Skip these when mining FDA data.
OTHER_DRUG_BRAND_KEYWORDS = frozenset(
    {
        "aspirin",
        "caffeine",
        "dextromethorphan",
        "diphenhydramine",
        "guaifenesin",
        "phenylephrine",
        "pseudoephedrine",
        "codeine",
        "hydrocodone",
        "oxycodone",
        "esomeprazole",
        "sudafed",
        "benadryl",
        "mucinex",
        "robitussin",
        "magnesium",
    }
)

# These common consumer brand names (Tylenol, Advil, Motrin, etc.) were
# present in the raw openFDA data but did not survive the per-drug
# record cap applied during ingestion (Step 1). Since these are the
# brand names caregivers are most likely to actually type, they are
# hardcoded here as a deliberate pragmatic addition rather than relying
# solely on FDA-mined keywords. See project report Section 4 for
# rationale.
CAREGIVER_BRAND_ALIASES: dict[str, str] = {
    "tylenol": "acetaminophen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "aleve": "naproxen",
    "coumadin": "warfarin",
    "eliquis": "apixaban",
    "zoloft": "sertraline",
    "lipitor": "atorvastatin",
    "norvasc": "amlodipine",
    "lopressor": "metoprolol",
    "glucophage": "metformin",
    "amoxil": "amoxicillin",
    "ativan": "lorazepam",
    "ozempic": "semaglutide",
}

# Built from drugs_filtered.json — rebuild with: python src/mapping.py
BRAND_TO_GENERIC: dict[str, str] = {
    "acetominophen": "acetaminophen",
    "advil": "ibuprofen",
    "aleve": "naproxen",
    "amoxil": "amoxicillin",
    "ativan": "lorazepam",
    "coumadin": "warfarin",
    "eliquis": "apixaban",
    "glucophage": "metformin",
    "lantus": "insulin glargine",
    "lipitor": "atorvastatin",
    "lopressor": "metoprolol",
    "loreev": "lorazepam",
    "motrin": "ibuprofen",
    "norliqva": "amlodipine",
    "norvasc": "amlodipine",
    "qbrelis": "lisinopril",
    "tylenol": "acetaminophen",
    "xarelto": "rivaroxaban",
    "zithromax": "azithromycin",
    "zocor": "simvastatin",
    "zoloft": "sertraline",
    "ozempic": "semaglutide",
}


def extract_core_brand_keyword(brand_name: str, matched_target: str) -> str | None:
    """
    Extract a short brand keyword from a messy openFDA brand_name string.

    Examples:
      "Tylenol Sinus Severe, CVP HEALTH" -> "tylenol"
      "Lil Drug Store Tylenol Sinus Severe" -> "tylenol"
    """
    if not brand_name or not brand_name.strip():
        return None

    tokens = re.findall(r"[a-z0-9]+", brand_name.lower())
    if not tokens:
        return None

    index = 0
    if tokens[0] in GENERIC_PREFIX_WORDS:
        index += 1

    # Skip store/pharmacy prefixes ("Lil Drug Store Tylenol...") before picking a keyword.
    while index < len(tokens) and tokens[index] in STORE_PREFIX_WORDS | GENERIC_PREFIX_WORDS:
        index += 1

    # First remaining token that is not generic/salt noise becomes the rewrite keyword.
    target_tokens = set(matched_target.lower().split())
    for token in tokens[index:]:
        if token.isdigit() or len(token) < 3:
            continue
        if token in SKIP_BRAND_TOKENS:
            continue
        if token in target_tokens or token == matched_target.lower():
            continue
        if token in OTHER_DRUG_BRAND_KEYWORDS:
            continue
        return token

    return None


def build_brand_to_generic(
    filtered_path: Path = DEFAULT_FILTERED_PATH,
    caregiver_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build brand keyword -> generic mapping from filtered drug records."""
    filtered_path = Path(filtered_path)
    caregiver_aliases = caregiver_aliases or CAREGIVER_BRAND_ALIASES

    with filtered_path.open(encoding="utf-8") as infile:
        records = json.load(infile)

    mapping: dict[str, str] = {}

    for record in records:
        matched_target = record["matched_target"]
        brand_names = record.get("openfda", {}).get("brand_name") or []

        for brand_name in brand_names:
            core_keyword = extract_core_brand_keyword(brand_name, matched_target)
            if core_keyword is None:
                continue

            existing_generic = mapping.get(core_keyword)
            if existing_generic is None:
                mapping[core_keyword] = matched_target
                continue

            # First mapping wins — avoids flip-flopping when the same brand keyword appears
            # on labels for different generics in the capped corpus.
            if existing_generic != matched_target:
                logger.warning(
                    "Brand keyword conflict for %r: keeping %r -> %r (skipped %r from %r)",
                    core_keyword,
                    core_keyword,
                    existing_generic,
                    matched_target,
                    brand_name,
                )

    for brand_keyword, generic_name in caregiver_aliases.items():
        if brand_keyword in mapping:
            continue
        mapping[brand_keyword] = generic_name

    return dict(sorted(mapping.items()))


def save_brand_mapping(
    mapping: dict[str, str],
    output_path: Path = DEFAULT_MAPPING_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(mapping, outfile, indent=2, ensure_ascii=False)
        outfile.write("\n")


def resolve_query(query: str, brand_mapping: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """
    Scan the query for known brand keywords (case-insensitive, whole-word match)
    and replace each with its generic equivalent.

    Returns:
        (rewritten_query, list_of_substitutions_made)
    """
    brand_mapping = brand_mapping or BRAND_TO_GENERIC
    substitutions: list[str] = []
    rewritten = query

    matched_brands = [
        (brand_keyword, generic_name)
        for brand_keyword, generic_name in brand_mapping.items()
        if re.search(rf"\b{re.escape(brand_keyword)}\b", query, flags=re.IGNORECASE)
    ]
    matched_brands.sort(key=lambda item: len(item[0]), reverse=True)

    for brand_keyword, generic_name in matched_brands:
        pattern = re.compile(rf"\b{re.escape(brand_keyword)}\b", flags=re.IGNORECASE)
        if pattern.search(rewritten):
            rewritten = pattern.sub(generic_name, rewritten)
            substitutions.append(f"{brand_keyword} -> {generic_name}")

    return rewritten, substitutions


def _print_mapping_summary(mapping: dict[str, str]) -> None:
    print(f"Built {len(mapping)} brand keyword mappings")
    print("\nSample mappings:")
    for brand_keyword, generic_name in list(mapping.items())[:12]:
        print(f"  {brand_keyword} -> {generic_name}")
    if len(mapping) > 12:
        print(f"  ... and {len(mapping) - 12} more")


def _run_sample_tests() -> None:
    test_queries = [
        "Is Tylenol safe with warfarin?",
        "Can my mom take Advil and her blood pressure medication?",
        "What does metformin do?",
        "Is Motrin safe to take with lisinopril?",
        "Should she stop Xarelto before surgery?",
    ]

    print("\n=== resolve_query() samples ===")
    for query in test_queries:
        rewritten, substitutions = resolve_query(query)
        print(f"IN:  {query}")
        print(f"OUT: {rewritten}")
        print(f"SUB: {substitutions or '(none)'}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    mapping = build_brand_to_generic()
    save_brand_mapping(mapping)

    print(f"Wrote brand mapping to {DEFAULT_MAPPING_PATH}")
    _print_mapping_summary(mapping)
    _run_sample_tests()
