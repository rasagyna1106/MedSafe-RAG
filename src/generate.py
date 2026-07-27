"""
Caregiver-facing RAG generation for MedSafe.

Wires brand resolution -> retrieval -> GPT-4o-mini with citations and abstention.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from embed import VectorStore, _strip_chunk_prefix, load_vector_store, retrieve
from ingest import TARGET_DRUGS
from mapping import BRAND_TO_GENERIC, CAREGIVER_BRAND_ALIASES, resolve_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5

# Starting threshold from Step 4 retrieval scores (good hits ~0.55–0.75). Tune in Step 6 eval.
MIN_TOP_SCORE = 0.45
# If a different drug sits within this margin of the top score, retrieval is ambiguous.
CROSS_DRUG_SCORE_MARGIN = 0.05

HIGH_RISK_PATTERNS = [
    r"\b(increase|increased|increases|increasing)\b.{0,40}\brisk\b.{0,20}\bbleed",
    r"\brisk of bleeding\b.{0,20}\b(increase|increased|may be increased)\b",
    r"\bbleed(ing)?\b.{0,40}\b(increase|increased|risk)\b",
    r"\bcontraindicated\b",
    r"\bdo not (use|take|combine|administer) (together|concurrently|concomitantly)\b",
    r"\bavoid concomitant use\b",
    r"\bfatal\b",
    r"\bserious(ly)? (increase|potentiate)\b",
    r"\baffect(s)? blood clotting\b",
    r"\bsynergistic effect on (hemostasis|bleeding)\b",
]
MODERATE_RISK_PATTERNS = [
    r"\bmonitor(ing)? (closely|regularly|patients?)\b",
    r"\bmay (reduce|increase|affect|alter)\b",
    r"\bcaution (is )?(advised|recommended)\b",
    r"\buse with caution\b",
    r"\bdose adjustment\b",
    r"\bclosely monitor\b",
]


def classify_risk_from_chunks(
    chunks: list[dict[str, Any]],
    drug_a: str,
    drug_b: str,
) -> str:
    """
    Deterministic risk classification computed directly from retrieved FDA
    label text, BEFORE any LLM call. This is the single source of truth for
    risk level — both answer_query() and answer_query_for_pair() must call
    this and pass the result into their prompts and return dicts, rather
    than letting the UI infer risk from generated prose.

    Only considers chunks belonging to drug_a or drug_b, restricted to
    drug_interactions / warnings / contraindications sections (adverse_reactions
    and dosage_and_administration are excluded — too noisy for risk signal).

    Returns one of: "HIGH", "MODERATE", "LOW", "UNKNOWN"
    ("UNKNOWN" = no relevant chunks found for this pair at all)
    """
    relevant_text = " ".join(
        c["text"].lower()
        for c in chunks
        if c.get("drug", "").lower() in {drug_a.lower(), drug_b.lower()}
        and c.get("section")
        in {"drug_interactions", "warnings", "contraindications"}
    )
    if not relevant_text:
        return "UNKNOWN"

    if any(re.search(p, relevant_text, flags=re.IGNORECASE) for p in HIGH_RISK_PATTERNS):
        return "HIGH"
    if any(re.search(p, relevant_text, flags=re.IGNORECASE) for p in MODERATE_RISK_PATTERNS):
        return "MODERATE"
    return "LOW"


ABSTENTION_MESSAGE = (
    "I don't have reliable FDA label data on this. Please consult a pharmacist or doctor."
)

SYSTEM_PROMPT = """You are a knowledgeable, warm, and caring assistant helping family
caregivers understand medication safety. Write as if you are a trusted friend who
happens to know a lot about medications - not as a doctor giving a formal consultation,
and not as a system generating a report.

Use conversational language. Use short paragraphs. Avoid bullet points and numbered lists
unless the information is genuinely a list (like multiple separate drug names). Never
use bold headers inside the answer. Write the way a caring, knowledgeable person would
actually speak.

Never start your response with a filler phrase like "That's a great question",
"Great question", "I'm glad you asked", "Happy to help", "That is a really important
question", or similar openers. Start directly with the relevant medical information.

Example of the tone to aim for:
"Taking warfarin and ibuprofen together can be risky, and here is why you should be careful..."

Not like this:
"Your mom should be cautious. The combination of ibuprofen and anticoagulants like
warfarin can increase the risk of serious bleeding."

The second version is correct but cold. The first version is correct AND warm.

Your job is to answer questions using ONLY the FDA label excerpts provided below.
Do not use outside knowledge, clinical assumptions, or information that is not
explicitly supported by those excerpts. Use plain English; if a technical term appears
in a label excerpt, briefly explain it in everyday language.

Citations:
- Support every factual claim with an inline citation in this exact format:
  (Source: drug_name, section_name)
  Example: (Source: acetaminophen, warnings)
- For EVERY drug chunk in the context, you must include at least one inline
  (Source: drug, section) citation in your answer that references that drug. If
  warfarin chunks are in the context and the user asked about warfarin, the answer
  MUST contain (Source: warfarin, drug_interactions) or (Source: warfarin,
  contraindications) or whichever warfarin section is most relevant. Not citing a
  retrieved drug's chunks is not allowed.

Combination products:
- If an excerpt has is_combo_product=true, clearly tell the reader that the text
  comes from a combination product (use full_product_name), not plain drug_name.
  Example: "Note: this dosing is from IBUPROFEN FAMOTIDINE, a combination product,
  not plain ibuprofen."
- If any chunk is marked as a COMBINATION PRODUCT, you MUST include a sentence in
  your answer saying: "Note: this information comes from [full_product_name], which
  is a combination product, not plain [drug name]. The interaction or safety profile
  may differ from the plain drug."

Multi-drug questions:
- You MUST cite at least one source from EACH drug mentioned in the user's query if
  chunks from that drug appear in the retrieved context. Do not ignore retrieved
  chunks from any drug the user asked about.
- Only discuss interactions between the drugs the user asked about. Do not invent
  or pull in unrelated drugs from the excerpts.
- Do not introduce unrelated third drugs as analogies (for example, do not bring up
  warfarin when the user asked about naproxen and sertraline). Stick to the two
  drugs in the question.
- If the labels describe an increased bleeding risk, prefer the plain phrase
  "increased risk of bleeding". Do not escalate to "serious bleeding", "fatal",
  or "synergistic" unless those exact words appear in the retrieved excerpts for
  the queried drugs.

Safety:
- Always end your response with this exact sentence on its own line:
  This is not medical advice - please consult a doctor or pharmacist before making medication decisions.

If the excerpts do not contain enough information to answer safely, say what is
missing rather than guessing."""

# Medication Checker only — short, neutral pair answers. Does not affect Ask-a-Question.
PAIR_SYSTEM_PROMPT = """You answer medication interaction questions for a schedule checker.
Write in friendly, plain English. Use "you" — never "mom", "your mom", or "family member".

Length and structure (STRICT):
- Exactly 2 sentences in the body, then the disclaimer line.
- Sentence 1: what the risk is and why, in everyday words.
- Sentence 2: what to watch for OR what to do next.
- Do NOT write a third body sentence.

Banned words/phrases (never use these):
hemostasis, synergistic, exacerbated, contraindicated, coadministration,
potentiate, anticoagulant therapy jargon beyond plain "blood thinner" /
"blood clotting".

Also banned:
- Filler openers like "That's a great question"
- "For reference" / "information was drawn from" lines
- Long clinical explanations
- Repeating the same warning in different wording

Tone examples to match exactly:

HIGH:
"Warfarin and ibuprofen are a risky combination — they both affect blood clotting
and can cause serious bleeding when taken together (Source: ibuprofen, drug_interactions).
Watch for unusual bruising, blood in stools, or vomiting blood."

MODERATE:
"Ibuprofen can reduce how well lisinopril lowers blood pressure, so this combination
needs monitoring (Source: lisinopril, drug_interactions). Ask a doctor if blood
pressure readings change while taking both."

LOW:
"No significant interaction found between these two medications in the FDA label data
(Source: metformin, drug_interactions). Still worth mentioning to a pharmacist at your
next visit."

Other rules:
- Cite inline as (Source: drug_name, section_name). Include at least one citation
  in sentence 1 or 2 covering each drug named in the question when possible.
- Use ONLY the FDA label excerpts provided. Do not invent interactions.
- Only warn about combining the two drugs if the excerpts describe an interaction
  between those two drugs. Solo drug warnings do not count as proof they interact.
- If no interaction is described, use the LOW example style.
- Output ONLY the 2 body sentences plus the disclaimer. Nothing else.
- End with this exact sentence on its own line:
  This is not medical advice - please consult a doctor or pharmacist before making medication decisions.
"""


def _load_env() -> None:
    load_dotenv(ENV_PATH)


def _require_openai_api_key() -> str:
    _load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY is not set.\n"
            f"Add it to {ENV_PATH} (see .env.example) before running generate.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _retrieve_for_drug(
    drug: str,
    store: VectorStore,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve interaction chunks from this drug's own label (not other drugs mentioning it)."""
    hits = retrieve(f"{drug} drug interactions bleeding NSAID anticoagulant", k=20, store=store)
    own_label_hits = [hit for hit in hits if hit["drug"] == drug]
    if not own_label_hits:
        # Broader query, still restricted to this drug's own label only.
        hits = retrieve(drug, k=20, store=store)
        own_label_hits = [hit for hit in hits if hit["drug"] == drug]
    if not own_label_hits:
        return []

    preferred_sections = {"drug_interactions", "warnings", "contraindications", "adverse_reactions"}
    preferred = [hit for hit in own_label_hits if hit["section"] in preferred_sections]
    source = preferred or own_label_hits
    return sorted(source, key=lambda row: row["score"], reverse=True)[:k]


def _retrieve_section_for_drug(
    drug: str,
    section: str,
    store: VectorStore,
    k: int = 3,
) -> list[dict[str, Any]]:
    hits = retrieve(f"{drug} {section.replace('_', ' ')}", k=20, store=store)
    own_label_hits = [hit for hit in hits if hit["drug"] == drug and hit["section"] == section]
    if not own_label_hits:
        own_label_hits = [hit for hit in hits if hit["drug"] == drug]
    return sorted(own_label_hits, key=lambda row: row["score"], reverse=True)[:k]


def _drugs_mentioned_in_query(query: str) -> list[str]:
    lowered = query.lower()
    return [drug for drug in TARGET_DRUGS if drug in lowered]


def _combined_brand_mapping() -> dict[str, str]:
    return {**CAREGIVER_BRAND_ALIASES, **BRAND_TO_GENERIC}


def _extract_query_drugs(original_query: str, rewritten_query: str) -> list[str]:
    """
    Extract generic drug names referenced in the query via brand aliases,
    BRAND_TO_GENERIC entries, or TARGET_DRUGS mentions.
    """
    combined_query = f"{original_query} {rewritten_query}".lower()
    brand_mapping = _combined_brand_mapping()
    found: list[str] = []

    for brand_keyword in sorted(brand_mapping, key=len, reverse=True):
        if re.search(rf"\b{re.escape(brand_keyword)}\b", combined_query, flags=re.I):
            generic = brand_mapping[brand_keyword]
            if generic not in found:
                found.append(generic)

    for drug in TARGET_DRUGS:
        if drug in combined_query and drug not in found:
            found.append(drug)

    for generic in set(brand_mapping.values()):
        if generic in found:
            continue
        if re.search(rf"\b{re.escape(generic)}\b", combined_query, flags=re.I):
            found.append(generic)

    return found


def _should_abstain_for_missing_corpus_drug(
    original_query: str,
    rewritten_query: str,
    chunks: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    Abstain when the query names a drug outside the corpus and top retrieval
    does not include that drug at all.
    """
    query_drugs = _extract_query_drugs(original_query, rewritten_query)
    missing_from_corpus = [drug for drug in query_drugs if drug not in TARGET_DRUGS]
    if not missing_from_corpus:
        return False, ""

    top_chunks = chunks[:DEFAULT_TOP_K]
    if any(chunk["drug"] in missing_from_corpus for chunk in top_chunks):
        return False, ""

    return True, "queried drug not in corpus"


def _chunk_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    return (chunk["drug"], chunk["section"], chunk["text"][:120])


def _merge_retrieval_results(
    *result_lists: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for results in result_lists:
        for chunk in results:
            key = _chunk_key(chunk)
            existing = best_by_key.get(key)
            if existing is None or chunk["score"] > existing["score"]:
                best_by_key[key] = chunk

    merged = sorted(best_by_key.values(), key=lambda row: row["score"], reverse=True)
    output: list[dict[str, Any]] = []
    for rank, chunk in enumerate(merged[:limit], start=1):
        row = dict(chunk)
        row["rank"] = rank
        output.append(row)
    return output


def retrieve_for_question(
    rewritten_query: str,
    k: int = DEFAULT_TOP_K,
    store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve label chunks for a question.

    Single-drug questions use one retrieve() call. Multi-drug questions merge a
    general query with per-drug interaction searches so both labels contribute
    (e.g. warfarin NSAID warnings + ibuprofen bleeding warnings).
    """
    store = store or load_vector_store()
    primary = retrieve(rewritten_query, k=k, store=store)

    mentioned_drugs = _drugs_mentioned_in_query(rewritten_query)
    if len(mentioned_drugs) == 1 and re.search(r"dos(e|ing|age)", rewritten_query, re.I):
        dosage_hits = _retrieve_section_for_drug(
            mentioned_drugs[0],
            "dosage_and_administration",
            store=store,
            k=3,
        )
        return list(_merge_retrieval_results(primary, dosage_hits, limit=k))

    if len(mentioned_drugs) < 2:
        return primary

    per_drug_hits = {
        drug: _retrieve_for_drug(drug, store=store, k=3) for drug in mentioned_drugs
    }
    interaction_query = (
        f"{' and '.join(mentioned_drugs)} drug interactions bleeding risk NSAID anticoagulant"
    )
    interaction_hits = retrieve(interaction_query, k=k, store=store)

    pool = list(
        _merge_retrieval_results(primary, interaction_hits, *per_drug_hits.values(), limit=k * 2)
    )

    guaranteed: list[dict[str, Any]] = []
    for drug in mentioned_drugs:
        if per_drug_hits[drug]:
            guaranteed.append(per_drug_hits[drug][0])

    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for chunk in guaranteed:
        key = _chunk_key(chunk)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(chunk)

    for chunk in sorted(pool, key=lambda row: row["score"], reverse=True):
        if len(selected) >= k:
            break
        key = _chunk_key(chunk)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(chunk)

    output: list[dict[str, Any]] = []
    for rank, chunk in enumerate(sorted(selected[:k], key=lambda row: row["score"], reverse=True), start=1):
        row = dict(chunk)
        row["rank"] = rank
        output.append(row)
    return output


def _should_abstain(
    chunks: list[dict[str, Any]],
    rewritten_query: str,
) -> tuple[bool, str]:
    if not chunks:
        return True, "no retrieval results"

    top_score = chunks[0]["score"]
    if top_score < MIN_TOP_SCORE:
        return True, f"top score {top_score:.4f} below MIN_TOP_SCORE ({MIN_TOP_SCORE})"

    # Multi-drug questions should surface several drugs — that is expected, not ambiguous.
    if len(_drugs_mentioned_in_query(rewritten_query)) >= 2:
        return False, ""

    top_drug = chunks[0]["drug"]
    for chunk in chunks[1:]:
        if chunk["drug"] == top_drug:
            continue
        if chunk["score"] >= top_score - CROSS_DRUG_SCORE_MARGIN:
            second_chunk_product = str(chunk.get("full_product_name", "")).lower()
            if top_drug.lower() in second_chunk_product:
                continue
            return True, (
                f"ambiguous retrieval: top drug={top_drug} ({top_score:.4f}) vs "
                f"{chunk['drug']} ({chunk['score']:.4f})"
            )

    return False, ""


def _combo_ingredients_text(chunk: dict[str, Any]) -> str:
    generic_names = chunk.get("generic_name") or []
    if isinstance(generic_names, list) and len(generic_names) > 1:
        return " + ".join(str(name) for name in generic_names)
    return str(chunk.get("full_product_name", chunk["drug"]))


def _count_chunks_per_drug(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        drug = str(chunk["drug"])
        counts[drug] = counts.get(drug, 0) + 1
    return counts


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines = [
            f"[Excerpt {index}]",
            f"[Drug: {chunk['drug']}][Section: {chunk['section']}]",
            f"Source: {chunk['full_product_name']}",
        ]
        if chunk.get("is_combo_product"):
            lines.append(
                "*** WARNING: This is a COMBINATION PRODUCT "
                f"({_combo_ingredients_text(chunk)}), NOT plain {chunk['drug']}. "
                "If citing this chunk, you MUST note in your answer that this information "
                "comes from a combination product label. ***"
            )
        lines.append(_strip_chunk_prefix(chunk["text"]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _build_answer_checklist(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []

    drugs_seen: list[str] = []
    for chunk in chunks:
        drug = str(chunk["drug"])
        if drug not in drugs_seen:
            drugs_seen.append(drug)

    for drug in drugs_seen:
        lines.append(
            f"[ ] You have included at least one inline (Source: {drug}, ...) "
            f"citation in your answer because {drug} chunks appear in the "
            f"context above."
        )

    combo_seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        if not chunk.get("is_combo_product"):
            continue
        key = (str(chunk["full_product_name"]), str(chunk["drug"]))
        if key in combo_seen:
            continue
        combo_seen.add(key)
        lines.append(
            "[ ] For any chunk marked *** COMBINATION PRODUCT ***, your answer "
            "includes this exact sentence structure: "
            f'"Note: this information comes from {chunk["full_product_name"]}, '
            f'which is a combination product, not plain {chunk["drug"]}."'
        )

    lines.append(
        '[ ] Your answer ends with: "This is not medical advice - please '
        'consult a doctor or pharmacist before making medication decisions."'
    )
    return "\n".join(lines)


def _build_user_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    risk_level: str | None = None,
) -> str:
    risk_block = ""
    if risk_level:
        risk_block = (
            f"PRE-DETERMINED RISK LEVEL: {risk_level}\n"
            f"Your answer's tone, urgency, and content must match this risk level exactly.\n"
            f"Do not describe this as more dangerous or less dangerous than {risk_level}.\n\n"
        )
    return (
        "---\n"
        "RETRIEVED CONTEXT:\n"
        f"{_format_chunks_for_prompt(chunks)}\n\n"
        f"{risk_block}"
        f"USER QUESTION: {query}\n\n"
        "BEFORE YOU WRITE YOUR ANSWER, confirm you will satisfy ALL of the "
        "following. Your answer is not complete until every item is met:\n\n"
        f"{_build_answer_checklist(chunks)}\n\n"
        "Now write your answer.\n"
        "---"
    )


def _build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    citations: list[dict[str, Any]] = []
    for chunk in chunks:
        key = (chunk["drug"], chunk["section"], chunk["full_product_name"])
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "drug": chunk["drug"],
                "section": chunk["section"],
                "full_product_name": chunk["full_product_name"],
                "is_combo_product": chunk["is_combo_product"],
            }
        )
    return citations


def append_combo_disclosures(answer: str, retrieved_chunks: list) -> str:
    """
    For each unique combo product in retrieved_chunks where
    is_combo_product=True, check if a disclosure sentence already
    exists in the answer. If not, insert it before the final
    disclaimer line.

    Disclosure sentence format:
    "Note: some information above comes from [full_product_name],
    which is a combination product containing [drug] alongside
    other ingredients. The interaction or safety profile may differ
    from plain [drug]."
    """
    seen: set[str] = set()
    disclosures: list[str] = []
    for chunk in retrieved_chunks:
        if chunk.get("is_combo_product") and chunk["full_product_name"] not in seen:
            seen.add(chunk["full_product_name"])
            drug = chunk["drug"]
            product = chunk["full_product_name"]
            disclosure = (
                f"Note: some information above comes from {product}, "
                f"which is a combination product containing {drug} "
                f"alongside other ingredients. The interaction or "
                f"safety profile may differ from plain {drug}."
            )
            disclosure_start = f"Note: some information above comes from {product},"
            if disclosure_start not in answer:
                disclosures.append(disclosure)

    if not disclosures:
        return answer

    disclaimer = "This is not medical advice"
    if disclaimer in answer:
        idx = answer.index(disclaimer)
        insert = "\n\n" + "\n".join(disclosures) + "\n\n"
        return answer[:idx] + insert + answer[idx:]
    return answer + "\n\n" + "\n".join(disclosures)


def validate_and_inject_citations(
    answer: str,
    retrieved_chunks: list,
) -> str:
    """
    For each unique drug in retrieved_chunks, check whether the
    answer contains at least one inline citation mentioning that
    drug. If not, inject a fallback citation line before the
    medical disclaimer.

    Looks for patterns like:
    (Source: warfarin, ...) or (Source: WARFARIN, ...)
    Case-insensitive match on drug name.

    Fallback line format:
    "For reference, information about [drug] was drawn from its
    FDA label ([most relevant section] section)."
    """
    drug_sections: dict[str, str] = {}
    for chunk in retrieved_chunks:
        drug = chunk["drug"]
        if drug not in drug_sections:
            drug_sections[drug] = chunk["section"]

    injections: list[str] = []
    for drug, section in drug_sections.items():
        pattern = re.compile(
            rf"\(Source:\s*{re.escape(drug)}",
            re.IGNORECASE,
        )
        product_names = [
            c["full_product_name"] for c in retrieved_chunks if c["drug"] == drug
        ]
        cited = pattern.search(answer)
        if not cited:
            for product in product_names:
                if f"Source: {product}" in answer or f"Source: {product.lower()}" in answer.lower():
                    cited = True
                    break

        if not cited:
            injections.append(
                f"For reference, information about {drug} was drawn "
                f"from its FDA label ({section} section)."
            )

    if not injections:
        return answer

    disclaimer = "This is not medical advice"
    insert = "\n\n" + "\n".join(injections) + "\n\n"
    if disclaimer in answer:
        idx = answer.index(disclaimer)
        return answer[:idx] + insert + answer[idx:]
    return answer + insert


def _call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL) -> str:
    api_key = _require_openai_api_key()
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned an empty completion.")
    return content.strip()


def answer_query(
    query: str,
    k: int = DEFAULT_TOP_K,
    store: VectorStore | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Resolve brands, retrieve FDA label chunks, and generate a cited caregiver answer.

    Returns:
        {
            answer, citations, substitutions_made, abstained,
            rewritten_query, abstention_reason
        }
    """
    rewritten_query, substitutions = resolve_query(query)
    store = store or load_vector_store()
    chunks = retrieve_for_question(rewritten_query, k=k, store=store)
    # Ask-tab: keep only chunks for drugs named in the query (stops cross-contamination).
    query_drugs = _extract_query_drugs(query, rewritten_query)
    risk_level: str | None = None
    if len(query_drugs) >= 2:
        # Ensure each named drug contributes at least its own-label chunks.
        for drug in query_drugs[:2]:
            if not any(drug.lower() in str(c.get("drug", "")).lower() for c in chunks):
                chunks.extend(_retrieve_for_drug(drug, store=store, k=3))
        chunks = filter_chunks_for_pair(chunks, query_drugs[0], query_drugs[1])
        risk_level = classify_risk_from_chunks(chunks, query_drugs[0], query_drugs[1])
    elif len(query_drugs) == 1:
        chunks = filter_chunks_for_drugs(chunks, query_drugs)

    abstained, abstention_reason = _should_abstain(chunks, rewritten_query)
    if not abstained:
        abstained, abstention_reason = _should_abstain_for_missing_corpus_drug(
            query, rewritten_query, chunks
        )

    if abstained:
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "substitutions": substitutions,
            "substitutions_made": substitutions,
            "abstained": True,
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
            "abstention_reason": abstention_reason,
            "risk_level": None,
        }

    user_prompt = _build_user_prompt(query, chunks, risk_level)
    answer = _call_llm(SYSTEM_PROMPT, user_prompt, model=model)
    answer = append_combo_disclosures(answer, chunks)
    answer = validate_and_inject_citations(answer, chunks)

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "substitutions": substitutions,
        "substitutions_made": substitutions,
        "abstained": False,
        "answer": answer,
        "citations": _build_citations(chunks),
        "chunks_per_drug": _count_chunks_per_drug(chunks),
        "abstention_reason": "",
        "risk_level": risk_level,
    }


def filter_chunks_for_drugs(
    chunks: list[dict[str, Any]],
    drugs: list[str],
) -> list[dict[str, Any]]:
    """Keep only chunks whose drug field matches one of the listed drugs."""
    if not drugs:
        return chunks
    lowers = [d.lower() for d in drugs if d]
    filtered = [
        chunk
        for chunk in chunks
        if any(d in str(chunk.get("drug", "")).lower() for d in lowers)
    ]
    return filtered if filtered else chunks[:3]


def filter_chunks_for_pair(
    chunks: list[dict[str, Any]],
    drug_a: str,
    drug_b: str,
) -> list[dict[str, Any]]:
    """
    Keeps only chunks whose drug field matches one of the two queried drugs.
    Discards chunks from third drugs that leaked in through semantic similarity.
    Used by Medication Checker pairs and Ask-a-Question two-drug queries.
    """
    return filter_chunks_for_drugs(chunks, [drug_a, drug_b])


def _build_pair_user_prompt(
    query: str,
    chunks: list[dict[str, Any]],
    risk_level: str | None = None,
) -> str:
    drugs_seen: list[str] = []
    for chunk in chunks:
        drug = str(chunk["drug"])
        if drug not in drugs_seen:
            drugs_seen.append(drug)
    cite_lines = "\n".join(
        f"[ ] Include at least one (Source: {drug}, ...) citation." for drug in drugs_seen
    )
    risk_block = ""
    if risk_level:
        risk_block = (
            f"PRE-DETERMINED RISK LEVEL: {risk_level}\n"
            f"Your answer's tone, urgency, and content must match this risk level exactly.\n"
            f"Do not describe this as more dangerous or less dangerous than {risk_level}.\n\n"
        )
    return (
        "---\n"
        "RETRIEVED CONTEXT:\n"
        f"{_format_chunks_for_prompt(chunks)}\n\n"
        f"{risk_block}"
        f"USER QUESTION: {query}\n\n"
        "Write EXACTLY 2 plain-English body sentences, then the disclaimer.\n"
        "Sentence 1 = risk + why. Sentence 2 = what to watch for or what to do.\n"
        "No jargon (no hemostasis, synergistic, exacerbated, contraindicated).\n"
        "Do NOT add any 'For reference' lines or extra paragraphs.\n"
        "Checklist:\n"
        f"{cite_lines}\n"
        '[ ] Ends with: "This is not medical advice - please consult a doctor or '
        'pharmacist before making medication decisions."\n'
        "[ ] No mom/family-member language. No filler openers.\n"
        "---"
    )


def answer_query_for_pair(
    drug_a: str,
    drug_b: str,
    k: int = DEFAULT_TOP_K,
    store: VectorStore | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """
    Medication Checker pair path: filtered context + short neutral answers.
    Does not change the Ask-a-Question / answer_query behavior.
    """
    query = f"Is it safe to take {drug_a} and {drug_b} together?"
    rewritten_query, substitutions = resolve_query(query)
    chunks = retrieve_for_question(rewritten_query, k=k, store=store)
    chunks = filter_chunks_for_pair(chunks, drug_a, drug_b)
    risk_level = classify_risk_from_chunks(chunks, drug_a, drug_b)

    abstained, abstention_reason = _should_abstain(chunks, rewritten_query)
    if not abstained:
        abstained, abstention_reason = _should_abstain_for_missing_corpus_drug(
            query, rewritten_query, chunks
        )

    if abstained:
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "substitutions": substitutions,
            "substitutions_made": substitutions,
            "abstained": True,
            "answer": ABSTENTION_MESSAGE,
            "citations": [],
            "abstention_reason": abstention_reason,
            "risk_level": None,
        }

    user_prompt = _build_pair_user_prompt(query, chunks, risk_level)
    answer = _call_llm(PAIR_SYSTEM_PROMPT, user_prompt, model=model)
    # Pair answers must stay exactly 2 body sentences. Do not inject the
    # Ask-tab "For reference..." fallback lines; the FDA Sources table covers metadata.

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "substitutions": substitutions,
        "substitutions_made": substitutions,
        "abstained": False,
        "answer": answer,
        "citations": _build_citations(chunks),
        "chunks_per_drug": _count_chunks_per_drug(chunks),
        "abstention_reason": "",
        "risk_level": risk_level,
    }


def print_answer_result(query: str, result: dict[str, Any]) -> None:
    print(f'Original query: "{query}"')
    if result["rewritten_query"] != query:
        print(f'Rewritten query: "{result["rewritten_query"]}"')
    else:
        print('Rewritten query: unchanged')
    substitutions = result.get("substitutions") or result.get("substitutions_made") or []
    print(f"Substitutions: {substitutions if substitutions else 'none'}")
    print(f"Abstained: {'Yes' if result['abstained'] else 'No'}")
    if result["abstained"] and result.get("abstention_reason"):
        print(f"Abstention reason: {result['abstention_reason']}")
    if not result["abstained"] and result.get("chunks_per_drug"):
        summary = ", ".join(
            f"{drug}={count}" for drug, count in sorted(result["chunks_per_drug"].items())
        )
        print(f"Context chunks per drug: {summary}")
    print(f"Risk level: {result.get('risk_level')}")
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- CITATIONS ---")
    if not result["citations"]:
        print("(none)")
    else:
        for index, citation in enumerate(result["citations"], start=1):
            combo = "Yes" if citation.get("is_combo_product") else "No"
            print(
                f"{index}. drug={citation['drug']} section={citation['section']} "
                f"product={citation['full_product_name']} combo={combo}"
            )
    print()


def main() -> None:
    _configure_utf8_stdout()
    _require_openai_api_key()

    parser = argparse.ArgumentParser(description="Generate caregiver answers with MedSafe RAG.")
    parser.add_argument("--query", required=True, help="Caregiver question to answer.")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Retrieval top-k.")
    args = parser.parse_args()

    store = load_vector_store()
    result = answer_query(args.query, k=args.k, store=store)
    print_answer_result(args.query, result)


if __name__ == "__main__":
    main()
