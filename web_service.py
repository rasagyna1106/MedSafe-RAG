"""API layer for MedSafe RAG web app — wraps the generation pipeline."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
load_dotenv(PROJECT_ROOT / ".env")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embed import _strip_chunk_prefix, load_vector_store
from generate import (
    _extract_query_drugs,
    answer_query,
    answer_query_for_pair,
    retrieve_for_question,
)
from ingest import TARGET_DRUGS
from mapping import resolve_query

EXAMPLE_QUESTIONS = [
    "Is Tylenol safe to take every day?",
    "What are the interactions for warfarin?",
    "Can my mom take warfarin and ibuprofen together?",
    "What are the side effects of sertraline?",
    "What is the normal dose of metformin?",
]

DISCLAIMER_PHRASE = "This is not medical advice"
SOURCE_CITATION_RE = re.compile(r"\s*\(Source:\s*[^)]+\)", re.I)
NOTE_PARAGRAPH_RE = re.compile(r"^Note:\s", re.I)
REFERENCE_PARAGRAPH_RE = re.compile(r"^For reference,", re.I)

HIGH_RISK_TERMS = (
    "fatal",
    "death",
    "serious bleeding",
    "liver damage",
    "liver failure",
    "liver injury",
    "kidney damage",
    "kidney failure",
    "contraindicated",
    "do not use",
    "don't use",
    "do not take",
    "do not give",
    "suicidal",
    "life-threatening",
    "life threatening",
    "risky",
    "can be risky",
    "serious bleeding",
    "serious risk",
    "serious side",
    "bleeding",
    "interaction",
    "side effect",
    "side effects",
    "adverse reaction",
    "harmful",
    "unsafe",
    "damage",
    "liver",
    "kidney",
)

MODERATE_RISK_TERMS = (
    "caution",
    "monitor",
    "check with doctor",
    "check with your doctor",
    "avoid if",
    "consult",
    "warning",
    "limit",
    "maximum dose",
    "dosage",
    "dose",
    "lactic acidosis",
)

GREEN_BLOCKER_TERMS = (
    "caution",
    "avoid",
    "risk",
    "danger",
    "serious",
    "warning",
    "bleeding",
    "damage",
    "liver",
    "kidney",
    "death",
    "fatal",
    "do not",
    "contraind",
)

FILLER_STARTS = (
    "that's a great",
    "that's a good",
    "that's an excellent",
    "that's a really",
    "that's an important",
    "that is a great",
    "that is a good",
    "that is an excellent",
    "that is a really",
    "that is an important",
    "great question",
    "good question",
)

FILLER_PHRASES = (
    "i'm glad you asked",
    "i am glad you asked",
    "happy to help",
    "i'd be happy to help",
    "i would be happy to help",
)

ASK_NO_INTERACTION_PHRASES = (
    "no specific",
    "no direct",
    "no significant",
    "no known interaction",
    "generally be taken together",
    "generally safe",
    "don't have specific information",
    "do not have specific information",
    "i don't have information about",
    "i don't have information",
    "i do not have information",
    "no interaction found",
    "no interaction",
    "safely together",
    "no mention of a direct interaction",
    "no specific warning",
    "no specific interaction",
    "doesn't appear to interact",
    "does not appear to interact",
    "there isn't specific information",
    "there is not specific information",
)

# Order matters: more specific HIGH phrases are checked before generic MODERATE ones.
ASK_HIGH_KEYWORDS = (
    "serious bleeding",
    "fatal",
    "liver damage",
    "do not use together",
    "dangerous combination",
    "synergistic effect on bleeding",
    "significantly increases the risk of serious",
    "avoid combining",
    "increased risk of bleeding",
    "heighten the risk of serious",
    "risk of serious bleeding",
)

ASK_MODERATE_KEYWORDS = (
    "monitor closely",
    "use caution",
    "kidney function",
    "renal function",
    "blood pressure may be reduced",
    "reduced effectiveness",
    "keep an eye on",
    "risk of bleeding",
)

MEDICAL_TERM_RE = re.compile(
    r"\b(?:mg|mcg|ml|g|tablet|capsule|dose|dosage|warning|risk|bleeding|liver|kidney|"
    r"interaction|contraind|side effect|infection|bacteria|diabetes|hypertension|pain|"
    r"warfarin|ibuprofen|acetaminophen|metformin|sertraline|amoxicillin|azithromycin|"
    r"antibiotic|antidepressant|nsaid|anticoagulant)\b",
    re.I,
)
DOSAGE_NUM_RE = re.compile(r"\b\d+\s*(?:mg|mcg|ml|g|%|times|tablets?|capsules?)\b", re.I)

CRITICAL_RISK_PATTERNS: list[tuple[str, str, str]] = [
    (r"serious bleeding", "Serious bleeding", "critical"),
    (r"stomach bleeding", "Stomach bleeding", "high"),
    (r"internal bleeding", "Internal bleeding", "critical"),
    (r"\brisk of bleeding\b", "Bleeding risk", "moderate"),
    (r"\bbleeding\b", "Bleeding risk", "high"),
    (r"allergic reaction", "Allergic reaction", "critical"),
    (r"anaphylaxis", "Anaphylaxis", "critical"),
    (r"liver (?:damage|failure|injury)", "Liver damage", "high"),
    (r"kidney (?:damage|failure|injury)", "Kidney damage", "high"),
    (r"\bcontraindicated\b", "Contraindicated", "critical"),
    (r"do not take", "Do not combine", "critical"),
    (r"fatal", "Fatal risk", "critical"),
    (r"heart attack", "Heart attack risk", "high"),
    (r"stroke", "Stroke risk", "high"),
    (r"respiratory depression", "Breathing suppression", "critical"),
    (r"overdose", "Overdose risk", "high"),
    (r"seizure", "Seizure risk", "high"),
]
_vector_store = None


def get_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = load_vector_store()
    return _vector_store


def _sanitize_dashes(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\s*—\s*", ", ", text)
    text = re.sub(r"\s*–\s*", ", ", text)
    text = re.sub(r"\s*--\s*", ", ", text)
    return text


def _sanitize_structured(structured: dict) -> dict:
    for key in ("verdict",):
        if key in structured and isinstance(structured[key], dict) and "text" in structured[key]:
            structured[key]["text"] = _sanitize_dashes(structured[key]["text"])
    for key in ("simple_guide", "clinical_details", "technical_notes"):
        if key in structured and isinstance(structured[key], list):
            structured[key] = [_sanitize_dashes(item) for item in structured[key]]
    if "tldr_bullets" in structured:
        for bullet in structured["tldr_bullets"]:
            if "text" in bullet:
                bullet["text"] = _sanitize_dashes(bullet["text"])
    if structured.get("disclaimer"):
        structured["disclaimer"] = _sanitize_dashes(structured["disclaimer"])
    return structured


def _answer_body_for_risk(answer_text: str) -> str:
    """Strip disclaimer so 'consult a doctor' in the footer does not inflate risk."""
    body = answer_text or ""
    if DISCLAIMER_PHRASE.lower() in body.lower():
        idx = body.lower().rfind(DISCLAIMER_PHRASE.lower())
        if idx >= 0:
            body = body[:idx].strip()
    return body


def _focus_answer_on_query_drugs(answer_text: str, query_drugs: list[str] | None) -> str:
    """
    Drop digression sentences that name unrelated corpus drugs (e.g. warfarin
    analogies in a naproxen + sertraline answer) so risk classification reflects
    the queried combination only.
    """
    body = _answer_body_for_risk(answer_text)
    if not query_drugs:
        return body
    allowed = {d.lower() for d in query_drugs}
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        s = sentence.strip()
        if not s:
            continue
        lower = s.lower()
        mentioned = [d for d in TARGET_DRUGS if re.search(rf"\b{re.escape(d)}\b", lower)]
        if not mentioned:
            kept.append(s)
            continue
        if all(m in allowed for m in mentioned):
            kept.append(s)
    return " ".join(kept) if kept else body


def classify_ask_risk_from_answer(
    answer_text: str,
    query_drugs: list[str] | None = None,
) -> str:
    """
    Deprecated. Superseded by generate.classify_risk_from_chunks / result['risk_level'].
    Kept for reference; Ask UI no longer calls this.
    """
    return "UNKNOWN"


def _ask_critical_risks_from_answer(
    answer_text: str,
    risk_level: str,
    query_drugs: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Critical Risks banner content — gated by the same risk_level.
    LOW => never show. HIGH/MODERATE => tags from answer (or defaults).
    """
    if risk_level in ("low", "unknown"):
        return []

    body = _focus_answer_on_query_drugs(answer_text, query_drugs)
    tags = _extract_critical_risks(body)
    if risk_level == "moderate":
        # Yellow/moderate styling — never paint moderate answers as critical-red.
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for tag in tags:
            label = tag["label"]
            if label in seen:
                continue
            seen.add(label)
            normalized.append({"label": label, "severity": "moderate"})
        if not normalized:
            normalized = [{"label": "Bleeding risk", "severity": "moderate"}]
        return normalized

    # HIGH
    if not tags:
        return [{"label": "Serious interaction risk", "severity": "critical"}]
    high_tags: list[dict[str, str]] = []
    seen_h: set[str] = set()
    for tag in tags:
        label = tag["label"]
        if label in seen_h:
            continue
        seen_h.add(label)
        sev = tag.get("severity", "high")
        if sev == "moderate":
            sev = "high"
        high_tags.append({"label": label, "severity": sev})
    return high_tags


def _strip_leading_filler(text: str) -> str:
    """Remove conversational filler openers from the start of answer text/paragraphs."""
    remaining = (text or "").strip()
    if not remaining:
        return remaining
    while True:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", remaining) if s.strip()]
        if not sentences:
            return remaining
        first = sentences[0]
        lower = SOURCE_CITATION_RE.sub("", first).strip().lower()
        lower = re.sub(r"\s{2,}", " ", lower).strip()
        is_filler = any(lower.startswith(prefix) for prefix in FILLER_STARTS) or any(
            phrase in lower for phrase in FILLER_PHRASES
        )
        if is_filler and len(first.split()) <= 14:
            # Remove the first sentence from the original string.
            pattern = re.compile(re.escape(first) + r"[.!?]?\s*", re.I)
            new_remaining = pattern.sub("", remaining, count=1).strip()
            if new_remaining == remaining:
                break
            remaining = new_remaining
            continue
        break
    return remaining.strip()


def _truncate_words(text: str, max_words: int = 200) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + "...", True


def _map_pipeline_risk(pipeline_risk: str | None) -> tuple[str, str, str]:
    """
    Map generate.py risk_level (HIGH/MODERATE/LOW/UNKNOWN/None) to UI fields:
    (display label, css level, internal class HIGH|MODERATE|LOW|UNKNOWN).
    """
    if pipeline_risk == "HIGH":
        return "High", "high", "HIGH"
    if pipeline_risk == "MODERATE":
        return "Moderate", "moderate", "MODERATE"
    if pipeline_risk == "LOW":
        return "Low", "low", "LOW"
    return "Unknown", "unknown", "UNKNOWN"


def _derive_ask_metrics(
    answer: str,
    citations: list[dict],
    substitutions: list[str],
    question: str,
    query_drugs: list[str],
    pipeline_risk: str | None = None,
) -> dict:
    active = "N/A"
    if citations:
        active = citations[0]["drug"].replace("_", " ").title()
    for item in substitutions:
        if "->" in item:
            active = item.split("->", 1)[1].strip().title()

    # Single source of truth from generate.classify_risk_from_chunks (pre-LLM).
    risk, risk_level, level = _map_pipeline_risk(pipeline_risk)

    lower = answer.lower()
    if any(k in lower for k in ("child", "pediatric", "under 12", "under 2", "infant")):
        demo = "Pediatric warnings apply"
    elif any(k in lower for k in ("pregnan", "nursing", "breastfeed")):
        demo = "Pregnancy / nursing"
    else:
        demo = "Adult (check label)"

    return {
        "active_ingredient": active,
        "risk": risk,
        "risk_level": risk_level,
        "demographic": demo,
        "is_multi_drug": len(query_drugs) >= 2,
        "risk_class": level,
        "query_drugs": list(query_drugs),
    }


def _build_ask_verdict(risk_level: str, is_multi_drug: bool) -> dict[str, str]:
    if risk_level == "high":
        return {
            "emoji": "⚠️",
            "text": "Use with extreme caution",
            "tone": "warning",
            "risk_level": "high",
        }
    if risk_level == "moderate":
        return {
            "emoji": "⚠️",
            "text": "Use with caution",
            "tone": "caution",
            "risk_level": "moderate",
        }
    if risk_level == "unknown":
        return {
            "emoji": "ℹ️",
            "text": "Risk unclear from FDA label data",
            "tone": "caution",
            "risk_level": "unknown",
        }
    if is_multi_drug:
        return {
            "emoji": "✅",
            "text": "Generally safe together",
            "tone": "safe",
            "risk_level": "low",
        }
    return {
        "emoji": "✅",
        "text": "Safe to use as directed",
        "tone": "safe",
        "risk_level": "low",
    }


def _detect_ask_combined_use(
    metrics: dict | None,
    query_drugs: list[str],
) -> dict | None:
    """Combined-use badge — driven only by the shared risk level + multi-drug query."""
    if not metrics or len(query_drugs) < 2:
        return None

    risk_level = metrics.get("risk_level", "low")
    if risk_level == "high":
        return {
            "risk": "High",
            "risk_level": "high",
            "message": "Combined use risk: High, these drugs interact",
        }
    if risk_level == "moderate":
        return {
            "risk": "Moderate",
            "risk_level": "moderate",
            "message": "Combined use risk: Moderate, monitor closely",
        }
    return None


def _structure_ask_answer(answer: str, metrics: dict | None, abstained: bool) -> dict:
    """Ask-tab structuring: concise Simple Guide, full Clinical Details, no filler."""
    paragraphs, disclaimer = _split_answer(answer)

    if abstained:
        return {
            "verdict": {
                "emoji": "⚠️",
                "text": "Unable to answer reliably",
                "tone": "abstain",
                "risk_level": "abstain",
            },
            "tldr_bullets": [
                {"icon": "ℹ️", "label": "Status", "text": paragraphs[0] if paragraphs else answer}
            ],
            "simple_guide": paragraphs,
            "clinical_details": paragraphs,
            "technical_notes": [],
            "critical_risks": [],
            "disclaimer": disclaimer,
            "card_risk_level": "abstain",
        }

    technical_notes: list[str] = []
    main_paragraphs: list[str] = []
    for para in paragraphs:
        cleaned = _strip_leading_filler(para)
        if not cleaned:
            continue
        if NOTE_PARAGRAPH_RE.match(cleaned) or REFERENCE_PARAGRAPH_RE.match(cleaned):
            technical_notes.append(cleaned)
        else:
            main_paragraphs.append(cleaned)

    if not main_paragraphs and paragraphs:
        main_paragraphs = [
            p for p in (_strip_leading_filler(x) for x in paragraphs) if p
        ]

    risk_level = (metrics or {}).get("risk_level", "low")
    is_multi = bool((metrics or {}).get("is_multi_drug"))
    verdict = _build_ask_verdict(risk_level, is_multi)
    tldr_bullets = _build_tldr_bullets(main_paragraphs)
    # Banner gated by the SAME risk_level as Quick Summary / Risk card.
    critical_risks = _ask_critical_risks_from_answer(
        answer,
        risk_level,
        (metrics or {}).get("query_drugs") or None,
    )

    clinical_details = main_paragraphs
    first = _strip_citations(main_paragraphs[0]) if main_paragraphs else ""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", first) if s.strip()]
    if len(sents) > 2:
        first = " ".join(sents[:2])
    first, truncated = _truncate_words(first, 200)
    if len(sents) > 2:
        truncated = True
    simple_guide = [first] if first else []
    if truncated:
        simple_guide.append("Read more in Clinical Details")

    return {
        "verdict": verdict,
        "tldr_bullets": tldr_bullets,
        "simple_guide": simple_guide,
        "clinical_details": clinical_details,
        "technical_notes": technical_notes,
        "critical_risks": critical_risks,
        "disclaimer": disclaimer,
        "card_risk_level": risk_level,  # same variable as metrics.risk_level
    }


def _derive_metrics(
    answer: str,
    citations: list[dict],
    substitutions: list[str],
    context_text: str = "",
    question: str = "",
) -> dict:
    active = "N/A"
    if citations:
        active = citations[0]["drug"].replace("_", " ").title()
    for item in substitutions:
        if "->" in item:
            active = item.split("->", 1)[1].strip().title()

    risk, risk_level = _classify_risk_level(answer, context_text, question)

    lower = answer.lower()
    if any(k in lower for k in ("child", "pediatric", "under 12", "under 2", "infant")):
        demo = "Pediatric warnings apply"
    elif any(k in lower for k in ("pregnan", "nursing", "breastfeed")):
        demo = "Pregnancy / nursing"
    else:
        demo = "Adult (check label)"

    return {"active_ingredient": active, "risk": risk, "risk_level": risk_level, "demographic": demo}


def _text_has_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _is_informational_query(question: str) -> bool:
    q = question.lower()
    if any(p in q for p in ("dose", "dosage", "how much", "how many")):
        return False
    if any(p in q for p in ("safe", "side effect", "interaction", "together", "take with")):
        return False
    return any(
        p in q
        for p in (
            "prescribed for",
            "used for",
            "what is",
            "what conditions",
            "conditions does",
            "indications",
            " treat",
            "treats",
            "treatment for",
        )
    )


def _is_dosage_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in ("dose", "dosage", "how much", "how many", "normal dose"))


def _is_safety_query(question: str) -> bool:
    q = question.lower()
    return any(
        p in q
        for p in ("safe", "side effect", "interaction", "together", "take with", "risky", "danger")
    )


def _classify_risk_level(answer: str, context_text: str, question: str = "") -> tuple[str, str]:
    answer_l = answer.lower()
    combined_l = f"{answer}\n{context_text}".lower()

    strong_red = (
        "fatal",
        "death",
        "serious bleeding",
        "suicidal",
        "life-threatening",
        "life threatening",
        "do not take",
        "do not give",
        "do not use",
        "risky",
        "can be risky",
    )

    if _is_informational_query(question):
        if _text_has_any(answer_l, strong_red):
            return "High", "high"
        return "Low", "low"

    if _is_dosage_query(question):
        if _extract_critical_risks(combined_l) or _text_has_any(answer_l, strong_red):
            return "High", "high"
        return "Moderate", "moderate"

    if _is_safety_query(question):
        if _extract_critical_risks(combined_l) or _text_has_any(combined_l, HIGH_RISK_TERMS):
            return "High", "high"
        if _text_has_any(answer_l, strong_red):
            return "High", "high"
        if _text_has_any(combined_l, MODERATE_RISK_TERMS) or _text_has_any(combined_l, GREEN_BLOCKER_TERMS):
            return "Moderate", "moderate"
        return "Moderate", "moderate"

    if _extract_critical_risks(combined_l):
        return "High", "high"
    if _text_has_any(combined_l, HIGH_RISK_TERMS) or _text_has_any(answer_l, strong_red):
        return "High", "high"
    if _text_has_any(combined_l, MODERATE_RISK_TERMS) or _text_has_any(combined_l, GREEN_BLOCKER_TERMS):
        return "Moderate", "moderate"
    if _text_has_any(answer_l, GREEN_BLOCKER_TERMS):
        return "Moderate", "moderate"
    return "Low", "low"


def _split_answer(answer: str) -> tuple[list[str], str | None]:
    body = answer
    disclaimer = None
    if DISCLAIMER_PHRASE.lower() in answer.lower():
        idx = answer.lower().rfind(DISCLAIMER_PHRASE.lower())
        if idx >= 0:
            body = answer[:idx].strip()
            disclaimer = answer[idx:].strip()
    raw_blocks = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    paragraphs: list[str] = []
    for block in raw_blocks:
        parts = re.split(r"\n(?=(?:Note:|For reference,))", block)
        paragraphs.extend(p.strip() for p in parts if p.strip())
    if not paragraphs:
        paragraphs = [body.strip()] if body.strip() else [answer]
    return paragraphs, disclaimer


def _strip_citations(text: str) -> str:
    cleaned = SOURCE_CITATION_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _extract_critical_risks(text: str) -> list[dict[str, str]]:
    lower = text.lower()
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern, label, severity in CRITICAL_RISK_PATTERNS:
        if re.search(pattern, lower) and label not in seen:
            seen.add(label)
            found.append({"label": label, "severity": severity})
    return found


def _build_verdict(risk_level: str) -> dict[str, str]:
    if risk_level == "high":
        return {
            "emoji": "⚠️",
            "text": "Use with extreme caution",
            "tone": "warning",
            "risk_level": "high",
        }
    if risk_level == "moderate":
        return {
            "emoji": "⚠️",
            "text": "Use with caution",
            "tone": "caution",
            "risk_level": "moderate",
        }
    return {
        "emoji": "✅",
        "text": "Safe to use as directed",
        "tone": "safe",
        "risk_level": "low",
    }


def _format_brand_substitutions(substitutions: list[str]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for item in substitutions:
        raw = item.strip()
        if "->" not in raw:
            continue
        left, right = raw.split("->", 1)
        brand = left.replace("Brand name resolved:", "").strip()
        generic = right.strip()
        if brand and generic:
            formatted.append(
                {
                    "brand": brand,
                    "generic": generic.title() if generic.islower() else generic,
                    "display": f"{brand} → {generic}",
                }
            )
    return formatted


def _detect_combined_use(citations: list[dict], metrics: dict | None) -> dict | None:
    drugs = {cit["drug"] for cit in citations}
    if len(drugs) < 2 or not metrics:
        return None
    risk = metrics.get("risk", "Moderate")
    return {
        "risk": risk,
        "risk_level": metrics.get("risk_level", "moderate"),
        "message": f"Combined use risk: {risk}, these drugs interact",
    }


def _norm_bullet_key(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_citations(text).lower())


def _is_duplicate_bullet(text: str, used: set[str]) -> bool:
    key = _norm_bullet_key(text)
    if not key:
        return True
    if key in used:
        return True
    for existing in used:
        if key in existing or existing in key:
            return True
    return False


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _is_filler_sentence(sentence: str) -> bool:
    stripped = _strip_citations(sentence).strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if any(lower.startswith(prefix) for prefix in FILLER_STARTS):
        return True
    if any(phrase in lower for phrase in FILLER_PHRASES) and len(stripped.split()) < 12:
        return True
    words = stripped.split()
    if len(words) < 20:
        has_medical = bool(MEDICAL_TERM_RE.search(stripped))
        has_dosage = bool(DOSAGE_NUM_RE.search(stripped))
        if not has_medical and not has_dosage:
            return True
    return False


def _substantive_sentences(paragraphs: list[str]) -> list[str]:
    substantive: list[str] = []
    for para in paragraphs:
        for sentence in _sentences(para):
            if not _is_filler_sentence(sentence):
                substantive.append(sentence)
    return substantive


def _pick_sentence(paragraphs: list[str], patterns: tuple[str, ...], used: set[str]) -> str | None:
    for sentence in _substantive_sentences(paragraphs):
        lower = sentence.lower()
        if any(p in lower for p in patterns) and not _is_duplicate_bullet(sentence, used):
            return sentence
    return None


def _build_tldr_bullets(main_paragraphs: list[str]) -> list[dict[str, str]]:
    bullets: list[dict[str, str]] = []
    used: set[str] = set()

    def add_bullet(icon: str, label: str, text: str | None) -> None:
        if not text or _is_duplicate_bullet(text, used):
            return
        used.add(_norm_bullet_key(text))
        bullets.append({"icon": icon, "label": label, "text": _strip_citations(text)})

    danger_patterns = (
        "risky",
        "can be risky",
        "danger",
        "harmful",
        "unsafe",
        "interaction",
        "do not",
        "avoid",
        "increase the risk",
        "serious bleeding",
        "contraind",
    )
    watch_patterns = (
        "watch for",
        "monitor for",
        "monitor any",
        "signs of",
        "symptoms like",
        "unusual bruising",
        "blood in",
        "vomiting blood",
        "look out for",
    )
    action_patterns = (
        "doctor",
        "pharmacist",
        "consult",
        "discuss",
        "call your",
        "seek medical",
        "healthcare",
        "talk to",
    )

    add_bullet("🚨", "Main danger", _pick_sentence(main_paragraphs, danger_patterns, used))
    add_bullet("👀", "Watch for", _pick_sentence(main_paragraphs, watch_patterns, used))
    add_bullet("📞", "What to do", _pick_sentence(list(reversed(main_paragraphs)), action_patterns, used))

    return bullets[:3]


def _structure_answer(answer: str, metrics: dict | None, abstained: bool) -> dict:
    paragraphs, disclaimer = _split_answer(answer)

    if abstained:
        return {
            "verdict": {
                "emoji": "⚠️",
                "text": "Unable to answer reliably",
                "tone": "abstain",
                "risk_level": "abstain",
            },
            "tldr_bullets": [{"icon": "ℹ️", "label": "Status", "text": paragraphs[0] if paragraphs else answer}],
            "simple_guide": paragraphs,
            "clinical_details": paragraphs,
            "technical_notes": [],
            "critical_risks": [],
            "disclaimer": disclaimer,
            "card_risk_level": "abstain",
        }

    technical_notes: list[str] = []
    main_paragraphs: list[str] = []
    for para in paragraphs:
        if NOTE_PARAGRAPH_RE.match(para) or REFERENCE_PARAGRAPH_RE.match(para):
            technical_notes.append(para)
        else:
            main_paragraphs.append(para)

    main_text = " ".join(main_paragraphs)
    risk_level = (metrics or {}).get("risk_level", "low")
    verdict = _build_verdict(risk_level)
    tldr_bullets = _build_tldr_bullets(main_paragraphs)
    critical_risks = _extract_critical_risks(main_text)
    simple_guide = [_strip_citations(p) for p in main_paragraphs]

    return {
        "verdict": verdict,
        "tldr_bullets": tldr_bullets,
        "simple_guide": simple_guide,
        "clinical_details": main_paragraphs,
        "technical_notes": technical_notes,
        "critical_risks": critical_risks,
        "disclaimer": disclaimer,
        "card_risk_level": verdict.get("risk_level", risk_level),
    }


def _context_from_citations(rewritten_query: str, citations: list[dict]) -> str:
    if not citations:
        return ""
    chunks = retrieve_for_question(rewritten_query, k=5, store=get_store())
    cited_keys = {(cit["drug"], cit["section"], cit["full_product_name"]) for cit in citations}
    texts: list[str] = []
    for chunk in chunks:
        key = (chunk["drug"], chunk["section"], chunk["full_product_name"])
        if key in cited_keys:
            texts.append(_strip_chunk_prefix(chunk["text"]))
    if not texts:
        cited_drugs = {cit["drug"] for cit in citations}
        for chunk in chunks:
            if chunk["drug"] in cited_drugs:
                texts.append(_strip_chunk_prefix(chunk["text"]))
    return " ".join(texts)


def _fetch_snippets(rewritten_query: str, citations: list[dict]) -> list[str]:
    chunks = retrieve_for_question(rewritten_query, k=5, store=get_store())
    snippets: list[str] = []
    for cit in citations:
        snippet = ""
        key = (cit["drug"], cit["section"], cit["full_product_name"])
        for chunk in chunks:
            ck = (chunk["drug"], chunk["section"], chunk["full_product_name"])
            if ck == key:
                snippet = _strip_chunk_prefix(chunk["text"])[:300]
                break
        if not snippet:
            for chunk in chunks:
                if chunk["drug"] == cit["drug"] and chunk["section"] == cit["section"]:
                    snippet = _strip_chunk_prefix(chunk["text"])[:300]
                    break
        if snippet and len(snippet) >= 280:
            snippet += "…"
        snippets.append(snippet or "Excerpt not available.")
    return snippets


def process_question(question: str) -> dict:
    question = (question or "").strip()
    if not question:
        return {"error": "Please enter a question."}

    start = time.perf_counter()
    try:
        result = answer_query(question, store=get_store())
    except Exception:
        return {"error": "Something went wrong. Please try again."}

    elapsed = time.perf_counter() - start
    answer = result.get("answer", "")
    abstained = bool(result.get("abstained"))
    substitutions = result.get("substitutions") or result.get("substitutions_made") or []
    citations = result.get("citations") or []
    rewritten = result.get("rewritten_query", question)
    query_drugs = _extract_query_drugs(question, rewritten)

    # Belt-and-suspenders: citations must match queried drugs when identifiable.
    if len(query_drugs) >= 2:
        citations = _filter_citations_for_pair(citations, query_drugs[0], query_drugs[1])
    elif len(query_drugs) == 1:
        drug = query_drugs[0].lower()
        filtered = [c for c in citations if drug in str(c.get("drug", "")).lower()]
        if filtered:
            citations = filtered

    paragraphs, disclaimer = _split_answer(answer)
    pipeline_risk = result.get("risk_level")
    metrics = (
        _derive_ask_metrics(
            answer,
            citations,
            substitutions,
            question,
            query_drugs,
            pipeline_risk=pipeline_risk,
        )
        if not abstained
        else None
    )
    structured = _structure_ask_answer(answer, metrics, abstained)
    structured = _sanitize_structured(structured)
    answer = _sanitize_dashes(answer)
    paragraphs = [_sanitize_dashes(p) for p in paragraphs]
    if disclaimer:
        disclaimer = _sanitize_dashes(disclaimer)
    snippets = _fetch_snippets(rewritten, citations) if citations else []

    cited_labels = []
    for cit, snippet in zip(citations, snippets, strict=False):
        cited_labels.append(
            {
                **cit,
                "section_display": cit["section"].replace("_", " "),
                "snippet": snippet,
            }
        )

    brand_substitutions = _format_brand_substitutions(substitutions)
    combined_use_risk = (
        None if abstained else _detect_ask_combined_use(metrics, query_drugs)
    )

    return {
        "original_query": result.get("original_query", question),
        "rewritten_query": rewritten,
        "substitutions": substitutions,
        "brand_substitutions": brand_substitutions,
        "combined_use_risk": combined_use_risk,
        "abstained": abstained,
        "abstention_reason": result.get("abstention_reason", ""),
        "answer": answer,
        "paragraphs": paragraphs,
        "structured": structured,
        "disclaimer": disclaimer,
        "metrics": metrics,
        "citations": cited_labels,
        "elapsed_seconds": round(elapsed, 1),
        "risk_level": pipeline_risk,
    }


PAIR_WARNING_TERMS = (
    "risk",
    "bleeding",
    "caution",
    "monitor",
    "avoid",
    "danger",
    "warning",
    "interaction",
    "liver",
    "kidney",
    "serious",
    "do not",
    "fatal",
    "contraind",
)


# superseded by classify_risk_from_chunks in generate.py, kept for reference
# def classify_risk_from_answer(answer_text: str, drug_a: str, drug_b: str) -> str:
#     """
#     Classifies interaction risk for Medication Checker pairs.
#
#     Order:
#       a. Sentences mentioning both drugs
#       b. HIGH keywords in those sentences -> HIGH
#       c. NO_INTERACTION phrases in those sentences -> LOW
#       d. If no both-drug sentences: full-answer HIGH / MODERATE / NO_INTERACTION / LOW
#     """
#     ... (prose-based classifier retired — UI must use result["risk_level"] from generate.py)


def classify_risk_from_answer(answer_text: str, drug_a: str, drug_b: str) -> str:
    """Deprecated stub. Use generate.classify_risk_from_chunks / result['risk_level']."""
    return "UNKNOWN"


def _filter_citations_for_pair(citations: list[dict], drug_a: str, drug_b: str) -> list[dict]:
    drug_a_lower = drug_a.lower()
    drug_b_lower = drug_b.lower()
    filtered = [
        cit
        for cit in citations
        if drug_a_lower in str(cit.get("drug", "")).lower()
        or drug_b_lower in str(cit.get("drug", "")).lower()
    ]
    return filtered


def _extract_key_warning(answer: str, abstained: bool, pair_risk: str) -> str:
    if abstained:
        return "UNKNOWN — insufficient FDA data"
    if pair_risk == "LOW":
        return "No significant interaction"
    body = answer or ""
    if DISCLAIMER_PHRASE.lower() in body.lower():
        idx = body.lower().rfind(DISCLAIMER_PHRASE.lower())
        if idx >= 0:
            body = body[:idx].strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    for sentence in sentences:
        cleaned = _strip_citations(sentence)
        lower = cleaned.lower()
        if _is_filler_sentence(cleaned):
            continue
        if any(term in lower for term in PAIR_WARNING_TERMS):
            return cleaned[:100] + ("…" if len(cleaned) > 100 else "")
    if sentences:
        cleaned = _strip_citations(sentences[0])
        return cleaned[:100] + ("…" if len(cleaned) > 100 else "")
    return "See detailed results"


def resolve_medication_entry(medication: str) -> dict:
    """Resolve a single medication name for Medication Checker add-to-list."""
    raw = (medication or "").strip()
    if not raw:
        return {"error": "Please enter a medication name."}

    rewritten, substitutions = resolve_query(raw)
    resolved = rewritten.strip() or raw
    brand = None
    generic = None
    for item in substitutions:
        if "->" not in item:
            continue
        brand_part, generic_part = item.split("->", 1)
        brand = brand_part.strip()
        generic = generic_part.strip()
        break

    notice = None
    if brand and generic and brand.lower() != generic.lower():
        notice = f"{brand.title()} resolved to {generic}"

    return {
        "input": raw,
        "resolved": resolved,
        "brand": brand or "",
        "generic": generic or resolved,
        "notice": notice,
        "substitutions": substitutions,
    }


def prepare_medication_check(medications_text: str) -> dict:
    raw_lines = [line.strip() for line in (medications_text or "").splitlines() if line.strip()]
    if len(raw_lines) < 2:
        return {"error": "Please enter at least 2 different medications"}

    seen: set[str] = set()
    drugs: list[str] = []
    resolutions: list[dict[str, str]] = []
    for line in raw_lines:
        rewritten, substitutions = resolve_query(line)
        resolved = rewritten.strip()
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        drugs.append(resolved)
        for item in substitutions:
            if "->" not in item:
                continue
            brand, generic = item.split("->", 1)
            brand = brand.strip()
            generic = generic.strip()
            resolutions.append(
                {
                    "brand": brand,
                    "generic": generic,
                    "display": f"{brand.title()} resolved to {generic}",
                }
            )

    if len(drugs) < 2:
        return {"error": "Please enter at least 2 different medications"}

    pairs = [(drugs[i], drugs[j]) for i in range(len(drugs)) for j in range(i + 1, len(drugs))]
    warning = None
    if len(drugs) > 10:
        warning = f"Checking {len(pairs)} pairs may take several minutes"

    return {
        "medications": drugs,
        "resolutions": resolutions,
        "pairs": [{"drug_a": a, "drug_b": b} for a, b in pairs],
        "pair_count": len(pairs),
        "medication_count": len(drugs),
        "warning": warning,
    }


def _process_pair_question(drug_a: str, drug_b: str) -> dict:
    """Medication Checker-only processing with pair-filtered context."""
    question = f"Is it safe to take {drug_a} and {drug_b} together?"
    start = time.perf_counter()
    try:
        result = answer_query_for_pair(drug_a, drug_b, store=get_store())
    except Exception:
        return {"error": "Something went wrong. Please try again."}

    elapsed = time.perf_counter() - start
    answer = result.get("answer", "")
    abstained = bool(result.get("abstained"))
    substitutions = result.get("substitutions") or result.get("substitutions_made") or []
    citations = result.get("citations") or []
    citations = _filter_citations_for_pair(citations, drug_a, drug_b)
    limited_sources_note = None
    if not abstained and len(citations) < 2:
        limited_sources_note = "Limited FDA label data available for this specific combination."

    paragraphs, disclaimer = _split_answer(answer)
    context_text = ""
    if not abstained:
        context_text = _context_from_citations(result.get("rewritten_query", question), citations)
    metrics = (
        _derive_metrics(answer, citations, substitutions, context_text, question)
        if not abstained
        else None
    )
    # Override prose-inferred metrics with pipeline risk from classify_risk_from_chunks.
    if metrics is not None:
        risk, risk_level, _ = _map_pipeline_risk(result.get("risk_level"))
        metrics["risk"] = risk
        metrics["risk_level"] = risk_level
    structured = _structure_answer(answer, metrics, abstained)
    if structured and metrics is not None:
        structured["card_risk_level"] = metrics["risk_level"]
        if metrics["risk_level"] in ("low", "unknown"):
            structured["critical_risks"] = []
    structured = _sanitize_structured(structured)
    answer = _sanitize_dashes(answer)
    paragraphs = [_sanitize_dashes(p) for p in paragraphs]
    if disclaimer:
        disclaimer = _sanitize_dashes(disclaimer)

    snippets = (
        _fetch_snippets(result.get("rewritten_query", question), citations) if citations else []
    )
    cited_labels = []
    for cit, snippet in zip(citations, snippets, strict=False):
        cited_labels.append(
            {
                **cit,
                "section_display": cit["section"].replace("_", " "),
                "snippet": snippet,
            }
        )

    return {
        "original_query": result.get("original_query", question),
        "rewritten_query": result.get("rewritten_query", question),
        "substitutions": substitutions,
        "brand_substitutions": _format_brand_substitutions(substitutions),
        "combined_use_risk": None if abstained else _detect_combined_use(citations, metrics),
        "abstained": abstained,
        "abstention_reason": result.get("abstention_reason", ""),
        "answer": answer,
        "paragraphs": paragraphs,
        "structured": structured,
        "disclaimer": disclaimer,
        "metrics": metrics,
        "citations": cited_labels,
        "limited_sources_note": limited_sources_note,
        "elapsed_seconds": round(elapsed, 1),
        "risk_level": result.get("risk_level"),
    }


def check_medication_pair(drug_a: str, drug_b: str) -> dict:
    drug_a = (drug_a or "").strip()
    drug_b = (drug_b or "").strip()
    if not drug_a or not drug_b:
        return {"error": "Both drug names are required."}

    # Safety net: same resolved drug should never hit the LLM pair path.
    if drug_a.lower() == drug_b.lower():
        warning = "Same medication — no interaction check needed"
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "pair_risk": "N/A",
            "key_warning": warning,
            "abstained": False,
            "abstention_reason": "",
            "answer": warning,
            "paragraphs": [warning],
            "structured": {
                "simple_guide": [warning],
                "clinical_details": [warning],
                "disclaimer": "",
            },
            "disclaimer": "",
            "citations": [],
            "limited_sources_note": None,
            "metrics": None,
            "substitutions": [],
            "brand_substitutions": [],
            "combined_use_risk": None,
            "elapsed_seconds": 0,
        }

    result = _process_pair_question(drug_a, drug_b)
    if result.get("error"):
        return result

    abstained = bool(result.get("abstained"))
    if abstained:
        pair_risk = "UNKNOWN"
    else:
        pipeline_risk = result.get("risk_level")
        if pipeline_risk in ("HIGH", "MODERATE", "LOW", "UNKNOWN"):
            pair_risk = pipeline_risk
        else:
            pair_risk = "UNKNOWN"
    key_warning = _extract_key_warning(result.get("answer", ""), abstained, pair_risk)

    result["drug_a"] = drug_a
    result["drug_b"] = drug_b
    result["pair_risk"] = pair_risk
    result["key_warning"] = key_warning
    result["risk_level"] = result.get("risk_level")
    return result
