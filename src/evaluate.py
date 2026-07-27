"""
Formal evaluation suite for MedSafe.

Runs test questions through the full generation pipeline, checks pass/fail
criteria, scores non-abstention answers with RAGAS (or LLM-as-judge fallback),
and writes eval/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from embed import VectorStore, _strip_chunk_prefix, load_vector_store
from generate import (
    DEFAULT_MODEL,
    ENV_PATH,
    _require_openai_api_key,
    answer_query,
    retrieve_for_question,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "test_questions.json"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "eval" / "results.csv"

SNIPPET_MAX_CHARS = 200
JUDGE_MODEL = DEFAULT_MODEL

CATEGORY_LABELS = {
    "single_drug_warnings": "single_drug_warnings",
    "single_drug_dosage": "single_drug_dosage",
    "two_drug_interaction": "two_drug_interaction",
    "brand_name": "brand_name",
    "out_of_scope": "out_of_scope",
}


def _configure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_test_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as infile:
        questions = json.load(infile)
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No test questions found in {path}")
    return questions


def _contexts_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    return [_strip_chunk_prefix(chunk.get("text", "")) for chunk in chunks]


def _answer_snippet(answer: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    compact = " ".join(answer.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def evaluate_pass_fail(question: dict[str, Any], result: dict[str, Any]) -> tuple[str, str]:
    """
    Apply automated pass/fail rules.

    Returns (pass_fail, notes).
    """
    expected = question["expected_behavior"]
    abstained = bool(result["abstained"])
    notes = result.get("abstention_reason", "") or ""

    if expected == "abstain":
        if abstained:
            return "PASS", notes or "Correctly abstained"
        return "FAIL", "Expected abstention but model answered"

    if abstained:
        return "FAIL", notes or "Expected answer but abstained"

    answer_lower = result["answer"].lower()
    keywords = question.get("ground_truth_keywords") or []
    if not keywords:
        return "PASS", "No keywords configured"

    matched = [kw for kw in keywords if kw.lower() in answer_lower]
    if matched:
        return "PASS", f"Matched keywords: {', '.join(matched)}"

    return "FAIL", f"No ground-truth keywords found in answer ({', '.join(keywords)})"


def _parse_judge_score(text: str) -> float | None:
    match = re.search(r"(?<!\d)([1-5](?:\.\d+)?)(?!\d)", text.strip())
    if not match:
        return None
    score = float(match.group(1))
    if 1.0 <= score <= 5.0:
        return score
    return None


def _llm_judge_score(client: OpenAI, prompt: str) -> float | None:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return _parse_judge_score(content)


def score_with_llm_judge(
    client: OpenAI,
    question: str,
    answer: str,
    contexts: list[str],
) -> tuple[float | None, float | None]:
    context_block = "\n\n---\n\n".join(contexts[:5])

    faithfulness_prompt = (
        "On a scale of 1-5, how faithful is this answer to the provided FDA label "
        "context? Score 1 if the answer adds unsupported claims; score 5 if every "
        "factual claim is supported by the context.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Answer:\n{answer}\n\n"
        "Reply with just a number from 1 to 5."
    )
    relevancy_prompt = (
        "On a scale of 1-5, how well does this answer address the caregiver's "
        "question? Score 1 if it misses the question; score 5 if it directly and "
        "helpfully addresses it.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}\n\n"
        "Reply with just a number from 1 to 5."
    )

    faithfulness = _llm_judge_score(client, faithfulness_prompt)
    relevancy = _llm_judge_score(client, relevancy_prompt)
    return faithfulness, relevancy


def score_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> tuple[float | None, float | None]:
    """Score a single example with RAGAS; returns None values on failure."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        return None, None

    if not contexts:
        return None, None

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )
    try:
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        row = scores.to_pandas().iloc[0]
        faithfulness_score = float(row["faithfulness"]) if row.get("faithfulness") == row.get("faithfulness") else None
        relevancy_score = (
            float(row["answer_relevancy"]) if row.get("answer_relevancy") == row.get("answer_relevancy") else None
        )
        if faithfulness_score is not None:
            faithfulness_score = min(5.0, max(1.0, faithfulness_score * 5))
        if relevancy_score is not None:
            relevancy_score = min(5.0, max(1.0, relevancy_score * 5))
        return faithfulness_score, relevancy_score
    except Exception:
        return None, None


def score_answer(
    client: OpenAI,
    question: str,
    answer: str,
    contexts: list[str],
    *,
    prefer_ragas: bool = True,
) -> tuple[float | None, float | None, str]:
    """Return faithfulness, relevancy, and scoring method used."""
    if prefer_ragas:
        faithfulness, relevancy = score_with_ragas(question, answer, contexts)
        if faithfulness is not None and relevancy is not None:
            return faithfulness, relevancy, "ragas"

    faithfulness, relevancy = score_with_llm_judge(client, question, answer, contexts)
    return faithfulness, relevancy, "llm_judge"


def evaluate_question(
    question: dict[str, Any],
    store: VectorStore,
    client: OpenAI,
    *,
    prefer_ragas: bool = True,
) -> dict[str, Any]:
    result = answer_query(question["question"], store=store)
    pass_fail, eval_notes = evaluate_pass_fail(question, result)

    faithfulness_score: float | None = None
    relevancy_score: float | None = None
    scoring_method = ""

    if not result["abstained"]:
        chunks = retrieve_for_question(result["rewritten_query"], store=store)
        contexts = _contexts_from_chunks(chunks)
        faithfulness_score, relevancy_score, scoring_method = score_answer(
            client,
            question["question"],
            result["answer"],
            contexts,
            prefer_ragas=prefer_ragas,
        )

    notes = eval_notes
    if scoring_method:
        notes = f"{eval_notes}; scored via {scoring_method}".strip("; ")

    return {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "original_query": result.get("original_query", question["question"]),
        "rewritten_query": result.get("rewritten_query", question["question"]),
        "abstained": result["abstained"],
        "pass_fail": pass_fail,
        "faithfulness_score": faithfulness_score,
        "relevancy_score": relevancy_score,
        "answer_snippet": _answer_snippet(result["answer"]),
        "answer": result["answer"],
        "citations_count": len(result.get("citations") or []),
        "notes": notes,
        "expected_behavior": question["expected_behavior"],
    }


def write_results_csv(rows: list[dict[str, Any]], path: Path = DEFAULT_RESULTS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "question",
        "abstained",
        "pass_fail",
        "faithfulness_score",
        "relevancy_score",
        "answer_snippet",
        "citations_count",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "question": row["question"],
                    "abstained": row["abstained"],
                    "pass_fail": row["pass_fail"],
                    "faithfulness_score": row["faithfulness_score"] if row["faithfulness_score"] is not None else "",
                    "relevancy_score": row["relevancy_score"] if row["relevancy_score"] is not None else "",
                    "answer_snippet": row["answer_snippet"],
                    "citations_count": row["citations_count"],
                    "notes": row["notes"],
                }
            )


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    passed = sum(1 for row in rows if row["pass_fail"] == "PASS")
    failed = total - passed

    abstention_rows = [row for row in rows if row["expected_behavior"] == "abstain"]
    abstention_pass = sum(1 for row in abstention_rows if row["pass_fail"] == "PASS")
    abstention_total = len(abstention_rows)

    scored_rows = [
        row
        for row in rows
        if not row["abstained"] and row["faithfulness_score"] is not None and row["relevancy_score"] is not None
    ]
    avg_faithfulness = (
        sum(row["faithfulness_score"] for row in scored_rows) / len(scored_rows) if scored_rows else 0.0
    )
    avg_relevancy = (
        sum(row["relevancy_score"] for row in scored_rows) / len(scored_rows) if scored_rows else 0.0
    )

    failures_by_category: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["pass_fail"] == "FAIL":
            failures_by_category[row["category"]] += 1

    print()
    print(f"Total questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Abstention accuracy: {abstention_pass}/{abstention_total}")
    print(f"Average faithfulness: {avg_faithfulness:.1f}/5")
    print(f"Average relevancy: {avg_relevancy:.1f}/5")
    print()
    print("Failures by category:")
    for category in CATEGORY_LABELS:
        print(f"  {category}: {failures_by_category[category]} failures")
    print()


def run_evaluation(
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    sample: int | None = None,
    *,
    prefer_ragas: bool = True,
) -> list[dict[str, Any]]:
    load_dotenv(ENV_PATH)
    _require_openai_api_key()

    questions = load_test_questions(questions_path)
    if sample is not None:
        questions = questions[:sample]

    store = load_vector_store()
    client = OpenAI(api_key=_require_openai_api_key())

    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question['id']}: {question['question']}")
        row = evaluate_question(question, store, client, prefer_ragas=prefer_ragas)
        rows.append(row)
        print(f"  -> {row['pass_fail']} | abstained={row['abstained']}")

    write_results_csv(rows, results_path)
    print(f"Wrote results to {results_path}")
    print_summary(rows)
    return rows


def main() -> None:
    _configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Evaluate MedSafe RAG on the test question suite.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to eval/test_questions.json",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to write eval/results.csv",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Run only the first N questions (quick smoke test).",
    )
    parser.add_argument(
        "--llm-judge-only",
        action="store_true",
        help="Skip RAGAS and use GPT-4o-mini judge scoring only.",
    )
    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions_path,
        results_path=args.results_path,
        sample=args.sample,
        prefer_ragas=not args.llm_judge_only,
    )


if __name__ == "__main__":
    main()
