"""
Evaluation and benchmarking of retrieval and generation quality, with markdown report output.

The benchmark is designed to be reproducible: it uses a fixed set of evaluation cases, and
the generator is called with deterministic greedy decoding."""

import json
import re
from statistics import mean

from . import config
from .data import EvaluationCase, load_evaluation_cases
from .generation import Answer, LocalGenerator, answer_question
from .retrieval import Retriever

SYSTEMS = ("rag-base", "rag-lora")
HEDGES = ("maybe", "probably", "perhaps")


# ---------- metric functions ----------

def contains_term(text: str, term: str) -> bool:
    """Case-insensitive whole-phrase match ("not" must not match inside "note")."""
    # (?<!\w) = no letter/digit right before, (?!\w) = none right after.
    # re.escape makes characters like "." or "+" in a term literal.
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def term_coverage(text: str, terms: list[str]) -> float:
    """Fraction of required terms present: a transparent relevance proxy."""
    if not terms:
        return 1.0
    return sum(contains_term(text, term) for term in terms) / len(terms)


def first_relevant_rank(retrieved: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first retrieved source that was expected."""
    for rank, source_id in enumerate(retrieved, start=1):
        if source_id in expected:
            return rank
    return None


def retrieval_metrics(rows: list[dict]) -> tuple[float, float]:
    """recall@k and MRR over supported cases only (unsupported have no right answer)."""
    supported = [row for row in rows if not row["should_abstain"]]
    if not supported:
        return 0.0, 0.0
    recall = sum(row["rank"] is not None for row in supported) / len(supported)
    mrr = mean(1 / row["rank"] if row["rank"] else 0.0 for row in supported)
    return recall, mrr


def relevance(case: EvaluationCase, a: Answer) -> float:
    """Unsupported: reward abstaining. Supported: required-term coverage of the shown answer."""
    if case.should_abstain:
        return 1.0 if a.status == "abstained" else 0.0
    if a.status != "answered":
        return 0.0                      # rejected/abstained text is not a usable answer
    return term_coverage(a.answer, case.required_terms)


def faithful(case: EvaluationCase, a: Answer) -> bool:
    """Citation-and-forbidden-claim proxy, NOT a semantic judge."""
    if case.should_abstain:
        return a.status == "abstained"
    if a.status != "answered" or not a.citations:
        return False
    cites_expected_only = set(a.citations) <= set(case.expected_sources)
    no_forbidden_claim = not any(contains_term(a.answer, t) for t in case.forbidden_terms)
    return cites_expected_only and no_forbidden_claim


def style_score(a: Answer) -> float:
    """Share of concise-reply checks passed; rejected answers score 0, abstentions 1."""
    if a.status == "abstained":
        return 1.0
    if a.status == "rejected":
        return 0.0
    text = a.answer
    checks = (
        len(text.split()) <= 80,                         # concise overall
        len(text.split(".")[0].split()) <= 30,           # direct first sentence
        not any(h in text.lower() for h in HEDGES),      # no vague hedging
    )
    return sum(checks) / len(checks)


def decision_correct(case: EvaluationCase, a: Answer) -> bool:
    """Answer when the policies cover it, abstain when they don't."""
    return a.status == ("abstained" if case.should_abstain else "answered")

# ---------- running the benchmark (complete) ----------

def evaluate_retrieval(retriever: Retriever, cases: list[EvaluationCase]) -> list[dict]:
    rows = []
    for case in cases:
        hits = retriever.search(case.question)
        retrieved = list(dict.fromkeys(h.document_id for h in hits))
        rows.append({
            "id": case.id,
            "should_abstain": case.should_abstain,
            "expected": case.expected_sources,
            "retrieved": retrieved,
            "rank": first_relevant_rank(retrieved, case.expected_sources),
            "top_score": round(hits[0].score, 3) if hits else None,
        })
    return rows


def evaluate_system(system, cases, retriever, generator) -> list[dict]:
    rows = []
    for case in cases:
        a = answer_question(case.question, system, retriever, generator)
        rows.append({
            "id": case.id,
            "system": system,
            "should_abstain": case.should_abstain,
            "status": a.status,
            "relevance": relevance(case, a),
            "raw_relevance": None if case.should_abstain
                             else term_coverage(a.raw_answer or "", case.required_terms),
            "faithful": faithful(case, a),
            "style": style_score(a),
            "decision_correct": decision_correct(case, a),
            "latency": round(a.latency_seconds, 3),
            "citations": a.citations,
            "answer": a.answer,
            "raw_answer": a.raw_answer,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    supported = [r for r in rows if not r["should_abstain"]]
    return {
        "relevance": mean(r["relevance"] for r in rows),
        "raw_relevance_supported": mean(r["raw_relevance"] for r in supported),
        "faithfulness": mean(float(r["faithful"]) for r in rows),
        "style": mean(r["style"] for r in rows),
        "decision_accuracy": mean(float(r["decision_correct"]) for r in rows),
        "rejections": sum(r["status"] == "rejected" for r in rows),
        "mean_latency": mean(r["latency"] for r in rows),
    }


# ---------- markdown rendering (complete) ----------

def _cell(text) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def render_markdown(retrieval_rows: list[dict], system_rows: dict[str, list[dict]],
                    training: dict | None = None) -> str:
    recall, mrr = retrieval_metrics(retrieval_rows)
    sup = [r["top_score"] for r in retrieval_rows if not r["should_abstain"]]
    uns = [r["top_score"] for r in retrieval_rows if r["should_abstain"]]
    summaries = {name: summarize(rows) for name, rows in system_rows.items()}

    out = ["# Benchmark Results", "",
           "_Generated by `uv run python -m src.evaluation`; do not edit by hand._", "",
           "## Configuration", "",
           f"- Embedding: `{config.EMBEDDING_MODEL}` @ `{config.EMBEDDING_REVISION}`",
           "- Vector store: Qdrant local (in-memory), cosine similarity, one chunk per paragraph",
           f"- Retrieval: `TOP_K={config.TOP_K}`, `MIN_SCORE={config.MIN_SCORE}`",
           f"- Base model: `{config.BASE_MODEL}` @ `{config.BASE_MODEL_REVISION}`",
           f"- Generation: greedy, `MAX_NEW_TOKENS={config.MAX_NEW_TOKENS}`, "
           f"prompt limit `{config.MAX_SEQUENCE_LENGTH}` tokens, shared prompt for both systems"]
    if training:
        out += [
            f"- LoRA: targets `{', '.join(training['lora_target_modules'])}`, r={training['lora_rank']}, "
            f"alpha={training['lora_alpha']}, dropout={training['lora_dropout']}",
            f"- Training: {training['epochs']} epochs, lr {training['learning_rate']}, seed "
            f"{training['random_seed']}, {training['training_examples']} examples, "
            f"`{training['device']}`, {training['training_seconds']} s",
            f"- Trainable parameters: {training['trainable_parameters']:,} / "
            f"{training['total_parameters']:,} ({training['trainable_percentage']}%)",
            f"- Loss: {training['epoch_losses'][0]} -> {training['epoch_losses'][-1]}",
            f"- Training data SHA-256: `{training['training_sha256']}`",
        ]

    out += ["", "## Retrieval (shared by both systems)", "",
            "| Metric | Score |", "| --- | ---: |",
            f"| recall@{config.TOP_K} (supported cases) | {recall:.3f} |",
            f"| MRR (supported cases) | {mrr:.3f} |", "",
            "| Case | Should abstain | Top score | Expected | Retrieved | First relevant rank |",
            "| --- | --- | ---: | --- | --- | ---: |"]
    for r in retrieval_rows:
        out.append(f"| {r['id']} | {r['should_abstain']} | {r['top_score']:.3f} | "
                   f"{', '.join(r['expected']) or '-'} | {', '.join(r['retrieved'])} | {r['rank'] or '-'} |")
    out += ["", f"Threshold evidence: supported top scores {min(sup):.3f}-{max(sup):.3f}; "
                f"unsupported top scores {min(uns):.3f}-{max(uns):.3f}; threshold {config.MIN_SCORE}."]

    out += ["", "## System Comparison", "",
            "| System | Relevance | Raw relevance (supported) | Faithfulness | Style | "
            "Decision accuracy | Rejections | Mean latency (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in summaries.items():
        out.append(f"| {name} | {s['relevance']:.3f} | {s['raw_relevance_supported']:.3f} | "
                   f"{s['faithfulness']:.3f} | {s['style']:.3f} | {s['decision_accuracy']:.3f} | "
                   f"{s['rejections']}/{len(system_rows[name])} | {s['mean_latency']:.2f} |")
    base, lora = summaries["rag-base"], summaries["rag-lora"]
    out += ["", "**Hybrid minus baseline:** " + ", ".join(
        f"{k} {lora[k] - base[k]:+.3f}"
        for k in ("relevance", "faithfulness", "style", "decision_accuracy", "mean_latency"))]

    out += ["", "## Per-Case Comparison", "",
            "| Case | System | Status | Relevance | Raw rel. | Faithful | Style | Decision | Citations | Latency (s) |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | ---: |"]
    for case_id in [r["id"] for r in retrieval_rows]:
        for name, rows in system_rows.items():
            r = next(x for x in rows if x["id"] == case_id)
            raw = "-" if r["raw_relevance"] is None else f"{r['raw_relevance']:.2f}"
            out.append(f"| {case_id} | {name} | {r['status']} | {r['relevance']:.2f} | {raw} | "
                       f"{'yes' if r['faithful'] else 'no'} | {r['style']:.2f} | "
                       f"{'yes' if r['decision_correct'] else 'no'} | {', '.join(r['citations']) or '-'} | "
                       f"{r['latency']:.2f} |")

    out += ["", "## Answers (shown vs raw)", ""]
    for case_id in [r["id"] for r in retrieval_rows]:
        out.append(f"**{case_id}**")
        for name, rows in system_rows.items():
            r = next(x for x in rows if x["id"] == case_id)
            out.append(f"- `{name}` ({r['status']}): {_cell(r['answer'])}")
            if r["raw_answer"] and r["raw_answer"] != r["answer"]:
                out.append(f"  - raw: {_cell(r['raw_answer'])}")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    cases = load_evaluation_cases(config.EVAL_FILE)
    retriever = Retriever()
    retrieval_rows = evaluate_retrieval(retriever, cases)

    system_rows = {}
    for system in SYSTEMS:
        generator = LocalGenerator(use_adapter=(system == "rag-lora"))
        system_rows[system] = evaluate_system(system, cases, retriever, generator)
        del generator                                   # free memory before loading the next

    summary_path = config.ADAPTER_PATH / "training_summary.json"
    training = json.loads(summary_path.read_text()) if summary_path.exists() else None

    report = config.ROOT / "reports" / "benchmark_results.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(render_markdown(retrieval_rows, system_rows, training), encoding="utf-8")

    # Full trace (raw answers etc.) stays out of Git: artifacts/ is ignored
    trace = config.ROOT / "artifacts" / "benchmark_trace.json"
    trace.parent.mkdir(exist_ok=True)
    trace.write_text(json.dumps({"retrieval": retrieval_rows, "systems": system_rows}, indent=2))

    for name, rows in system_rows.items():
        print(name, {k: round(v, 3) for k, v in summarize(rows).items()})
    print(f"\nWrote {report.relative_to(config.ROOT)} and {trace.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()