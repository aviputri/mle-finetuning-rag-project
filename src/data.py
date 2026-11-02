"""
Data models and loading functions for the knowledge base, training examples, and evaluation cases.
Fails loudly if the three layers disagree or overlap, e.g. a training example cites a source that doesn't exist.
"""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ValidationError


class SourceDocument(BaseModel):
    document_id: str   # filename stem, e.g. "refunds"
    title: str         # first line without "# "
    text: str          # everything after the title


class Chunk(BaseModel):
    chunk_id: str      # f"{document_id}-{position}"
    document_id: str
    title: str
    position: int
    text: str


class TrainingExample(BaseModel):
    id: str
    reference_ids: list[str]
    context: str
    question: str
    answer: str


class EvaluationCase(BaseModel):
    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    forbidden_terms: list[str]
    should_abstain: bool


CITATION = re.compile(r"\[([a-z0-9-]+)\]")   # finds "refunds" in "[refunds]"


def load_documents(kb_dir: Path) -> list[SourceDocument]:
    """Load each Markdown policy; the filename becomes its stable source ID."""
    documents = []
    # sorted() keeps the order identical on every machine -> reproducible chunk IDs
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{path.name} must start with a '# Title' line")
        documents.append(
            SourceDocument(
                document_id=path.stem,                      # "refunds.md" -> "refunds"
                title=lines[0].removeprefix("# ").strip(),
                text="\n".join(lines[1:]).strip(),
            )
        )
    if not documents:
        raise ValueError(f"No Markdown policies found in {kb_dir}")
    return documents


def chunk_documents(documents: list[SourceDocument]) -> list[Chunk]:
    """One chunk per paragraph, carrying the source metadata needed for citations."""
    chunks = []
    for doc in documents:
        # Paragraphs are separated by blank lines; lines inside one are just wrapped text
        paragraphs = [p.replace("\n", " ").strip() for p in doc.text.split("\n\n")]
        position = 0
        for paragraph in paragraphs:
            if not paragraph:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-{position}",
                    document_id=doc.document_id,
                    title=doc.title,
                    position=position,
                    text=paragraph,
                )
            )
            position += 1
    return chunks


def load_training_examples(path: Path) -> list[TrainingExample]:
    """Parse JSON Lines; report the exact line if a record is malformed."""
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(TrainingExample.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"Invalid training record at {path.name}:{line_number}") from error
    if not examples:
        raise ValueError(f"No training examples in {path}")
    return examples


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Parse the held-out benchmark (a single JSON array)."""
    records = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvaluationCase.model_validate(record) for record in records]
    if not cases:
        raise ValueError(f"No evaluation cases in {path}")
    return cases


def _check_unique(ids: list[str], label: str) -> None:
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {duplicates}")


def validate_data(documents, examples, cases) -> None:
    """Fail loudly if the three data layers disagree or overlap."""
    source_ids = {d.document_id for d in documents}

    # 1. IDs must be unique, or traces and metrics become ambiguous
    _check_unique([e.id for e in examples], "training")
    _check_unique([c.id for c in cases], "evaluation")

    for example in examples:
        # 2. Every referenced source must exist in the knowledge base
        missing = set(example.reference_ids) - source_ids
        if missing:
            raise ValueError(f"{example.id} references unknown sources: {sorted(missing)}")
        # 3. The target answer may only cite sources present in its own context
        cited = set(CITATION.findall(example.answer))
        if not cited <= set(example.reference_ids):
            raise ValueError(
                f"{example.id} answer cites {sorted(cited - set(example.reference_ids))}, "
                "which is not in its reference_ids"
            )

    # 4. Expected evaluation sources must exist (abstain cases have [] - that's fine)
    for case in cases:
        missing = set(case.expected_sources) - source_ids
        if missing:
            raise ValueError(f"{case.id} expects unknown sources: {sorted(missing)}")

    # 5. Held-out questions must not appear in training (case-insensitive)
    training_questions = {e.question.casefold() for e in examples}
    leaked = [c.id for c in cases if c.question.casefold() in training_questions]
    if leaked:
        raise ValueError(f"Evaluation questions leaked into training: {leaked}")