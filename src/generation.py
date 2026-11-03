"""
Generation of answers from retrieved evidence, with citation validation and abstention.

This module provides a LocalGenerator class that wraps a base model and an optional LoRA adapter for
greedy generation. 

It also defines the answer_question function that orchestrates retrieval, generation,
and validation, returning a structured Answer object.

The generator ensures that the prompt is never truncated, and the model is called only when sufficient
evidence is retrieved. 

The generated answer is validated for content and citations before being returned.

The module also includes utility functions for citation extraction and validation, as well as adapter
provenance checks to ensure compatibility with the base model and training data.
"""

import hashlib
import json
import re
from time import perf_counter
from typing import Literal

import torch
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .prompts import FALLBACK_ANSWER, evidence_text, inference_prompt
from .retrieval import Retriever, SearchHit

SystemName = Literal["rag-base", "rag-lora"]
Status = Literal["answered", "abstained", "rejected"]
REJECTED_MESSAGE = "The generated answer failed citation validation. Escalate for manual review."

# Deliberately broader than data.CITATION: it catches ANYTHING in [...], so an
# invented or malformed ID like [Shipping Policy] is seen and rejected, not ignored.
ANY_BRACKET = re.compile(r"\[([^\[\]]*)\]")


class Answer(BaseModel):
    system: SystemName
    question: str
    status: Status
    answer: str                          # what the customer/agent would see
    raw_answer: str | None = None        # model text before validation (None = model not called)
    citations: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    top_score: float | None = None
    latency_seconds: float = 0.0


def citations_in(text: str) -> list[str]:
    """Unique bracketed IDs in order of appearance."""
    return list(dict.fromkeys(c.strip() for c in ANY_BRACKET.findall(text)))


def citations_are_valid(text: str, hits: list[SearchHit]) -> bool:
    """Accept only answers that say something AND cite only retrieved sources."""
    has_content = bool(ANY_BRACKET.sub("", text).strip())   # a) not just "[shipping]"
    cited = set(citations_in(text))
    retrieved = {hit.document_id for hit in hits}
    return has_content and bool(cited) and cited <= retrieved  # b) cites something, c) nothing invented


def is_model_abstention(text: str) -> bool:
    """The prompt tells the model to reply with FALLBACK_ANSWER when evidence is insufficient."""
    return "approved material is insufficient" in text.lower()


def validate_adapter() -> None:
    """Refuse to load an adapter that doesn't match the current model and training data."""
    summary_path = config.ADAPTER_PATH / "training_summary.json"

    # 1. No provenance -> we can't tell what this adapter is
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No training summary at {summary_path}; run: uv run python -m src.training"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # 2. LoRA weights only fit the exact base model (and revision) they were trained on
    trained_on = (summary.get("base_model"), summary.get("base_model_revision"))
    configured = (config.BASE_MODEL, config.BASE_MODEL_REVISION)
    if trained_on != configured:
        raise ValueError(f"Adapter was trained on {trained_on}, but config uses {configured}; retrain")

    # 3. Training data must be byte-identical to what the adapter learned from
    current_sha = hashlib.sha256(config.TRAIN_FILE.read_bytes()).hexdigest()
    if summary.get("training_sha256") != current_sha:
        raise ValueError("Training data changed since the adapter was trained; retrain")


class LocalGenerator:
    """Greedy local generation with the base model, optionally plus the LoRA adapter."""

    def __init__(self, use_adapter: bool) -> None:
        self.use_adapter = use_adapter
        if use_adapter:
            validate_adapter()          # check BEFORE spending time loading weights
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL, revision=config.BASE_MODEL_REVISION
        )
        # Each generator loads its OWN base model. PeftModel injects LoRA layers into
        # the model object you pass in, so sharing one base would silently turn
        # rag-base into rag-lora.
        base = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL, revision=config.BASE_MODEL_REVISION, dtype=torch.float32
        )
        self.model = PeftModel.from_pretrained(base, str(config.ADAPTER_PATH)) if use_adapter else base
        self.model.eval()               # disables dropout -> deterministic outputs

    def generate(self, question: str, hits: list[SearchHit]) -> str:
        prompt = inference_prompt(self.tokenizer, question, evidence_text(hits))
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_len = inputs["input_ids"].shape[1]

        # 1. Never truncate: that would cut off the question or part of the evidence
        if prompt_len > config.MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Prompt has {prompt_len} tokens > MAX_SEQUENCE_LENGTH={config.MAX_SEQUENCE_LENGTH}"
            )

        # 2. Greedy decoding: same input -> same answer, so both systems are comparable
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3. generate() returns prompt + answer; keep only the newly generated tokens
        new_tokens = output[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_question(
    question: str, system: SystemName, retriever: Retriever, generator: LocalGenerator
) -> Answer:
    started = perf_counter()
    hits = retriever.search(question)
    common = {
        "system": system,
        "question": question,
        "retrieved_sources": list(dict.fromkeys(h.document_id for h in hits)),
        "top_score": hits[0].score if hits else None,
    }

    def finish(**fields) -> Answer:
        return Answer(**common, **fields, latency_seconds=perf_counter() - started)

    # 1. Retrieval-layer abstention: weak evidence -> don't even call the model
    if not hits or hits[0].score < config.MIN_SCORE:
        return finish(status="abstained", answer=FALLBACK_ANSWER)

    # 2. Generate from the shared prompt + evidence
    raw = generator.generate(question, hits)

    # 3. Model-layer abstention: the model itself said the evidence is insufficient
    if is_model_abstention(raw):
        return finish(status="abstained", answer=FALLBACK_ANSWER, raw_answer=raw)

    # 4. Citation validation: hide the raw text, keep it for analysis
    if not citations_are_valid(raw, hits):
        return finish(
            status="rejected",
            answer=REJECTED_MESSAGE,
            raw_answer=raw,
            citations=citations_in(raw),
        )

    # 5. Passed every check
    return finish(status="answered", answer=raw, raw_answer=raw, citations=citations_in(raw))