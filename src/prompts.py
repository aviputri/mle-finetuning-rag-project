"""
Prompts for the model, including the system instruction and the formatting of the context and question.
"""

from transformers import PreTrainedTokenizerBase

FALLBACK_ANSWER = "The approved material is insufficient. Escalate for manual review."
SYSTEM_INSTRUCTION = (
    "Answer only from approved context. Start with the direct answer, then "
    "give the next action. Cite every policy claim with source IDs in square "
    "brackets. If the context is insufficient, reply exactly: "
    f"{FALLBACK_ANSWER}"
)


def evidence_text(hits) -> str:
    """Retrieved chunks -> '[doc_id] text' lines (same format as the training contexts)."""
    return "\n".join(f"[{hit.document_id}] {hit.text}" for hit in hits)


def _messages(question: str, context: str) -> list[dict[str, str]]:
    context = context or "No approved policy evidence was supplied."
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": f"Approved context:\n{context}\n\nQuestion: {question}"},
    ]


def inference_prompt(tokenizer: PreTrainedTokenizerBase, question: str, context: str) -> str:
    """Everything the model sees BEFORE it starts answering."""
    return tokenizer.apply_chat_template(
        _messages(question, context), tokenize=False, add_generation_prompt=True
    )


def training_text(tokenizer: PreTrainedTokenizerBase, example) -> str:
    """The same prompt + the target answer (+ end-of-turn token)."""
    messages = _messages(example.question, example.context)
    messages.append({"role": "assistant", "content": example.answer})
    return tokenizer.apply_chat_template(messages, tokenize=False)