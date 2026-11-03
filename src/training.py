"""
Training script for LoRA fine-tuning of a base causal LM on the approved knowledge base.
The training examples are in data/training/instruction_examples.jsonl, 
and the evaluation cases are in data/evaluation/cases.json. 
The knowledge base is in data/knowledge_base/*.json.

The training script:
1. Loads the knowledge base, training examples, and evaluation cases.
2. Validates that the three layers are consistent (no missing or overlapping sources).
3. Loads the base model and tokenizer, and attaches LoRA adapters to the attention query/value projections.
4. Tokenizes the training examples, masking the prompt so that loss is only computed on the answer tokens.
5. Trains for a fixed number of epochs, saving only the LoRA adapter weights (a few MB) and the tokenizer, not the full base model (135M parameters).
6. Saves a training summary with the hyperparameters, number of trainable parameters, epoch losses, and a SHA256 fingerprint of the training examples.
"""

import hashlib
import json
from time import perf_counter

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from . import config
from .data import load_training_examples
from .prompts import inference_prompt, training_text

LORA_TARGETS = ["q_proj", "v_proj"]   # attention query/value projections
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05


def pick_device() -> torch.device:
    """Apple GPU (MPS) if available, otherwise CPU."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def encode_example(tokenizer, example) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one example; labels = -100 on the prompt so loss is only on the answer."""
    full_text = training_text(tokenizer, example)
    prompt = inference_prompt(tokenizer, example.question, example.context)

    # 1. Tokenize. add_special_tokens=False because the chat template already
    #    inserted <|im_start|>/<|im_end|>; adding more would shift the tokens.
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

    # 2. Refuse overlong examples instead of truncating (truncation cuts the answer).
    if full_ids.shape[1] > config.MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"{example.id} has {full_ids.shape[1]} tokens > MAX_SEQUENCE_LENGTH="
            f"{config.MAX_SEQUENCE_LENGTH}; shorten it or raise the limit"
        )

    # 3. The training text must begin with EXACTLY the inference prompt tokens,
    #    otherwise the model trains on a format it will never see at inference.
    prompt_len = prompt_ids.shape[1]
    if not torch.equal(full_ids[:, :prompt_len], prompt_ids):
        raise ValueError(f"{example.id}: training text does not start with the inference prompt")

    # 4. Mask the prompt: -100 = "ignore this position in the loss".
    labels = full_ids.clone()
    labels[:, :prompt_len] = -100
    if bool((labels == -100).all()):
        raise ValueError(f"{example.id} has no answer tokens to learn from")
    return full_ids, labels


def count_parameters(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train() -> dict:
    set_seed(config.RANDOM_SEED)                          # reproducible init + dropout
    training_bytes = config.TRAIN_FILE.read_bytes()       # fingerprinted in the summary
    examples = load_training_examples(config.TRAIN_FILE)

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL, revision=config.BASE_MODEL_REVISION)
    # Tokenize + validate everything BEFORE loading model weights (fail fast)
    encoded = [encode_example(tokenizer, example) for example in examples]

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL, revision=config.BASE_MODEL_REVISION, dtype=torch.float32
    )
    model.loss_type = "ForCausalLM"   # this small checkpoint's config omits it
    model.config.use_cache = False    # the generation cache is useless during training

    # 5. Freeze the base model and attach small trainable LoRA matrices
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    model = get_peft_model(model, lora_config)

    # 6. Trainable vs total parameters
    trainable, total = count_parameters(model)
    print(f"trainable {trainable:,} / total {total:,} ({100 * trainable / total:.3f}%)")

    device = pick_device()
    model.to(device)
    # Only the LoRA parameters go to the optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.LEARNING_RATE
    )

    epoch_losses = []
    started = perf_counter()
    model.train()                     # enables LoRA dropout
    for epoch in range(config.TRAIN_EPOCHS):
        running = 0.0
        for input_ids, labels in encoded:
            # 7. One optimization step per example (batch size 1, no padding needed)
            outputs = model(input_ids=input_ids.to(device), labels=labels.to(device))
            outputs.loss.backward()   # gradients flow only into LoRA weights
            optimizer.step()
            optimizer.zero_grad()
            running += outputs.loss.item()
        epoch_losses.append(running / len(encoded))
        print(f"epoch {epoch + 1}/{config.TRAIN_EPOCHS}  loss {epoch_losses[-1]:.4f}")
    training_seconds = perf_counter() - started

    # 8. Save ONLY the adapter (a few MB), not the 135M base weights
    config.ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.ADAPTER_PATH, save_embedding_layers=False)
    tokenizer.save_pretrained(config.ADAPTER_PATH)

    # 9. Everything the project asks you to record, for the report
    summary = {
        "base_model": config.BASE_MODEL,
        "base_model_revision": config.BASE_MODEL_REVISION,
        "prompt_format": (
            "SmolLM2 chat template: system instruction + user turn "
            "'Approved context:\\n[doc_id] text\\n\\nQuestion: ...'; "
            "loss on assistant answer tokens only (prompt masked with -100)"
        ),
        "lora_target_modules": LORA_TARGETS,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
        "epochs": config.TRAIN_EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "random_seed": config.RANDOM_SEED,
        "training_examples": len(examples),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": round(100 * trainable / total, 4),
        "adapter_path": str(config.ADAPTER_PATH.relative_to(config.ROOT)),
        "epoch_losses": [round(loss, 4) for loss in epoch_losses],
        "training_seconds": round(training_seconds, 1),
        "device": str(device),
        "training_sha256": hashlib.sha256(training_bytes).hexdigest(),
    }
    (config.ADAPTER_PATH / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))