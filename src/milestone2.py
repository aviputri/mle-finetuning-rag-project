"""
Milestone 2: Retrieval-Augmented Generation (RAG) with PEFT Adapter
This script demonstrates how to use a PEFT adapter for RAG. 

It loads a base model and a LoRA adapter, retrieves relevant knowledge base entries for a given question, 
constructs a prompt, and generates an answer using the combined model. 

The adapter is trained to focus on the answer tokens while ignoring the prompt tokens during training, 
allowing for efficient fine-tuning on specific tasks without modifying the base model.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config
from .prompts import evidence_text, inference_prompt
from .retrieval import Retriever, retrieve_or_abstain

question = "Can support change a delivery address after the warehouse packed the order?"

tokenizer = AutoTokenizer.from_pretrained(config.ADAPTER_PATH)
base = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL, revision=config.BASE_MODEL_REVISION, dtype=torch.float32
)

USE_ADAPTER = False   # flip to compare

model = PeftModel.from_pretrained(base, str(config.ADAPTER_PATH)) if USE_ADAPTER else base
#model = PeftModel.from_pretrained(base, str(config.ADAPTER_PATH))   # base + adapter
model.eval()

hits = retrieve_or_abstain(Retriever(), question)
prompt = inference_prompt(tokenizer, question, evidence_text(hits))
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,                      # greedy: same answer every run
        pad_token_id=tokenizer.eos_token_id,
    )
new_tokens = output[0, inputs["input_ids"].shape[1]:]   # drop the echoed prompt
print("Evidence:", [h.chunk_id for h in hits])
print("Answer:", tokenizer.decode(new_tokens, skip_special_tokens=True).strip())