"""
Milestone 1: Retrieval and abstention
This script loads the knowledge base, training examples, and evaluation cases, then runs a retrieval test
on the evaluation cases to see if the retriever can find relevant sources or should abstain.
"""

from . import config
from .data import *
from .retrieval import Retriever, retrieve_or_abstain

docs = load_documents(config.KB_DIR)
examples = load_training_examples(config.TRAIN_FILE)
cases = load_evaluation_cases(config.EVAL_FILE)
validate_data(docs, examples, cases)
print(f"OK: {len(docs)} docs, {len(examples)} training, {len(cases)} eval cases")

r = Retriever()
print(f"\n{len(r.chunks)} chunks. Sample:", r.chunks[0].model_dump())

print(f"\n{'case':<28}{'abstain?':<10}{'top score':<11}{'top sources':<60}decision")
for case in cases:
    hits = r.search(case.question)
    kept = retrieve_or_abstain(r, case.question)
    print(f"{case.id:<28}{case.should_abstain!s:<10}{hits[0].score:<11.3f}"
          f"{str([h.document_id for h in hits]):<60}{'ABSTAIN' if not kept else 'answer'}")