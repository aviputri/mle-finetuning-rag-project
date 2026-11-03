"""
Milestone 3: Compare RAG with and without adapter

This script compares the performance of a base model and a LoRA adapter on a set of questions
that are supported, off-topic, or near-domain unsupported.

It uses the LocalGenerator class to generate answers and validate citations, and prints the results
for each question and system.
"""


from .generation import LocalGenerator, answer_question
from .retrieval import Retriever

QUESTIONS = [
    # supported: retrieval finds shipping-1
    "Can support change a delivery address after the warehouse packed the order?",
    # off-topic: should abstain at retrieval, without calling the model
    "Where can I find next week's cafeteria menu?",
    # near-domain unsupported: passes the threshold (0.662); can the MODEL abstain?
    "How many days will my external bank take to resolve a chargeback for a duplicate payment?",
]

retriever = Retriever()
generators = {
    "rag-base": LocalGenerator(use_adapter=False),
    "rag-lora": LocalGenerator(use_adapter=True),
}

for question in QUESTIONS:
    print("\n" + "=" * 90 + f"\nQ: {question}")
    for system, generator in generators.items():
        a = answer_question(question, system, retriever, generator)
        print(f"\n[{system}] status={a.status}  top_score={a.top_score:.3f}  "
              f"latency={a.latency_seconds:.2f}s  sources={a.retrieved_sources}")
        print("  shown:", a.answer)
        if a.raw_answer and a.raw_answer != a.answer:
            print("  raw:  ", a.raw_answer)