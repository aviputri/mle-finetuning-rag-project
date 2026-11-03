# Northstar Hybrid Support Assistant (RAG + LoRA) Project
Avi Putri Pertiwi

Compares baseline RAG (SmolLM2-135M-Instruct) with hybrid RAG + a locally
trained LoRA adapter on the same retriever, prompt and 14 held-out cases.

## Architecture

```text
data/knowledge_base/*.md ─► chunk (paragraphs) ─► MiniLM embeddings ─► Qdrant (local)
                                                                          │
question ─► top-3 retrieval ─► score < 0.30? ─► abstain (no model call)    │
                 │                                                        ◄┘
                 └─► shared prompt ─► base model  (rag-base)
                                  └─► base + LoRA (rag-lora)
                                         │
                       citation validation ─► answered / abstained / rejected
```

| Module | Responsibility |
| --- | --- |
| `src/config.py` | Settings from `.env` with pinned model revisions |
| `src/data.py` | Load + validate policies, training and evaluation data; chunking |
| `src/retrieval.py` | Embeddings, Qdrant index, search, threshold abstention |
| `src/prompts.py` | One prompt format shared by training and inference |
| `src/training.py` | PEFT LoRA training with prompt masking + `training_summary.json` |
| `src/generation.py` | Base/LoRA generation, adapter provenance check, citation validation |
| `src/evaluation.py` | Shared benchmark, metrics, markdown report |

## Commands

```bash
uv sync && cp .env.example .env
uv run python -m src.milestone1    # validate data, inspect retrieval scores
uv run python -m src.training      # train LoRA -> artifacts/support-lora-adapter/
uv run python -m src.milestone3    # compare both systems on 3 example questions
uv run python -m src.evaluation    # full benchmark -> reports/benchmark_results.md
uv run pytest tests -v             # 32 tests, no language model needed
```

Generated adapters and traces live in `artifacts/` (git-ignored).

## Results

| System | Rejections | Decision acc. | Raw relevance | Served wrong |
| --- | ---: | ---: | ---: | ---: |
| rag-base | 12/14 | 0.143 | 0.70 | 0 |
| rag-lora | 4/14 | 0.571 | 0.60 | 5 |

Retrieval recall@3 = MRR = 1.0. The adapter improved citation format but not
evidence use, and served cited but wrong answers on both near-domain unsupported
questions. **Recommendation: revise.** See
[reports/evaluation_report.md](reports/evaluation_report.md).