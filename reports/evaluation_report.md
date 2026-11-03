# Evaluation Report: Baseline RAG vs Hybrid RAG + LoRA

## 1. Summary

The LoRA adapter mainly taught the citation *format*: citation rejections fell
from 12/14 to 4/14. It did not improve evidence use: relevance measured before
rejection fell from 0.70 to 0.60, and the adapter served 5 wrong but
correctly-cited answers, including made-up claims on both near-domain
unsupported questions. Baseline RAG served no wrong answers, but also no
useful ones. **Recommendation: revise the hybrid design; do not deploy either
system as-is.**

## 2. Configuration

Full generated configuration: [benchmark_results.md](benchmark_results.md#configuration).

| Component | Setting |
| --- | --- |
| Policy sources | 6 Markdown files in `data/knowledge_base/`; filename = source ID |
| Chunking | One chunk per paragraph (16 chunks); `chunk_id = <doc>-<position>` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243…`, normalized |
| Vector store | Qdrant local mode (in-memory), cosine; payload keeps chunk_id, document_id, title, position, text |
| Retrieval | `TOP_K=3`, `MIN_SCORE=0.30` (top-1 score) |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` @ `12fd25f7…` |
| Prompt | SmolLM2 chat template; system instruction + "Approved context: [doc_id] text… Question: …"; identical for both systems |
| Decoding | Greedy, `MAX_NEW_TOKENS=80`, prompt limit 256 tokens (overlength raises; no truncation) |
| LoRA | `q_proj`, `v_proj`; r=8, alpha=16, dropout=0.05 |
| Training | 16 examples, 8 epochs, lr 2e-4, AdamW, batch 1, seed 42, `mps`, 13 s; loss on answer tokens only |
| Parameters | 460,800 trainable / 134,975,808 total (0.341%) |
| Loss | 1.547 → 0.506 (two runs gave identical losses, so training is reproducible) |
| Training data SHA-256 | `a842ca7e98e2f0e37877f0c6d8d60b62479b217e1e0338ef0187b9fd6e65de07` |

## 3. Metric Rubric

- **recall@3 / MRR:** supported cases only; computed once because both systems share the retriever.
- **Relevance:** share of required terms (whole-phrase, case-insensitive) in the *shown* answer; rejected or abstained = 0; unsupported cases score 1 only when abstaining.
- **Raw relevance (diagnostic):** the same score on the model's text *before* citation validation. Not a served answer.
- **Faithfulness (proxy):** answered, cites only expected sources, and no forbidden term; unsupported cases must abstain. It does **not** check meaning.
- **Style:** abstained = 1, rejected = 0; answered = mean of (≤80 words, first sentence ≤30 words, no hedge words). It does not check for a "next action".
- **Decision accuracy:** answered on supported cases / abstained on unsupported ones. It does **not** check correctness.
- **Latency:** retrieval + generation per answer; excludes model loading.

Lexical metrics miss synonyms and can reward contradictions (see 6.3), so every
conclusion below was checked against the raw answers.

## 4. Results

### Retrieval (shared)

| Metric | Score |
| --- | ---: |
| recall@3 | 1.000 |
| MRR | 1.000 |

### System comparison

| System | Relevance | Raw relevance (supported) | Faithfulness | Style | Decision acc. | Rejections | Mean latency (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rag-base | 0.143 | 0.70 | 0.143 | 0.143 | 0.143 | 12/14 | 0.63 |
| rag-lora | 0.357 | 0.60 | 0.571 | 0.714 | 0.571 | 4/14 | 0.46 |

### What the user actually sees (manual count from the per-case answers)

| Outcome | rag-base | rag-lora |
| --- | ---: | ---: |
| Correct, useful answer (supported, relevance 1.0) | 0 | 3 |
| **Answer served but wrong** | **0** | **5** |
| Rejected → "escalate" | 12 | 4 |
| Abstained | 2 | 2 |

### Per-case evidence

<!-- Paste the "Per-Case Comparison" table and the "Answers (shown vs raw)"
     section from benchmark_results.md here. -->

## 5. Threshold Decision

Top-1 cosine scores: supported cases 0.525–0.891; off-topic unsupported
(cafeteria, bicycle) 0.132–0.208; near-domain unsupported (bank chargeback
0.662, shipping compensation 0.549) **inside the supported range**.

| MIN_SCORE | Supported answered | Unsupported abstained | Correct decisions |
| ---: | ---: | ---: | ---: |
| **0.30** | 10/10 | 2/4 | **12/14** |
| 0.55 | 9/10 | 3/4 | 12/14 |
| 0.67 | 5/10 | 4/4 | 9/14 |

0.30 sits in the wide gap between off-topic and supported scores (0.09 above
the highest off-topic case, 0.22 below the weakest supported case). Raising it
never helps: catching near-domain questions costs supported answers.
**Similarity is not answerability**: near-domain cases must be handled by the
model's abstention, which neither model managed. The threshold was set
before generation and applied identically to both systems. With only 4
unsupported cases it is a hypothesis, not a calibrated value.

## 6. Failure and Fragile-Success Analysis

### 6.1 Wrong paragraph, valid citation (`eval-packed-address`, rag-lora)
Retrieval ranked the correct chunk (`shipping-1`) first, but the adapter
replied with shipping times from `shipping-0` plus `[shipping]`. It passed
citation validation and was served. Likely cause: 3 of 4 shipping training
examples answer with delivery times, and training contexts always contain a
single pre-selected paragraph, so the adapter never learned to choose among
retrieved distractors. The base model instead contradicted the policy ("Yes,
support can change…") without a citation and was rejected, which failed safely.

### 6.2 Made-up, cited claims on unsupported questions (`eval-bank-chargeback`, `eval-shipping-compensation`, rag-lora)
"The standard refund window is three business days. [refunds]" is false
(the window is 30 days; three business days is the finance review time).
Both near-domain questions passed the threshold, and the adapter answered both
with plausible, cited inventions. Only 1 of 16 training examples teaches the
fallback reply, and none involves related-but-insufficient evidence. The base
model's inventions were uncited and therefore rejected.

### 6.3 Fragile rejections (`eval-log-secrets`, `eval-refund-ticket`, rag-lora)
Raw `eval-log-secrets`: "Yes, diagnostic logs may include password,
authentication tokens…", a security-critical reversal of the policy, blocked
**only** because the citation was missing. Raw `eval-refund-ticket` says "Yes,
the customer can still receive a refund" and then that requests inside 24 hours
are outside the window; it still scores raw relevance 1.0 because both
required terms appear. As citation compliance improves, these errors would be
served, and lexical metrics would not catch them.

### 6.4 Correct content blocked by format (rag-base, 12 cases)
The base model often stated the right content (e.g. "The standard shipping time
is three to five business days.") but rarely cited it: raw relevance 0.70
vs. served relevance 0.14. Its weakness is format; the adapter fixed format but
not reasoning.

## 7. Cost and Trade-offs

- **Training:** 13 s on an Apple GPU for 128 steps. Cheap now, but it must be
  repeated whenever training data or the base model changes; the adapter
  enforces this through its SHA-256/base-model provenance check.
- **Artifacts:** about 1.8 MB of adapter weights (460,800 fp32 parameters) to
  version, validate and deploy next to a 135M base model.
- **Inference:** the adapter adds no measurable latency (0.46 s vs 0.63 s; the
  difference comes from shorter outputs). It can be merged into the base model
  for serving. Peak memory and energy were not measured.
- **Hidden cost:** more fluent, cited output makes wrong answers *harder* to
  detect, which raises the burden of manual review and evaluation.
- **Limits:** 14 cases, one seed, one base model; training and evaluation share
  policies and some scenarios (within-domain check, not generalization).

## 8. Recommendation: Revise

Keep the shared retrieval, threshold, citation validation and benchmark. Do not
deploy `rag-lora`: it serves 5/14 wrong answers, including both unsupported
cases. Do not deploy `rag-base` either: it serves nothing useful. The decision
is based on held-out behavior, not on the falling training loss.

### Prioritized next experiments
1. **Retrieval-shaped training data:** build training contexts from the
   retriever's top-3 chunks (with distractors) instead of one paragraph.
2. **Near-domain abstention examples:** new, made-up related-but-insufficient
   questions (never the evaluation ones), with the fallback reply.
3. **Rebalance and reduce memorization:** balanced topics per policy, fewer
   epochs (e.g. 3), compared against the current run.
4. **Larger base model** (e.g. SmolLM2-360M-Instruct) to separate "too small"
   from "LoRA doesn't help".
5. **Bigger held-out set and a semantic judge** reported as a separate metric,
   since lexical proxies over-credit wrong answers.

Change one thing at a time, document data changes, and keep the evaluation cases untouched.

## 9. Submission Questions

1. **Which held-out behaviors changed after loading the adapter?**
   Citation compliance (rejections 12 → 4), shorter and more direct replies
   (style 0.14 → 0.71), and lower latency from shorter outputs. Off-topic
   abstention did not change (retrieval handles it). Near-domain behavior got
   worse: uncited (rejected) inventions became cited (served) ones.
2. **Did the adapter improve evidence use, style, both, or neither?**
   Style and citation format, not evidence use: raw relevance fell 0.70 → 0.60,
   and 3 of 8 served answers drew on the wrong paragraph or wrong reasoning.
3. **Which retrieval errors limit both systems regardless of fine-tuning?**
   Ranking is perfect (recall@3 = MRR = 1.0), so no ranking errors. Two
   retrieval properties still limit both: near-domain unsupported questions score
   above any usable threshold, and the top-3 includes distractor chunks (e.g.
   `shipping-0` for the packed-address case) that the model can copy.
4. **Does the measured improvement justify adapter training and maintenance?**
   Not in its current form. Training is cheap, but the gain is format only, and
   it comes with 5 served wrong answers versus 0. Revise the training data
   before adding the adapter to a deployment.
5. **What experiment would most reduce uncertainty in that decision?**
   Retrain on retrieval-shaped contexts with distractors plus new near-domain
   abstention examples, then re-run the unchanged benchmark (ideally expanded
   with more held-out unsupported cases). That directly tests whether the
   failures come from the training data or from model capacity.