# Non-Qwen judge for generated text

This implements the mean-of-20-samples option in `docs/metrics_commitment.md`:
faithfulness, relevance, coherence and fluency, each scored from 1 to 5. It is a
G-Eval adaptation using Ministral, not a reproduction of the original GPT-4 judge.
The structured-response client does not extract criterion-token probabilities;
it therefore uses the committed sampling fallback. It does not select ranking
models or replace retrieval/ranking metrics.

## Frozen configuration

`configs/judge/ministral.json` pins
`mistralai/Ministral-3-8B-Instruct-2512-BF16` at revision
`f6fae9795746f63c9be8344932f01275f3c63734`. The rubric and fixed evaluation steps are
in `configs/judge/rubric.txt`. The client rejects uncommitted changes to the config,
rubric or runner before scoring. Changes require a new version committed before
scoring new outputs; do not tune the rubric after inspecting final scores.

The client checks the served model alias, not the remote model weights. The
included launcher pins both model and tokenizer revisions and records its command,
GPU, versions and package freeze. Retain that serving record with any scores.

## Serving and smoke run

Use the Colab notebook `notebooks/model_experiments/15_ministral_judge_colab.ipynb`
on an A100, or a dedicated Linux CUDA environment. The serving requirements are
separate from the recommender's pinned PyTorch environment.

```bash
python -m pip install -r src/evaluation/judge/requirements-serving.txt
python -m src.evaluation.judge.serve --record-dir results/judge_serving_attempt1
```

In a second terminal in the same environment:

```bash
python -m src.evaluation.judge.run \
  --cases tests/fixtures/judge/synthetic_cases.jsonl \
  --output results/judge_smoke_attempt1 --mode smoke
```

The server binds only to localhost:8001. It is not a public or persistent cloud
deployment. The client also accepts an HTTPS API base URL and reads an optional
key from `JUDGE_API_KEY`. A future gateway can route Qwen and judge requests using
the `model` field; one shared public endpoint has not been deployed or tested.
Do not assume that both models fit simultaneously on the same GPU.

## Input and output contract

Input is JSONL: one object per case. Required fields:

| Field | Meaning |
|---|---|
| `case_id` | Unique opaque identifier; no account identifiers |
| `task` | `explanation` or `profile` |
| `generator_family` | Generator family, distinct from Mistral; hidden from judge |
| `request` | Original request or profile-generation task |
| `evidence` | List of `{product_id, text}` objects grounding the answer |
| `answer` | Explanation or profile being evaluated |
| `recommended_product_ids` | Product-ID list; empty for profiles/no-match |
| `synthetic` | `true` only for smoke fixtures; `false` for real evaluation |

Final evaluation requires 50 explanations and 50 profiles, drawn using seed 2026
as specified in the metrics commitment. Freeze their selection and preserve the
selection record before running the judge. The client validates cohort sizes;
it cannot independently verify the sampling procedure or training-only provenance
from supplied text. Profile evidence must come from the audited training histories.

```bash
python -m src.evaluation.judge.run --cases /path/to/frozen_cases.jsonl \
  --output results/judge_final_attempt1 --mode evaluation
```

Each case receives 20 stochastic samples at temperature 0.7, averaged separately
for each criterion. Seeds are 2026 plus case position after sorting opaque IDs.
There is no combined overall score. Twenty valid complete responses are required;
invalid responses fail the run rather than silently reducing the sample count.
A failed run can retain completed case records but must not be reported as a final
result. Output directories are never reused.

`scores.jsonl` contains case IDs, input hashes, all 20 score vectors and criterion
means. `run_metadata.json` records configuration, protocol/input hashes, completion
counts and latency. Answers, evidence, authentication keys and raw server errors
are not copied to these reports. Store real inputs and outputs in controlled data
storage; only synthetic fixtures belong in Git.

## Human calibration and delivery status

Two people must score the same cases independently and without seeing judge scores.
For each criterion, report Spearman correlation with the mean human score and
agreement between raters. Under the commitment, rho <= 0 or p >= 0.05 means the
criterion is unreliable. A successful API call does not establish judge quality.

Local automated tests exercise the client and serving command without downloading
model weights. Actual model loading, GPU memory, response validity and human
calibration require the GPU run and human ratings. No real judge results or live
shared endpoint are claimed by this implementation.

References:
- https://aclanthology.org/2023.emnlp-main.153/
- https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16
- https://docs.vllm.ai/en/latest/features/structured_outputs/
