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

The notebook clones the repository without naming a temporary PR branch, then
checks out and verifies its exact `CODE_COMMIT`. Keep that commit reachable in
the repository history when merging. Its pin includes the single-read input
fix; the previously saved L4 smoke evidence remains tied to its original code
hashes and is not a GPU validation of this newer runner.

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

The runner reads the UTF-8 JSONL once. Both parsed cases and `input_sha256`
come from those same bytes, including original whitespace and line endings.
Later edits to the source file cannot change the provenance recorded for that run.

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

## Pilot checks before real evaluation

The rubric now explicitly defines profile relevance using training-history likes
and dislikes, including low-rated items. Keep the four criteria in one request
and examine their pairwise correlations in the pilot. This is a diagnostic choice,
not evidence that the criteria are independent. Commit the changed rubric before
scoring; the historical GPU smoke folder remains evidence for its original rubric.

Run the existing positive smoke fixtures and the new negative controls into separate,
new output directories using the same updated committed protocol. The negative file
contains four human-authored synthetic cases: an unsupported/contradictory answer
and an off-topic answer for each task. Case IDs and control labels are not sent to
the judge. The unsupported explanation should receive low faithfulness; the
contradictory profile should receive low faithfulness and relevance; the off-topic
answers should receive low relevance. Fluency and coherence need not be low.
Inspect and record the observed means rather than claiming the controls passed
merely because scoring completed. No numeric control acceptance cutoff is committed.

```bash
python -m src.evaluation.judge.run \
  --cases tests/fixtures/judge/pilot_negative_cases.jsonl \
  --output results/judge_negative_attempt1 --mode smoke
```

These controls are separate from the frozen 50 explanations and 50 profiles;
do not append synthetic examples to that cohort. The runner has no separate pilot
mode: real cases use `--mode evaluation` and its existing 50+50 requirement.
Freeze selection and get independent blinded human ratings as described above.

After a complete run, analyze the saved criterion means without another GPU call:

```bash
python -m src.evaluation.judge.analyze_pilot \
  --scores results/judge_final_attempt1/scores.jsonl \
  --output results/judge_final_attempt1/criterion_correlations.json
```

The standard-library-only analysis requires a matching adjacent PASS
`run_metadata.json`, recomputes each mean from all 20 score vectors, and reports
all six Spearman pairs separately for explanations and profiles. Each case is one
observation; the 20 samples are not treated as 20 independent cases. Ties use average
ranks. Constant criteria and cohorts with fewer than three cases produce null rho
with an explicit reason. Thus the four negative controls alone cannot establish
cross-criterion behavior. Input hashes and cohort mode are included in the report.
Output files cannot be overwritten. Keep real scores in controlled storage.

Interpret correlations alongside examples and independent human calibration.
High correlation alone does not prove prompt-induced coupling; low correlation
alone does not establish reliability. This diagnostic has no acceptance threshold
or p-values and does not replace the committed human rho/p-value rule.

In Colab, set `SMOKE_CASES` to
`tests/fixtures/judge/pilot_negative_cases.jsonl` for the negative-control run.
Leave `RUN_MODE="smoke"`. Use a fresh output folder; the notebook writes the
correlation diagnostic after scoring. For real cases, set `RUN_MODE="evaluation"`
and `REAL_CASES` to the frozen JSONL instead. Before running, use the notebook
whose `CODE_COMMIT` includes these changes and is reachable on GitHub.

## Recorded pilot-control result

The November 27 L4 run completed all six synthetic controls (120 score vectors).
The intended faithfulness/relevance scores were low on the negative examples.
See [results and verification](../../../docs/judge/pilot_controls_20261127/README.md).
This validates the recorded control run, not real-case reliability. Real-cohort
correlations and independent human calibration remain pending.
