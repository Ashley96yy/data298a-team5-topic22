# Judge pilot controls: November 27, 2026

The updated rubric and four negative controls were scored on an **NVIDIA L4**
(22.03 GiB VRAM), using the unchanged pinned Ministral BF16 configuration. This
is a synthetic control run, not the real 50-explanation/50-profile evaluation.
The UTC run folder is `pilot_controls_20261127T184514_242735Z` (10:45 Pacific).

Implementation: `3fa4046cad6718fe449c300eb6f6c3410c436da8`. The run used the
standalone upload notebook with the test-discovery and duplicate-upload fixes
applied interactively. The committed upload notebook includes those fixes and a
clear error for a runtime without `nvidia-smi`; that new error path is a usability
fix, not part of the GPU-scored protocol. The six recorded protocol/source hashes
match the implementation commit. Original artifacts are retained unchanged.

## Results

Both sets completed: **6 cases, 20 valid samples each, 120 score vectors**.
All **24 criterion means** were independently recomputed. Input hashes, case hashes,
case order/seeds, protocol hashes, summary values and correlation-report provenance
were checked. Total scoring time was 16.012 seconds, excluding startup/downloads.

| Synthetic case | Faithfulness | Relevance | Coherence | Fluency |
|---|---:|---:|---:|---:|
| Positive explanation | 4.90 | 5.00 | 5.00 | 5.00 |
| Positive profile | 4.60 | 4.00 | 4.95 | 4.95 |
| Off-topic explanation | 1.00 | 1.00 | 3.65 | 4.30 |
| Unsupported explanation | 1.00 | 1.20 | 3.80 | 4.20 |
| Off-topic profile | 1.00 | 1.00 | 4.50 | 4.30 |
| Contradictory profile | 1.20 | 1.05 | 3.40 | 4.30 |

The four negative controls received low scores on the intended criteria. This is
a qualitative observation; no numerical acceptance threshold was precommitted.
Fluency/coherence can remain high even when faithfulness/relevance are low.
The positive cases are not matched controls. These examples do not establish
real-world reliability or causal independence between scoring criteria.

Each separate set has only one or two cases per task. Pairwise correlations are
therefore correctly null with `fewer_than_3_cases`, not evidence of zero correlation.
Real-cohort correlations and independent blinded human calibration remain pending.

## Evidence and limitations

- [Original control summary](control_summary.json) retains the notebook's automatic
  `PENDING inspection` label. The independent assessment is in
  [verification.json](verification.json); the original summary was not edited.
- [Positive scores](positive/scores.jsonl) and [negative scores](negative/scores.jsonl)
  include all 20 vectors per case, with adjacent run metadata and correlation reports.
- [Serving metadata](serving/serving_metadata.json) and
  [environment](serving/environment.txt) record L4, Python 3.13.16,
  PyTorch 2.14.0+cu130, CUDA 13.0 and vLLM 0.34.0.
- Six HTTP 200 scoring responses precede the intentional shutdown. The source log
  then records forced process cleanup and EngineDeadError. This does not invalidate
  completed scores; graceful shutdown and persistent service stability are not claimed.
- The ZIP and server log stay outside Git. Their hashes are in verification.json.
  Remote model weights were not independently rehashed.
- Historical November 22 smoke evidence remains unchanged.

## Reproducing the run

The Git-backed notebook is
[`15_ministral_judge_colab.ipynb`](../../../notebooks/model_experiments/15_ministral_judge_colab.ipynb).
Use the positive fixture and the negative fixture in separate fresh output folders,
with the same pinned protocol. See the [judge instructions](../../../src/evaluation/judge/README.md).

The optional upload notebook is
[`16_ministral_pilot_controls_upload.ipynb`](../../../notebooks/model_experiments/16_ministral_pilot_controls_upload.ipynb).
It runs both sets and shuts down its server in `finally`. It needs the supplied
`judge_pilot_code.bundle` (SHA-256 recorded in package_provenance.json), which is
not tracked in Git. To build your own bundle from a repository containing the
pinned commit:

```bash
git merge-base --is-ancestor 3fa4046cad6718fe449c300eb6f6c3410c436da8 HEAD
git bundle create judge_pilot_code.bundle HEAD
python3 -c 'import hashlib; print(hashlib.sha256(open("judge_pilot_code.bundle", "rb").read()).hexdigest())'
```

Set the upload notebook's `BUNDLE_SHA256` to the hash of the bundle you just built;
keep `CODE_COMMIT` unchanged. A bundle made by another Git version can have a
different byte hash while containing the same commit. The notebook verifies both
bundle bytes and the exact checked-out commit before running the protocol checks.
Only use a bundle from the trusted repository. GPU/package downloads still need
internet. Select L4/A100; the unchanged BF16 launcher requires at least 22 GiB.

Local validation: all 24 judge tests passed. Offline bundle restoration and
mocked six-case orchestration/failure cleanup were checked. Test discovery also
passed with an unrelated `tests` package on PYTHONPATH, reproducing and fixing the
Colab import collision. GPU results support only the recorded protocol and cases.
