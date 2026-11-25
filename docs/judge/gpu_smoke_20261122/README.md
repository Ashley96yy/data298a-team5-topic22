# GPU judge smoke verification

The supplied Colab artifacts were verified against implementation commit
`7ef50d3e9f0fc9a13dda2bc248bffb22e67318bd` and the source hashes saved with
that run. The later input-snapshot and notebook checkout fixes are covered by
local regression tests, not by this historical GPU run. Its original evidence
files and recorded hashes remain unchanged.
The run started on **November 22, 2026 (America/Los_Angeles)**; its folder uses UTC:
`smoke_20261123T073402_051577Z`.

- Hardware: **NVIDIA L4**, 22.03 GiB VRAM. This run did not use an A100.
- Runtime: Python 3.13.16, PyTorch 2.14.0+cu130, CUDA 13.0, vLLM 0.34.0.
- Model: pinned Ministral-3-8B-Instruct-2512-BF16, revision `f6fae9795746f63c9be8344932f01275f3c63734`.
- Result: PASS for two synthetic cases; 20 valid samples per case (40 total).
- Scoring elapsed time: 5.804 seconds; excludes environment setup, model download and startup.
- All criterion means were independently recomputed from the saved samples.
- Input, case, config, rubric and implementation hashes match the committed records.

These are operational smoke results, not evidence of recommendation accuracy or
judge reliability. High scores on the two synthetic examples are not a benchmark.
The serving record's launch-only status is retained unchanged; successful model
readiness and two HTTP 200 scoring responses are supported by the run record and
original server log. Remote model weights were not independently rehashed.

The original log contains EngineDeadError and a semaphore warning during shutdown,
after both scoring requests succeeded and shutdown was triggered. The output records
remain internally consistent; graceful shutdown and persistent-service stability
have not been established. The original log stays with the source ZIP; its hash is
recorded in [verification.json](verification.json).

Included evidence: [scores](scores/scores.jsonl), [run metadata](scores/run_metadata.json),
[serving metadata](serving/serving_metadata.json), and [environment](serving/environment.txt).

Independent human calibration, real fixed-case evaluation, and a persistent shared
endpoint remain separate deliverables.
