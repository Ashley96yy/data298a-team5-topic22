# Item text embeddings and SASRec fusion

Written by: Abhinita Sanabada

Status updated November 26, 2026. This implements the item-text component in
Section 2a of the merged [Model 1 specification](model1_spec.md) (PR #8).
It is usable independently of user-profile generation. The notebook pins the
source commit, encoder revisions and frozen manifest instead of following a
temporary PR branch or resolving the latest model revision at runtime.

The full MiniLM catalog encoding is already recorded in the merged specification
and used by the [product retrieval index](project_notes/README_product_search_index.md)
(PR #24). That separate encoding run does not establish a trained `TextSASRec`
result. BLaIR full encoding and the trained item-only ablations remain pending;
this PR supplies their implementation and smoke checks, not research results.

## Encoding

Run [Notebook 13](../notebooks/model_experiments/13_item_text_embeddings_colab.ipynb)
in Colab. Default settings encode 16 items only and do not train. The Drive API
uses the frozen processed-folder ID directly; a Drive shortcut path is not needed.
Authentication uses the signed-in account's existing access. After a successful
smoke check, full encoding produces one float32 matrix with shape
`(n_items + 1, d_text)`, row zero reserved for padding and rows indexed by item ID.

Encoder families from the specification and their documented representations:

| Family | Model ID | Dimension | Pooling | Token limit |
|---|---|---:|---|---:|
| BLaIR | `hyp1231/blair-roberta-base` | 768 | First token (CLS), L2 normalized | 512 |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Attention-masked mean, L2 normalized | 256 |

Sources: [BLaIR model card](https://huggingface.co/hyp1231/blair-roberta-base)
and [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Notebook revisions are MiniLM `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
and BLaIR `88adfe2b621cb202dee9aabb19b59de8f622844c`, matching the saved
November 11 encoder smoke evidence. The manifest is pinned to
`e4b7801c0d9ea27d0b39a6af253ac154d6f4a7181d8a7c1dc31d651a050af0e5`.
Different dimensions/pooling require adapting the family specification and tests.
The encoder and artifact loader enforce these exact model IDs, revisions,
dimensions, pooling methods and token limits. A different 40-character model
revision is not accepted merely because it is immutable. An approved protocol
change must update `ENCODERS` and its tests explicitly.

The source clone does not require the feature branch to survive. It checks out
the exact `CODE_COMMIT` and rejects an existing checkout at another commit or
with tracked edits. Use a fresh `PROJECT` directory when changing source pins.
Preserve the pinned source commit in repository history when merging this PR.

`src.data.item_embeddings` verifies `dataset_info.json` and `item_text.parquet`
against `manifest.json`, rejects missing/duplicate item IDs and empty text, and
encodes only the prepared `item_text` column. It does not concatenate rating,
whole-period metadata, review snippets or profile fields. Metadata-only text
can also be used when cold items are removed from training, without leaking
held-out interaction-derived snippets. The catalog's text remains available;
pretraining exposure to Amazon products remains a research limitation.

Outputs in a new directory are `item_text_emb_blair.npy` or
`item_text_emb_minilm.npy`, plus `metadata.json` with input/output hashes, encoder
revision, pooling, token limit, dimension, runtime versions and encoding duration.
Partial smoke artifacts cannot be loaded for training. Existing output directories
are never overwritten. Model downloads use the ignored `results/encoder_cache/`.
Source JSON/Parquet and embedding arrays are hashed and parsed from the same
in-memory byte snapshots, so replacing a file between verification and parsing
cannot silently change the data returned with the recorded checksum.

## Fusion and training

`TextSASRec` inherits the existing SASRec attention, hidden-state and prediction
methods. Its item representation is:

`ID embedding + bias-free Linear(text dimension, hidden size)(frozen text vector)`.

The fused representation is used for both input tokens and output catalog scores.
Text vectors are frozen buffers. The projection and ID embeddings are learned.
The projection starts at zero so initial scores match Model 0 for the same seed;
this is an implementation choice to review, not a hyperparameter selected on test
results. The padding row is forced to zero in the tied output weights so training
cannot update it through the output softmax. Model 0 code is unchanged.

Install `requirements-item-text.txt` for encoding. For a complete embedding
artifact, run the item-only ablation with the frozen Model 0 winner as a controlled
starting point:

```sh
python -m src.training.train_item_text --config configs/config.json --grid-index 7 \
  --seed 42 --embeddings results/item_text/minilm_full --device cuda \
  --output-dir results/model1_items/minilm_seed42_smoke --smoke
```

Remove `--smoke` and choose a new output directory for full training. Smoke uses
128 users and two epochs; full mode uses all users and the configured training
settings. Selection calls frozen full-catalog validation NDCG@10; test metrics
are never evaluated here. The complete text buffer, projection and model weights
are saved by the shared trainer. Load trusted checkpoints with
`src.models.text_sasrec.load_text_checkpoint`; no encoder download is needed for
inference after loading a checkpoint.

## Integration with the rest of Model 1

The fused class exposes the same `hidden`, `training_logits` and
`next_item_scores` interfaces as SASRec. A synthetic test exercises Ruchika's
existing distillation hook through the real trainer, including stage-2-only
checkpoint selection and an exact checkpoint prediction round trip.

PR #14 is merged; its real-data entry point, `src.training.train_model1`, still
constructs plain SASRec. Section 9 of the merged specification assigns the
combined arm to Ruchika as a follow-up after this PR: add an optional
`--item-text-embeddings` argument, share the input-verification logic with the
item-only entry point, construct `TextSASRec`, and record `model_class`,
`text_architecture` and artifact provenance. The combined arm is not implemented
by this PR. Full profiles are submitted in PR #29, pending review; no real-profile
training or full Model 1 result is claimed here.

Existing Part E and demo loaders assume plain SASRec checkpoints. Do not send a
fused checkpoint to them or label it Model 0. Their model-factory integration,
final multi-seed evaluation and hosted Model 1 adapter remain separate follow-ups.

## Validation scope

Ashley’s November 26 review identified missing encoder-configuration checks and
a race between file hashing and loading. Both are fixed. Regression cases for
unapproved/missing metadata, incorrect actual dimensions, replaced embedding
arrays and replaced Parquet files fail against the pre-fix loader and pass with
the snapshot-based loader. The notebook source pin includes these fixes.
After the fixes, all 21 item-text/notebook tests passed; the repository suite ran
261 tests, with 260 passing and the same optional notebook-execution test skipped.
See the [review-fix verification](project_notes/item_text_readiness_20261126/review_fixes.json).

The November 26 refresh was tested against current `main`: 32 targeted tests
passed, including four tests that execute the notebook's source-checkout cell
against local Git repositories. The full suite ran 257 tests: 256 passed and
one existing notebook-pipeline execution test was skipped because its optional
dependencies were unavailable. The real Model 0 checkpoint UI test passed.
See the [current verification record](project_notes/item_text_readiness_20261126/verification.json).
No GPU encoding, research training or full Colab session was run for this refresh.

`python -m unittest tests.test_item_text -v` checks input tampering, item alignment,
pooling, partial output rejection, finite vectors, frozen text, learnable fusion,
causality, padding, distillation compatibility and checkpoint restoration. Small
real-encoder smoke checks are not Amazon benchmark results. BLaIR full encoding
and trained item-text ablations still need deliberate GPU runs. The separately
completed MiniLM encoding is described above.

The November 11 verification ran 148 repository tests: 147 passed and one existing
notebook-pipeline execution test was skipped because its separate dependencies
were absent. All 13 new tests passed. Both candidate encoders also encoded 16
manifest-verified real catalog items successfully; their artifacts are marked
partial. See [encoder smoke metadata](project_notes/item_text_implementation_20261111/encoder_smoke.json)
and the [verification record](project_notes/item_text_implementation_20261111/verification.json).
The notebook was syntax-checked locally, not executed in a Colab session.
