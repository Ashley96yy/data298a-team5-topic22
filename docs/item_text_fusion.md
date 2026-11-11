# Item text embeddings and SASRec fusion

Written by: Abhinita Sanabada

This implements the item-text component in Section 2a of the proposed Model 1
specification (PR #8). It is usable independently of user-profile generation.
The exact encoder choices still need team confirmation. The code accepts model
IDs and immutable Hugging Face revisions explicitly; the Colab notebook shows
candidate defaults and records the resolved revision. No full-catalog embedding
or Model 1 research result is claimed by this change.

## Encoding

Run [Notebook 13](../notebooks/model_experiments/13_item_text_embeddings_colab.ipynb)
in Colab. Default settings encode 16 items only and do not train. The Drive API
uses the frozen processed-folder ID directly; a Drive shortcut path is not needed.
Authentication uses the signed-in account's existing access. After a successful
smoke check, full encoding produces one float32 matrix with shape
`(n_items + 1, d_text)`, row zero reserved for padding and rows indexed by item ID.

Candidate encoders and their documented representations:

| Family | Candidate model ID | Dimension | Pooling | Token limit |
|---|---|---:|---|---:|
| BLaIR | `hyp1231/blair-roberta-base` | 768 | First token (CLS), L2 normalized | 512 |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Attention-masked mean, L2 normalized | 256 |

Sources: [BLaIR model card](https://huggingface.co/hyp1231/blair-roberta-base)
and [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
These are supported candidate variants, not a claim that the team has selected them.
Different dimensions/pooling require adapting the family specification and tests.

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

PR #14's real-data entry point still constructs plain SASRec. Once that work is
merged, its model construction can use `TextSASRec` with a verified embedding
artifact and record `model_class`, `text_architecture` and artifact provenance as
the item-only entry point does. This change does not edit another open PR or
provide real profiles. The full combined experiment still needs Disha's profile
file and the team's encoder/distillation choices.

Existing Part E and demo loaders assume plain SASRec checkpoints. Do not send a
fused checkpoint to them or label it Model 0. Their model-factory integration,
final multi-seed evaluation and hosted Model 1 adapter remain separate follow-ups.

## Validation scope

`python -m unittest tests.test_item_text -v` checks input tampering, item alignment,
pooling, partial output rejection, finite vectors, frozen text, learnable fusion,
causality, padding, distillation compatibility and checkpoint restoration. Small
real-encoder smoke checks use synthetic product phrases, not Amazon benchmark
results. Full encoding and trained item-text ablations remain pending the team
choices and a deliberate Colab execution.

The November 11 verification ran 148 repository tests: 147 passed and one existing
notebook-pipeline execution test was skipped because its separate dependencies
were absent. All 13 new tests passed. Both candidate encoders also encoded 16
manifest-verified real catalog items successfully; their artifacts are marked
partial. See [encoder smoke metadata](project_notes/item_text_implementation_20261111/encoder_smoke.json)
and the [verification record](project_notes/item_text_implementation_20261111/verification.json).
The notebook was syntax-checked locally, not executed in a Colab session.
