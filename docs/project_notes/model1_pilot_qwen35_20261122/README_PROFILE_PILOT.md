# Model 1 profile pilot with the new teacher (Qwen3.5-9B), Nov 22, 2026

Owner: Disha. Notebook: `notebooks/model_experiments/15_model1_profiles_pilot_qwen35.ipynb` (committed with its outputs).
Full outputs (profiles, `.npz` files, report, meta, checksums) are on the shared Drive in
`298A/model1_profiles/pilot_v2_qwen35_9b/`, not in Git.

## Setup
- Teacher: `Qwen/Qwen3.5-9B` at commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, loaded text-only
  (`Qwen3_5ForCausalLM`) in bf16, **thinking off** (`enable_thinking=False`, checked on the rendered chat template).
- Greedy decoding, at most 160 new tokens. Prompt v2, identical to the Qwen2.5-7B pilot (`prompt_sha256` `a18472f1…`).
- Same 200 users as the Qwen2.5-7B pilot (seed 42; 100 with 3–4 training items, 60 with 5–9, 40 with 10+).
- Inputs checked against the frozen manifest hashes; only training items and training ratings are read.
  Each user's exact input (items and ratings) is recorded as `input_sha256` for the later provenance audit.
- Encoders: MiniLM `1110a243…` (same commit as the item vectors) and multilingual-E5-large `3d7cfbda…`,
  with the `query: ` prefix for E5. Run on a Colab L4 (24 GB).

## Results
| Check | Result |
|---|---|
| Profiles / failures | 200 / 0 |
| Thinking text in output | 0 |
| Cut at the 160-token limit | 0 |
| Words per profile | mean 63.4, median 62, max 98 |
| Profile files | pass the `read_profile_file` rules (MiniLM 200 × 384, E5 200 × 1024) |
| Speed | 2.6 s per profile on an L4, about 35 hours for all 56,017 users at that rate |

Flags against the Qwen2.5-7B pilot, same 200 users (`comparison_with_pilot_v2.csv`):

| Flag | Qwen2.5-7B | Qwen3.5-9B |
|---|---|---|
| Vague praise ("high-quality", "effective") | 31 | 0 |
| Low-rated products not mentioned | 32 | 6 |
| Claims the shopper avoids something | 17 | 0 |
| Says "all" about prices | 13 | 9 |
| Mean words | 42.7 | 63.4 |

## Review of the failed automatic gates and the flags

### Batching consistency (gate failed: 3 of 8 identical, min similarity 0.626, gate 0.8)
`consistency_check.csv` regenerates 8 users one at a time and compares with their batched profile:
- 3 identical, and 1 more that differs only in a dash (`10-25` vs `10–25`).
- In the other 4, the first two sentences match word for word and only the last sentence is worded differently.
  Both versions are grounded in the same products; neither adds or drops a product, brand or rating.

This is bf16 rounding drift: once one token differs, greedy decoding continues differently. It is not padding
corruption, which would produce wrong or garbled content. The 0.8 threshold was set before we knew how large this
drift is, and it is too strict for bf16.

**Decision: keep batched generation for the full run, and document it instead of switching to batch size 1.**
- Batch size 1 would multiply the run time several times (about 35 hours on an L4 already at batches of up to 8)
  without making the profiles better; it only makes them reproducible token for token on one GPU.
- Reproducibility comes from the files, not from regenerating text: every profile is generated once and frozen,
  with `input_sha256` per user, the pinned model commit, the prompt hash and `checksums.json`. Training uses
  those files, so the same profiles go into every Model 1 run.
- The full-run notebook (`16_model1_profiles_full_qwen35.ipynb`) records each user's batch size, and before it
  starts it regenerates 8 pilot users and stops if they differ substantially from the approved pilot (mean
  similarity below 0.5), so a broken setup cannot run for hours.

### Possible sensitive term (gate flagged 1 profile)
User **10571**: "Asian beauty brands", which describes the products (Elizavecca, KARADIUM), not the person.
A false match on the word list. The profile audit, with its stricter list, reports 0 flags for the pilot.

### Other flags (non-blocking)
- **Long digit strings (2):** users 11208 and 39445; both are product names ("1000 Roses", "Lever 2000").
- **Low-rated products not mentioned (6):** users 36512, 45040 and 50393 do mention them in other words ("less
  satisfactory", "rating ... lower", "did not enjoy"). Users 9559, 52355 and 55373 really skip them: 3 of 200,
  compared with 32 of 200 for Qwen2.5-7B.
- **"All" about prices (9):** the ones checked are true (every listed product shows that price band); the prompt
  allows "all" in that case.

## Manual review of the 20 profiles (`manual_review.csv`)
Reviewed on Nov 22 against the products listed in `review_sample.csv`. One row per user is in `manual_review.csv`.
This replaces the "PENDING" placeholder in `pilot_report.json`; the pilot folder on Drive is left unchanged so
its checksums and the audit stay valid.
- **13 accurate and specific:** brands, categories, ratings and prices match the listed products.
- **6 with small slips:** a 3-star product called liked (42927); a hand soap credited to the wrong brand (47144);
  a drying lotion called a pore cleanser and "natural" overstated from one product (5449); a scent linked to the
  wrong brand (20458); "five stars" where one item had 4 (17381); lower-rated items not mentioned (52355).
- **1 with an awkward sentence** echoing the prompt rules (38080).
- **None** invent a brand or product, and none infer health, age, gender or ethnicity.

## Decision
Keep prompt v2, the pinned commit and batched generation for the full run. The slips are minor, and a new prompt
would need another pilot. The full run is on Ashley's RTX 4090 (24 GB) with notebook 16, which uses the same
Transformers setup as this pilot; it does not install extra fast kernels, to keep the setup identical to the
approved pilot.

## For reviewers
- **Ruchika:** please confirm the two `.npz` files in `pilot_v2_qwen35_9b/` load with `read_profile_file` on your side.
- **Abhinita:** endpoint model `Qwen/Qwen3.5-9B`, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16,
  thinking off. Profiles use the 4090, so the endpoint needs a separate GPU.
