# Model 1 specification: LLM-distilled SASRec student (298A)

Status: updated Nov 24, 2026 for the Qwen3.5-9B teacher and the current repository state (implementation status in section 9). Owners: distillation loss and training integration (Ruchika); item-text embeddings (Abhinita); user-profile generation (Disha). Every claim about a paper below was read from the paper itself; anything marked **[decision]** is ours, not the paper's.

## 1. What Model 1 is (and is not)

Model 1 is SASRec (Model 0, unchanged architecture) trained with an auxiliary loss that pulls its pooled sequence representation toward a frozen embedding of an LLM-written text profile of the user. At inference the student is plain SASRec: no LLM, no text encoder, same latency as Model 0.

It is **not** R2END. R2END (Han et al., Findings of ACL 2026) distills into a *text encoder* that scores items by cosine similarity, and its own paper reports that moving the same reasoning into a SASRec backbone ("R2SASRec") performed *worse* than SASRec on Beauty (NDCG@5 0.0181 vs 0.0249 in its Table 1). We therefore cite R2END as related work and as a reason for caution, not as our method.

The method we implement is the one in **Severin et al., "Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation"** (accepted at ECIR 2026; arXiv 2604.21536; public code: `sb-ai-lab/ECIR26_Pre-trained_LLMs_Meet-Sequential_Recommenders`). Model 1 is therefore a re-implementation and extension of that method on a new dataset and split, with an added item-text-embedding component. The publishable contribution of the project is the privacy-constrained study built on top (Models 2 to 4 and the LLM-attacker evaluation), not Model 1 itself.

## 2. Components

### 2a. Item-text embeddings (Abhinita)
- Encode `item_text.parquet` (25,903 items) with BLaIR and with all-MiniLM-L6-v2 (the second is the control for BLaIR's pretraining on Amazon Reviews 2023).
- Output `item_text_emb_<encoder>.npy`, shape `(n_items + 1, d_text)`, row 0 = zeros.
- Fusion into SASRec **[decision, as implemented in `src/models/text_sasrec.py`, PR #17]**: item representation = ID embedding + a bias-free `Linear(d_text -> hidden)` of the frozen text embedding. The projection starts at zero, so a `TextSASRec` begins exactly at Model 0 for the same seed. Input and output item vectors are tied, and padding row 0 stays zero. The embedding artifact is verified against the frozen manifest before use. Variant if time allows: initialize the ID table from the projected text embeddings.

### 2b. User profiles (Disha)
- Prompt structure follows Severin et al. (analyze history; identify preferences; separate high-rated from low-rated items; synthesize an overall characterization). Input: titles, categories, and ratings of the user's **training** items only.
- Output `profiles.parquet`: `user_idx, profile_text, teacher_model, prompt_hash`. Cache every generation.
- Paper setup for reference (Section 3 and footnotes 7 and 8 of the paper): profiles from `UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3`, encoded with `intfloat/multilingual-e5-large` (1024 numbers), reduced with UMAP to the model's hidden size; the profile prompt asks the LLM to separate highly rated from poorly rated items. **[decision]** Our default reduction is PCA fit on training-user profile embeddings (deterministic, no extra dependency); UMAP, the paper's choice, is added as an option and both are compared.

### 2c. Distillation (Ruchika)
- Pooled student representation `H(S_u)` = mean over non-padding positions of the **last transformer layer's** hidden states (the paper found the last layer best; exponential recency weighting is its alternative).
- `L_distill = MSE(H(S_u), target_u)`, target frozen.
- Stage 1: `L = alpha * beta * L_distill + (1 - alpha) * L_model`, with `beta = detach(L_model / L_distill)` (per batch). Paper grid: `alpha` in {0.4, 0.6, 0.8}; beta on/off.
- Stage 2: `L_model` only.
- The paper spends about half the epochs in stage 1. **[decision]** We fix stage 1 to `E1` epochs (a grid value) and apply early stopping and checkpoint selection (validation NDCG@10) in stage 2 only; early stopping during stage 1 is meaningless because the loss changes at the switch.
- Interface: uses `SASRec.hidden(tokens)`. Implemented in `src/models/distill.py` (tests in `tests/test_distill.py` and `tests/test_model1_integration.py`), merged. The trainer hook (`epoch_fn`, `selection_start_epoch`) is merged too; see `docs/model1_training_hooks.md`.

## 3. Leakage rules (add as check 14 onward)
1. Profiles use `train_items` only. Validation and test items never appear in a prompt. No profile is needed for validation or test: distillation happens in training only.
2. PCA/UMAP is fit on training users only.
3. Cold-start variant: users whose training sequences changed when the cold items were removed need their profiles **regenerated** from the reduced sequences. Reusing the original profiles would leak the cold items' semantics.
4. Do not use whole-period metadata (`average_rating`, `rating_number`, `details`) anywhere.
5. LLMs may have seen Amazon products in pretraining. We cannot remove this; report it as a limitation.
6. Test is evaluated once per seed, after configuration is frozen.

## 4. Evaluation and ablation ladder
M0 SASRec; +BLaIR item text; +MiniLM item text; +profile distillation only; full Model 1 (item text + distillation). Report Recall@5/10/20, NDCG@5/10/20, MRR (frozen `rec_eval.py`), seeds 42/43/44, mean and std, paired bootstrap over users. Slices: user history group, head/mid/tail target items, and the cold-start variant (a separate training run).

Reference points measured on this dataset (test, full ranking): popularity NDCG@10 0.0131; last-item Markov 0.0274; cosine item-kNN 0.0204. Markov scores 0.0405 on head targets but 0.0072 (mid) and 0.0042 (tail), and 21,000 of 56,017 test cases are mid or tail: that is where text signal has room.

Model 0 (SASRec-style, grid winner hidden 128, dropout 0.5, learning rate 0.0005; three seeds, `docs/project_notes/model0_test_20261108/`, commit `c203d42`): test NDCG@10 0.0381 ± 0.0003, Recall@10 0.0648 ± 0.0006, MRR 0.0350 ± 0.0002 (validation NDCG@10 0.0454 ± 0.0002). On the 1,142 cold-start test cases Model 0 reaches NDCG@10 0.0000 and a mean relative rank of 0.940, worse than chance (0.5) and close to popularity (0.974): a model that only knows item ids cannot place a product it never saw, which is the gap that item text and profile distillation target.

**Expectation setting.** Severin et al. report small effects: on their Beauty data SASRec NDCG@10 0.0106 to 0.0111 (+4.9%, standard deviations 0.0002 to 0.0004) and +2% to +6% relative on the other datasets. Plan for a small gain, test it properly, and report null results honestly. Thin profiles are a risk: median training length here is 4 and 37% of users have exactly 3 items.

## 5. Budget and compute
- Profile generation: 56,017 users, plus a regenerated subset for the cold variant. Run a 200-user pilot first; record tokens per second and extrapolate before committing.
- Fallback if full generation is too slow: distill on a random 25% to 50% of users and report the subset ratio as an ablation. R2END shows distillation data ratios down to 20% still beat its baseline, but that is evidence for its method, not ours.
- Profile generation and training run on the college-assigned GPU (RTX 4090 or 5090, as reported by the team) and on Colab; nothing here needs more than one GPU. Pilot measurement (Qwen3.5-9B, bf16, thinking off, NVIDIA L4): 2.596 s per profile, about 35 hours for all 56,017 users on an L4; the college GPU time is to be measured.

## 6. Tests (new)
`tests/test_distill.py`: padding positions excluded from pooling; `beta` carries no gradient; `alpha = 0` reproduces `L_model` exactly; stage switch turns the auxiliary term off; target shapes match hidden size; fixed seed gives identical losses.

## 7. Decisions (updated Nov 11, 2026)
| Decision | Owner | Status |
|---|---|---|
| Teacher LLM | Disha, Abhinita | **Changed Nov 22-23:** `Qwen/Qwen3.5-9B`, commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (Apache 2.0), bf16, thinking off (`enable_thinking=False`), greedy decoding, approved prompt v2 (hash `a18472f1aefb1a5f`). Replaces `Qwen/Qwen2.5-7B-Instruct` (decided Nov 11) because the professor asked for the latest models. The paper used `UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3` (gated). Pilot: 200 users, 0 failures, audit PASS (37 checks), 2.596 s per profile on an L4. Qwen3.6 (Apr 2026) exists, but its open sizes found so far are 27B and 35B-A3B, which need a larger GPU; whether Qwen3.5-9B counts as "latest" is a question for the professor. |
| Profile text encoder | Disha, Abhinita | **Decided Nov 11:** main encoder `intfloat/multilingual-e5-large` (1024 numbers, as in the paper; input text prefixed with `query: `, mean pooling, L2-normalised), with `sentence-transformers/all-MiniLM-L6-v2` (384 numbers) as the light comparison and for the pilot. Profile text is generated once, independently of the encoder, and embedded with both. Revisions are pinned. |
| Item-text encoders | Abhinita | Done: MiniLM (`all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) encoded for all 25,903 items and verified; BLaIR `hyp1231/blair-roberta-base` next, with its commit pinned. |
| PCA vs UMAP, alpha grid, E1 | Ruchika | PCA is implemented; UMAP (the paper's choice) is added as an option and compared with it. First control run (no LLM; item-text centroids as profiles) fixed in advance, not tuned: alpha 0.6, beta on, E1 = 5, Model 0 winner settings, seed 42. Grid for the real runs afterwards: alpha in {0.4, 0.6, 0.8}, beta on/off, E1 values chosen from the control run. |
| Trainer hook for the auxiliary loss | Ashley, Ruchika | Done (merged, commit `c368050`). |

Differences from the paper, to be reported: teacher LLM (Qwen3.5-9B instead of Gemma-2-9B-It-SPPO-Iter3); dimensionality reduction (PCA, with UMAP as a comparison); data and split (Amazon Reviews 2023 Skin Care with a per-user leave-one-out split, instead of 2015 Beauty and other datasets with a global temporal split); 3 seeds instead of 5 (5 for the final configurations if time allows). Kept as in the paper: SASRec with a full-catalog cross-entropy loss, mean pooling of the last layer, the two-stage loss with the detached scaling factor, alpha in {0.4, 0.6, 0.8}, about half of the epochs in stage 1, and a profile prompt that includes the user's own star ratings of their training items.

## 8. References (verified from the papers)
- Severin et al., arXiv 2604.21536 (ECIR 2026 accepted version).
- Han et al., "Distilling LLM Reasoning into Dense Encoders: Bridging the Accuracy-Efficiency Gap in Recommendation" (R2END), Findings of ACL 2026, pp. 22513-22531.
- Hou et al., "Bridging Language and Items for Retrieval and Recommendation" (BLaIR), arXiv 2403.03952; ACL 2026 benchmark version "...: Benchmarking LLMs as Semantic Encoders".
- Kang and McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018 (arXiv 1808.09781).

## 9. Implementation status (Nov 23, 2026)
| Part | Status |
|---|---|
| Distillation module (pooling, loss, schedule, PCA targets, epoch loop, trainer adapter with per-epoch log) | Merged (commit `9e1039c`, PR #7), 19 unit tests |
| Trainer hook in `fit()`: optional `epoch_fn` and `selection_start_epoch` | Merged (commit `c368050`, PR #11, Ashley). Defaults keep Model 0 unchanged; stage-1 epochs cannot become the best checkpoint or consume patience. |
| Integration of the module with the real trainer | Merged (`tests/test_model1_integration.py`, 3 tests: stage 1 distils but never selects, stage 2 selects, a stage 1 longer than the epoch budget is rejected) |
| Reduction of profile embeddings | PCA implemented. UMAP (the paper's choice) is to be added as an option (umap-learn) and compared. |
| LLM profile generation | Pilot done with Qwen3.5-9B (200 users, 0 failures). The audit notebook and summary are merged (PR #25, 37/37 checks, including the checks added after review: every expected input hash, the recomputed prompt-template hash, the teacher tag on every row, both encoder records). The pilot notebook is PR #23 (open; the batching-consistency check and one sensitive-term flag are still to be settled before the full run). Full run on all 56,017 users pending (Disha, college GPU); the same audit is run on it before any training. |
| Item-text embeddings and fusion | Item-only arm implemented in PR #17 (open draft, builds on #16): `TextSASRec`, `train_item_text.py`, `load_text_checkpoint`. Combining it with distillation (the full Model 1) is not yet in `train_model1.py` (Ruchika). Plan: after #17 merges, add `--item-text-embeddings <dir>` to `train_model1.py`, which builds `TextSASRec` instead of `SASRec`, verifies the embedding artifact exactly as `train_item_text.py` does (shared helper), and records `model_class`, `text_architecture` and the embedding artifact in the run metadata. `distill.py` and the trainer already use `model.item_embedding.weight`, which `TextSASRec` provides. Checkpoints of this arm are loaded with `load_text_checkpoint`, not the generic loader in `checkpoints.py`, which expects the key `item_embedding.weight`. |
| Training entry point for Model 1 (loads data, SASRec, targets; records the schedule and target hashes in `metadata`) and the profile file format | Merged (PR #14, commit `0ec1a67`): `src/training/train_model1.py`, `src/data/profile_file.py`. Tested end to end on synthetic data only, including failure and resume; no real profiles and no real Model 1 run yet. (Ruchika) |
| Ablation runner and slice analysis | Not started (Ashley, Sneha) |
| Control runs and records | `docs/project_notes/model1_controls_20261111/` and the centroid and shuffled control profiles: PR #18, open. |
| Partial-coverage fix (distillation weight scaled by the share of covered users per batch) | Merged (PR #19). |
