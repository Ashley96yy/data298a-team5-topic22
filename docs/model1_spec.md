# Model 1 specification: LLM-distilled SASRec student (298A)

Status: draft for team review, Nov 7, 2026 (implementation status in section 9). Owners: distillation loss and training integration (Ruchika); item-text embeddings (Abhinita); user-profile generation (Disha). Every claim about a paper below was read from the paper itself; anything marked **[decision]** is ours, not the paper's.

## 1. What Model 1 is (and is not)

Model 1 is SASRec (Model 0, unchanged architecture) trained with an auxiliary loss that pulls its pooled sequence representation toward a frozen embedding of an LLM-written text profile of the user. At inference the student is plain SASRec: no LLM, no text encoder, same latency as Model 0.

It is **not** R2END. R2END (Han et al., Findings of ACL 2026) distills into a *text encoder* that scores items by cosine similarity, and its own paper reports that moving the same reasoning into a SASRec backbone ("R2SASRec") performed *worse* than SASRec on Beauty (NDCG@5 0.0181 vs 0.0249 in its Table 1). We therefore cite R2END as related work and as a reason for caution, not as our method.

The method we implement is the one in **Severin et al., "Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation"** (accepted at ECIR 2026; arXiv 2604.21536; public code: `sb-ai-lab/ECIR26_Pre-trained_LLMs_Meet-Sequential_Recommenders`). Model 1 is therefore a re-implementation and extension of that method on a new dataset and split, with an added item-text-embedding component. The publishable contribution of the project is the privacy-constrained study built on top (Models 2 to 4 and the LLM-attacker evaluation), not Model 1 itself.

## 2. Components

### 2a. Item-text embeddings (Abhinita)
- Encode `item_text.parquet` (25,903 items) with BLaIR and with all-MiniLM-L6-v2 (the second is the control for BLaIR's pretraining on Amazon Reviews 2023).
- Output `item_text_emb_<encoder>.npy`, shape `(n_items + 1, d_text)`, row 0 = zeros.
- Fusion into SASRec **[decision]**: item representation = ID embedding + `Linear(d_text -> hidden)(text embedding)`. Variant if time allows: initialize the ID table from the projected text embeddings.

### 2b. User profiles (Disha)
- Prompt structure follows Severin et al. (analyze history; identify preferences; separate high-rated from low-rated items; synthesize an overall characterization). Input: titles, categories, and ratings of the user's **training** items only.
- Output `profiles.parquet`: `user_idx, profile_text, teacher_model, prompt_hash`. Cache every generation.
- Paper setup for reference: Gemma-2-9B for profiles, multilingual-E5-large (1024-d) as encoder, UMAP to the model's hidden size. **[decision]** Our default reduction is PCA fit on training-user profile embeddings (deterministic, no extra dependency); UMAP is an ablation.

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

**Expectation setting.** Severin et al. report small effects: on their Beauty data SASRec NDCG@10 0.0106 to 0.0111 (+4.9%, standard deviations 0.0002 to 0.0004) and +2% to +6% relative on the other datasets. Plan for a small gain, test it properly, and report null results honestly. Thin profiles are a risk: median training length here is 4 and 37% of users have exactly 3 items.

## 5. Budget and compute
- Profile generation: 56,017 users, plus a regenerated subset for the cold variant. Run a 200-user pilot first; record tokens per second and extrapolate before committing.
- Fallback if full generation is too slow: distill on a random 25% to 50% of users and report the subset ratio as an ablation. R2END shows distillation data ratios down to 20% still beat its baseline, but that is evidence for its method, not ours.
- Embeddings and training run on Colab GPU; nothing here needs more than one GPU.

## 6. Tests (new)
`tests/test_distill.py`: padding positions excluded from pooling; `beta` carries no gradient; `alpha = 0` reproduces `L_model` exactly; stage switch turns the auxiliary term off; target shapes match hidden size; fixed seed gives identical losses.

## 7. Open decisions
| Decision | Owner | Needed by |
|---|---|---|
| Teacher LLM and where it runs | Disha | after the pilot |
| Profile text encoder | Disha, Abhinita | Nov 9 |
| PCA vs UMAP, alpha grid, E1 | Ruchika | Nov 12 |
| Trainer hook for the auxiliary loss | Ashley, Ruchika | Nov 9 |

## 8. References (verified from the papers)
- Severin et al., arXiv 2604.21536 (ECIR 2026 accepted version).
- Han et al., "Distilling LLM Reasoning into Dense Encoders: Bridging the Accuracy-Efficiency Gap in Recommendation" (R2END), Findings of ACL 2026, pp. 22513-22531.
- Hou et al., "Bridging Language and Items for Retrieval and Recommendation" (BLaIR), arXiv 2403.03952; ACL 2026 benchmark version "...: Benchmarking LLMs as Semantic Encoders".
- Kang and McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018 (arXiv 1808.09781).

## 9. Implementation status (Nov 7, 2026)
| Part | Status |
|---|---|
| Distillation module (pooling, loss, schedule, PCA targets, epoch loop, trainer adapter with per-epoch log) | Merged (commit `9e1039c`, PR #7), 19 unit tests |
| Trainer hook in `fit()`: optional `epoch_fn` and `selection_start_epoch` | Merged (commit `c368050`, PR #11, Ashley). Defaults keep Model 0 unchanged; stage-1 epochs cannot become the best checkpoint or consume patience. |
| Integration of the module with the real trainer | Merged (`tests/test_model1_integration.py`, 3 tests: stage 1 distils but never selects, stage 2 selects, a stage 1 longer than the epoch budget is rejected) |
| Reduction of profile embeddings | PCA implemented. UMAP (the paper's choice) is not implemented and stays an ablation. |
| LLM profile generation | Not started: teacher LLM and pilot pending (Disha) |
| Item-text embeddings and fusion | Not started (Abhinita) |
| Training entry point for Model 1 (loads data, SASRec, targets; records the schedule and target hashes in `metadata`) | Not started (Ruchika) |
| Ablation runner and slice analysis | Not started (Ashley, Sneha) |
