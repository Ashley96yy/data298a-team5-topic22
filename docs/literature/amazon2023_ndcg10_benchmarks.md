# Amazon Reviews 2023 literature benchmarks

Source review: 2026-10-26.

## Scope and verification

Four original papers explicitly using Amazon Reviews 2023 and reporting sequential-recommendation NDCG@10 were inspected. Values below are author-reported results checked against the cited tables, not independent reproductions. Scores use the 0–1 scale. Publication year is distinct from dataset year. Missing protocol details remain unknown rather than being inferred from model names or training losses.

## 1. HSTU-BLaIR (Liu, 2025)

[HSTU-BLaIR: Lightweight Contrastive Text Embedding for Generative Recommender](https://kdd2025llm4ecommerce.github.io/assets/accepted/13_HSTU_BLaIR_Lightweight_Cont.pdf). LLM4ECommerce workshop at KDD 2025. Evidence: Table 2, PDF page 3; Sections 2.4–2.5, pages 2–3.

| Amazon 2023 category | SASRec NDCG@10 | HSTU-BLaIR NDCG@10 |
|---|---:|---:|
| Video Games | 0.0573 | 0.0760 |
| Office Products | 0.0153 | 0.0271 |
| Musical Instruments | 0.0356 | 0.0406 |

Uses 5-core filtering and chronological last-item test holdout; remaining interactions train the model. A separate validation holdout and candidate-pool size are not explicit in the inspected experimental description. Training lasts 100 epochs. HSTU-BLaIR combines a sequential HSTU backbone with semantic item embeddings; the SASRec row is the ID-based comparison.

**Use:** contextual SASRec reference. Category, holdout design, and unresolved ranking details limit comparisons with other datasets and evaluation protocols. Do not label it full-catalog solely from its training loss.

## 2. How Well Does Generative Recommendation Generalize? (Ding et al., 2026)

[Original paper, arXiv:2603.19809v1](https://arxiv.org/html/2603.19809v1). Evidence: Table 4 (N@10 rows), Sections 3.1 and 5.3. The [authors' repository](https://github.com/Jamesding000/MemGen-GR) identifies the work as KDD 2026 Oral; the linked arXiv version is the extraction source.

| Amazon 2023 category | SASRec NDCG@10 | TIGER NDCG@10 |
|---|---:|---:|
| Industrial and Scientific | 0.0209 | 0.0243 |
| Musical Instruments | 0.0291 | 0.0323 |
| Office Products | 0.0190 | 0.0254 |

The last interaction is test, second-last validation. SASRec uses cross-entropy over all items during training; validation selects the checkpoint. Training is capped at 150 epochs with early stopping. Filtering thresholds, evaluation candidate count, and history masking are not explicit in the inspected setup. All-item training is not proof of full-catalog evaluation.

**Use:** contextual comparison between ID-based SASRec and semantic-ID TIGER. The paper's Beauty and Sports results use Amazon **2014** and are deliberately excluded. Table 4 reports overall results; Table 1's generalization subgroups must not be substituted.

## 3. DISR (Bajjou and Nfaoui, 2026)

[DISR: Distributional Item Representations for Sequential Recommendation](https://thesai.org/Downloads/Volume17No8/Paper_92-DISR_Distributional_Item_Representations.pdf). IJACSA 17(8), DOI: [10.14569/IJACSA.2026.0170892](https://doi.org/10.14569/IJACSA.2026.0170892). Evidence: Tables II–III, PDF page 8 (printed page 1038); Section IV, pages 5–6.

| Category / scale | SASRec NDCG@10 | DISR NDCG@10 |
|---|---:|---:|
| Beauty and Personal Care / 50K users | 0.3139 ± 0.0023 | 0.3539 ± 0.0033 |
| Sports and Outdoors / 50K users | 0.2774 ± 0.0020 | 0.3281 ± 0.0014 |
| Beauty and Personal Care / full | 0.4600 | 0.4789 |
| Sports and Outdoors / full | 0.3984 | 0.4418 |

Uses official 5-core data, chronological last/second-last test/validation, and sequence length 50. Evaluation ranks one positive against **99 cached uniform negatives**. The 50K results are mean ± standard deviation over five seeds; full-scale results use seed 42 only. DISR learns Gaussian item/query representations.

**Use:** related-domain evidence, not a full-catalog performance target. Randomly sampled Beauty users do not represent a category-filtered Skin Care subset.

## 4. MEMOIR (Bae, 2026)

[MEMOIR: Temporal Behavioral Memory for Recommendation Across the Preference-Drift Spectrum](https://arxiv.org/html/2607.23986v1). arXiv preprint 2607.23986v1. Evidence: Section 4.1 and Table 1 in Section 4.2.

| Reported scope | Model | NDCG@10 |
|---|---|---:|
| Electronics + Clothing, Shoes and Jewelry; aggregate | SASRec | 0.0543 |
| Same aggregate | UniSRec | 0.0641 |
| Same aggregate | MEMOIR | 0.0643 |

Uses users with at least five interactions, monthly behavioral windows, and chronological 80/10/10 splitting. Evaluation uses **999 negative items**, although baseline training uses full-catalog cross-entropy. MEMOIR combines LLM-derived temporal memories with a GRU and aggregation layers. Table 1 supplies aggregate scores, not separate scores for each category; it supplies no seed standard deviations.

**Use:** semantic/temporal-method context. Do not duplicate the aggregate as a category-specific result, describe its evaluation as full-catalog, or treat the small difference from UniSRec as a demonstrated superiority claim.

## Cross-paper interpretation

The reported NDCG@10 values are specific to each paper's dataset and evaluation protocol. Candidate pools, preprocessing, sequence construction, and holdouts change the prediction task. The scores should not be averaged across papers or used to rank methods across different experimental settings.

DISR uses 99 sampled negatives and MEMOIR uses 999 negatives. Their results do not establish numerical targets for full-catalog evaluation. Evaluation candidate counts remain unresolved in the inspected descriptions of HSTU-BLaIR and How Well Does Generative Recommendation Generalize?. Full-catalog training losses alone do not establish the evaluation protocol.

A controlled comparison requires matching dataset subsets, preprocessing, splits, candidate selection, history masking, and metric definitions. Seed variability and differences between aggregate and category-specific results must also be considered. Independent reproduction would require resolving missing protocol details from the authors' implementations.

## Screening and limitations

- TATK (arXiv:2609.14565v1) was screened but not included in the four-paper benchmark. Its Table 2 lists SASRec CDs NDCG@10 0.0119 above HR@10 0.0081. Under single-positive binary relevance, NDCG@10 cannot exceed HR@10. This is a table/protocol inconsistency to clarify, not evidence of misconduct. [Source](https://arxiv.org/html/2609.14565v1#S4.T2)
- HSTU-BLaIR and BLaIR are distinct works; the HSTU-BLaIR results above should not be attributed to BLaIR.
- Candidate protocols are explicitly unresolved for the first two entries. Source extraction is complete; independent reproduction and protocol matching are not.
- All four papers supply SASRec results on Amazon 2023. Their proposed-model results are labeled separately. No Amazon 2014 or Steam score is included in the benchmark tables.
