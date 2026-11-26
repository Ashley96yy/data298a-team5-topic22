# Paired significance tests, full-catalog test split

| Comparison (A minus B) | NDCG@10 A | NDCG@10 B | Difference | Relative | 95% CI | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|
| Model 0 vs popularity | 0.0381 | 0.0131 | +0.0250 | +190.7% | [+0.0237, +0.0263] | < 0.0001 | 0.0003 | yes |
| Model 0 vs markov_last_item | 0.0381 | 0.0274 | +0.0107 | +38.9% | [+0.0098, +0.0116] | < 0.0001 | 0.0003 | yes |
| Model 0 vs itemknn_cosine | 0.0381 | 0.0204 | +0.0177 | +86.8% | [+0.0166, +0.0189] | < 0.0001 | 0.0003 | yes |

Holm correction covers only the comparisons in this table. Committed comparisons not yet available: Model 1 vs Model 0, Model 1 vs popularity, Model 1 vs markov_last_item, Model 1 vs itemknn_cosine.
