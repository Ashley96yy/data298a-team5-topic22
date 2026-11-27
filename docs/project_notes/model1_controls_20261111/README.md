# Preliminary Model 1 control runs (November 11, 2026)

Validation NDCG@10, seed 42, Model 0 winner configuration (grid index 7: hidden 128, dropout 0.5, learning rate 0.0005), run on a MacBook (MPS), Python 3.11, torch 2.8.0. One seed, validation split only; the test split was not scored. Stage-1 settings were fixed in advance and not tuned: alpha 0.6, beta on, 5 stage-1 epochs.

| Run | Best validation NDCG@10 | Best epoch |
|---|---|---|
| Model 0 (Part D, Colab) | 0.04558 | 7 |
| Model 0 rerun on this machine (src.training.train) | 0.04533 | 7 |
| Shuffled profiles (control: each user gets another user's vector) | 0.04545 | 11 |
| Item-text centroid profiles (no LLM, MiniLM embeddings) | 0.05387 | 9 |

These are control arms, not Model 1 results: no LLM profiles exist yet. The profile files (86 MB each) and checkpoints are not in Git. Profiles use training items only; the MiniLM item embeddings are Abhinita Sanabada's pinned encoding (all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41).
