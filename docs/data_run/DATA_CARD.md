# Data card: Amazon Reviews 2023 - Beauty_and_Personal_Care (frozen for Models 0-4)

Created: 2026-10-18T03:19:49.529511+00:00

## Source
Amazon Reviews 2023 (Hou et al., 2024), McAuley Lab, UCSD. Official 5-core ratings + product metadata
+ review text (training interactions only).

## Subset
category = **Skin Care**. Reason: Sub-category within the user and item targets with the longest average sequence: a coherent catalog gives sharper item semantics for Model 1 while keeping enough users for Models 2 and 4.

## Processing (in order)
1. Cleaning: missing values, ratings outside 1-5, timestamps outside 1996-2023, exact duplicates removed;
   repeat purchases of the same item -> earliest kept; items without a title dropped (shared item set for all models).
2. Subset applied, then iterative 5-core until stable.
3. IDs re-indexed: items 1..25,903 (0 = padding), users 0..56,016.
4. Chronological leave-one-out split, ties broken by original file order. Max sequence length 50.

## Statistics
|  | users | items | interactions | density_% | avg_seq_len | median_seq_len |
|---|---|---|---|---|---|---|
| raw 5-core (official) | 729576.0 | 207649.0 | 6624441.0 | 0.0044 | 9.08 | 7.0 |
| after cleaning | 729576.0 | 207640.0 | 6624230.0 | 0.0044 | 9.08 | 7.0 |
| subset: Skin Care | 489260.0 | 45322.0 | 1428191.0 | 0.0064 | 2.92 | 2.0 |
| final (5-core, re-indexed) | 56017.0 | 25903.0 | 501399.0 | 0.0346 | 8.95 | 6.0 |

## Evaluation protocol (identical for all models)
- Main: full-catalog ranking, items in the user's history masked. Secondary: 100 uniform sampled negatives (fixed).
- Metrics: Recall@K, HitRate@K, NDCG@K (K = 5, 10, 20), MRR, Coverage@K, category diversity.
- Model selection on validation NDCG@10; test evaluated once. Report mean over seeds 42, 43, 44.
- Popularity baseline (test, full ranking): Recall@10 = 0.0251, NDCG@10 = 0.0131.

## Analysis groups
- Users by training length: {'short': 32133, 'medium': 17276, 'long': 6608}
- Items by training popularity: head = top 20%, tail = bottom 50%.
- Cold start: 1,295 new items held out of training; 1,142 cold test cases.

## Federated clients
One client per user (56,017 clients). Median local training size
4.0, size Gini 0.4466,
median KL(client || global) over cat_l2 = 0.5048.

## Leakage safeguards (all passed)
- every user has exactly one val and one test interaction
- every user has >= 3 training interactions
- chronology: max(train_ts) <= val_ts <= test_ts for all users
- val item never in that user's training sequence
- test item never in that user's training sequence or val
- item ids within 1..N_ITEMS (0 reserved for padding)
- train + val + test == all interactions
- item popularity computed from training split only
- sampled negatives never include a history item or a target
- cold items absent from cold-variant training sequences
- cold items absent from cold-variant validation targets
- item groups use training counts only
- Review snippets for item text come only from training interactions.

## Known limitations
- Leave-one-out uses a per-user cut, so some training interactions of one user occur after another user's test time
  (standard in sequential-recommendation benchmarks; the timestamp split is an alternative).
- BLaIR was pretrained on Amazon Reviews 2023; compare against a general text encoder to rule out leakage through it.
- Reviews are a proxy for purchases; ratings are kept only as side information.
