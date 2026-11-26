# Profile audit: full_v2_qwen35_9b

**Verdict: PASS with warnings** (0 failed, 1 warnings, 39 passed). 56,017 profiles, 0 failed users.

| Group | Check | Result | Detail |
|---|---|---|---|
| Files | required files present | PASS | all 5 |
| Files | checksums.json matches every listed file | PASS | 94/94 match |
| Format | user_profiles.parquet has the agreed columns | PASS | ok |
| Format | user_idx unique in user_profiles.parquet | PASS |  |
| Format | profiles_minilm.npz loads with read_profile_file; rows match user_profiles.parquet | PASS | (56017, 384), source llm_profile |
| Format | profiles_minilm.npz has 384 dimensions | PASS | 384 |
| Format | profiles_minilm.npz vectors are unit length | PASS | norm 1.0000-1.0000 |
| Format | profiles_e5_large.npz loads with read_profile_file; rows match user_profiles.parquet | PASS | (56017, 1024), source llm_profile |
| Format | profiles_e5_large.npz has 1024 dimensions | PASS | 1024 |
| Format | profiles_e5_large.npz vectors are unit length | PASS | norm 1.0000-1.0000 |
| Training data only | frozen data files match the manifest | PASS | ok |
| Training data only | the run recorded every expected input file with the frozen hash | PASS | {'interactions.parquet': '298cab581ac9', 'sequences.parquet': '0a55f2c375c0', 'item_text.parquet': '39dbd1ac817d'} |
| Training data only | every profiled user has training rows | PASS | 0 without |
| Training data only | rebuilt input hash equals the stored input_sha256 for every profile | PASS | 56,017/56,017 match |
| Training data only | n_train_items_used equals the rebuilt count | PASS | 0 differ |
| Training data only | no input contains the user's own validation or test item | PASS | 0 users |
| Training data only | prompt template uses product metadata and the user's own ratings only | PASS | ['Title', 'Brand', 'Category', 'Price'] |
| Coverage | all 56,017 users expected | PASS | 56017 |
| Coverage | profiles + failed users = every expected user | PASS | 56,017 profiles + 0 failed = 56,017 of 56,017 expected; 0 missing, 0 unexpected |
| Coverage | no user is both profiled and failed | PASS | 0 |
| Coverage | no failed users | PASS | 0 |
| Empty / duplicate | no empty profiles | PASS | 0 |
| Empty / duplicate | no very short profiles (under 10 words) | PASS | 0 |
| Empty / duplicate | identical profile texts only for users with identical inputs | PASS | 21 users share a text; 0 texts shared across different inputs |
| Empty / duplicate | near-duplicate vectors (MiniLM cosine >= 0.98) between users with different inputs | PASS | 25 users; median nearest cosine 0.8406 |
| Text leakage | profiles do not name held-out-only brands more often than chance | PASS | own 0.47% vs baseline 0.06% |
| Vectors match texts | minilm: encoder record names the pinned model and commit | PASS | sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4 |
| Vectors match texts | minilm: re-embedded sample matches stored vectors | PASS | {'model': 'sentence-transformers/all-MiniLM-L6-v2', 'revision': '1110a243fdf4706b3f48f1d95db1a4f5529b4d41', 'sample': 100, 'min_cosine': 1.0, 'mean_cosine': 1.0 |
| Vectors match texts | e5_large: encoder record names the pinned model and commit | PASS | intfloat/multilingual-e5-large@3d7cfbdacd47 |
| Vectors match texts | e5_large: re-embedded sample matches stored vectors | PASS | {'model': 'intfloat/multilingual-e5-large', 'revision': '3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3', 'sample': 100, 'min_cosine': 1.0, 'mean_cosine': 1.0, 'self_ |
| Settings | approved prompt (v2): the stored template itself hashes to the approved hash | PASS | template hashes to a18472f1aefb1a5f, stored a18472f1aefb1a5f, approved a18472f1aefb1a5f |
| Settings | pinned teacher model and commit | PASS | Qwen/Qwen3.5-9B@c20223623576 |
| Settings | thinking off | PASS | False |
| Settings | every profile row names the approved teacher and commit | PASS | Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a |
| Hygiene | no thinking text | PASS | 0 |
| Hygiene | no reasoning or meta text | WARN | 33 |
| Hygiene | no profile over 120 words | PASS | 0 |
| Hygiene | possible sensitive terms (read these by hand) | PASS | 43 profiles, e.g. users [1197, 1307, 1575, 2422, 3698] |
| Chunks | per-part vectors equal the final vectors | PASS | {'parts': 29, 'users_in_part_files': 56017, 'mismatching_part_files': 0, 'setups': 1} |
| Chunks | all parts made with one setup | PASS | 1 |
