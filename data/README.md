# Amazon Beauty 2018: Data Pipeline and Model Preparation

This repository prepares the public Amazon Beauty 2018 interaction data for a fair comparison of four recommendation-system models. All models use the same cleaned interaction table, chronological train/validation/test split, item vocabulary, and evaluation protocol.

The project uses Amazon Beauty only; movie data is not included.

## Models supported

| Model | How it uses the prepared data | Main purpose |
|---|---|---|
| Centralized SASRec | Reads all users' chronological item sequences from the central training table. | Reproduced sequential-recommendation baseline. |
| FedSASRec with secure aggregation | Treats each `client_id` as a simulated device, trains locally, and sends updates for secure aggregation. | Measures the accuracy, convergence, and communication cost of decentralized training. |
| Compressed/on-device SASRec | Uses the same sequences with a compressed SASRec model for local inference or local updates. | Measures model size, memory, latency, and local-update cost. |
| Differentially private SASRec | Uses the same sequences with user-level DP-SGD, gradient clipping, and calibrated noise. | Measures the privacy–utility trade-off using `epsilon` and `delta`. |

The prepared dataset provides the common interaction inputs. Communication, device, compression, differential-privacy, and attack metrics are produced during the model experiments rather than stored in the source data.

## Dataset source

The project uses the Amazon Review Data 2018 **All Beauty** category from the official UCSD source:

- Dataset documentation: <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html>
- Ratings-only CSV: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/All_Beauty.csv>
- Full review JSON: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz>
- Product metadata JSON: <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz>

The source-level All Beauty category contains approximately 371,345 review/rating records and 32,992 product-metadata records. The final model-ready row count is smaller because the pipeline removes invalid records, collapses repeated user–item pairs, keeps positive interactions, and filters users with too little history. Always record the exact post-cleaning counts in the generated manifest.

## Repository layout

```text
data/
├── raw/amazon_beauty_2018/       # Downloaded source files; never overwrite
├── interim/                      # Temporary normalized data
├── processed/v1/                 # Versioned model-ready tables
└── client_partitions/v1/         # Optional per-client sequence files

notebooks/
├── amazon_beauty_data_cleaning.ipynb
└── amazon_beauty_all_models_etl.ipynb

src/
├── data/                         # Reusable loading, cleaning, splitting code
├── models/                       # SASRec, federated, compressed, and DP code
└── evaluation/                  # Shared ranking evaluation and reporting

configs/                          # Dataset and experiment configuration
experiments/<model>/<run_id>/     # Checkpoints, metrics, and run metadata
logs/                             # ETL and training logs
reports/                          # EDA, comparison tables, and figures
```

Keep large raw files outside Git when required by repository limits. Commit the notebook or source code, schema, data dictionary, checksums, configuration, and a small sample. Raw files must remain unchanged so the pipeline can be reproduced.

## End-to-end ETL pipeline

```text
Extract → Validate → Clean → Transform → Split → Encode
       → Partition clients → Quality checks → Publish → Train/evaluate
```

### 1. Extract

The pipeline loads the ratings-only CSV as the common interaction source. Review JSON and product metadata are optional enrichments and are kept in separate outputs so they do not change the fair interaction-model comparison.

### 2. Validate

Before changing the data, the pipeline records:

- source URL and download date;
- file name, file size, and optional checksum;
- row and column counts;
- column names and data types;
- null and empty-string counts;
- exact duplicate counts;
- repeated user–item pair counts;
- rating and timestamp ranges.

### 3. Clean

The pipeline:

1. normalizes common Amazon field names;
2. removes missing or blank user and item identifiers;
3. converts ratings and timestamps to numeric values;
4. keeps ratings from 1 through 5 and positive timestamps;
5. removes exact duplicate rows;
6. sorts interactions by user and timestamp;
7. keeps the earliest chronological record for a repeated user–item pair;
8. removes users with fewer than five interactions for sequential modeling.

### 4. Transform

Ratings are converted into implicit positive interactions for next-item recommendation:

```text
positive_interaction = 1 when rating >= 4
```

The pipeline adds a readable UTC datetime, a fixed domain label (`beauty`), a globally unique item key, a client identifier, integer user/item indices, and each user's chronological sequence position.

### 5. Split without temporal leakage

The same per-user chronological split is used by every model:

- earlier interactions → training;
- second-to-last interaction → validation;
- last interaction → test.

Validation and test users must already appear in the training set. Future interactions must never be used to train a model. Federated experiments use the same split inside each simulated client.

## Shared model-ready schema

The common table has 12 columns. A model may use only the subset it needs, but all four models receive the same published table and split definitions.

| Column | Type | Meaning and use |
|---|---|---|
| `user_id` | string | Original Amazon reviewer identifier. Used to build user sequences. |
| `item_id` | string | Original Amazon product identifier (`asin`). |
| `rating` | float | Original 1–5 rating retained for analysis and optional explicit-feedback experiments. |
| `timestamp` | int | Original Unix interaction time used for chronological ordering. |
| `datetime` | datetime | Human-readable version of `timestamp`. |
| `domain` | string | Fixed value `beauty`; supports consistent namespacing. |
| `global_item_key` | string | Namespaced item key, such as `beauty:<item_id>`. |
| `client_id` | string | Simulated device/client key, such as `beauty:<user_id>`. Used by federated and on-device models. |
| `user_index` | int | Contiguous integer user index for model tensors. |
| `item_index` | int | Contiguous integer item index for model embeddings and ranking. |
| `split` | string | `train`, `validation`, or `test`, assigned chronologically per user. |
| `sequence_position` | int | Zero-based position within each user's sorted interaction sequence. |

The initial cleaning notebook saves a smaller five-column file: `user_id`, `item_id`, `rating`, `timestamp`, and `datetime`. The all-model ETL notebook extends this into the 12-column table needed by all four experiments.

## Optional review and metadata fields

These fields are not required for the common SASRec/FedSASRec sequence input, but they may support future content-aware or explainability analysis.

| Optional source | Example fields | Possible use |
|---|---|---|
| Review JSON | `reviewerID`, `asin`, `reviewText`, `summary`, `vote`, `style`, `verified`, `reviewTime`, `unixReviewTime`, `image` | Sentiment, explanations, verified-review analysis, or content features. |
| Product metadata JSON | `asin`, `title`, `feature`, `description`, `price`, `brand`, `categories`, `also_buy`, `also_viewed`, `salesRank`, `similar` | Item content features, cold-start analysis, and product-network features. |

Keep these enrichments in separate tables keyed by `item_id` or `user_id`. Do not join them into the common interaction table unless the experiment explicitly requires them; otherwise the models would no longer be compared on identical inputs.

## Generated outputs

Running `amazon_beauty_all_models_etl.ipynb` writes the following directory:

```text
amazon_beauty_prepared_outputs/
├── amazon_beauty_model_ready.csv
├── amazon_beauty_train.csv
├── amazon_beauty_validation.csv
├── amazon_beauty_test.csv
├── amazon_beauty_client_sequences.csv
├── amazon_beauty_metadata.csv
└── amazon_beauty_reviews_loaded.jsonl
```

The review output is created only when full review loading is enabled. The metadata output is created when metadata loading is enabled. Treat the generated directory as a versioned dataset release, for example `processed/v1/`, and save its configuration and manifest next to it.

## Data-quality checks before modeling

The pipeline should fail or report clearly when any of the following conditions is violated:

- required columns are missing;
- required fields contain null or blank values;
- ratings are outside 1–5 or timestamps are not positive;
- exact duplicate rows remain;
- duplicate user–item handling is not deterministic;
- a user's sequence is not sorted by timestamp;
- validation/test interactions occur before the user's training cutoff;
- a validation or test user has no training history;
- user/item indices are not contiguous and reproducible;
- the three split files do not reconcile with the model-ready table;
- row counts, unique-user counts, and unique-item counts do not match the manifest.

Record the checks, counts, parameters, source version, and code version for every pipeline run.

## Common evaluation protocol

Use one evaluation implementation and the same candidate-generation procedure for all four models. The initial recommendation metrics are:

- Recall@5 and Recall@10;
- NDCG@5 and NDCG@10;
- MRR@10;
- Hit Rate@K.

Keep the following fixed across models: item vocabulary, chronological split, maximum sequence length, negative-sampling or candidate-ranking procedure, ranking code, embedding dimension where practical, and random seeds. Report mean and standard deviation or confidence intervals across multiple seeds.

Add model-specific measurements:

| Model | Additional measurements |
|---|---|
| Centralized SASRec | Training time, parameter count, and centralized accuracy reference. |
| FedSASRec | Communication rounds, bytes transferred, client participation, convergence, and local epochs. |
| Compressed/on-device SASRec | Model size, compression ratio, memory use, inference latency, and local-update time. |
| DP-SASRec | `epsilon`, `delta`, clipping norm, noise multiplier, and accuracy at each privacy budget. |
| All models | Cold-start/long-tail performance, average and worst-user performance, and run-to-run variability. |

Privacy experiments may additionally report membership-inference attack AUC and whether raw user data, individual gradients, or only protected aggregates are visible to the server.

## Running the pipeline

Install the basic notebook dependencies:

```bash
pip install pandas numpy jupyter
```

Then run:

```bash
jupyter notebook amazon_beauty_all_models_etl.ipynb
```

Recommended notebook settings:

- keep full review loading disabled unless review text is needed, because the JSON file is large;
- enable metadata loading only for content or cold-start experiments;
- preserve the generated configuration and row-count manifest;
- use `amazon_beauty_model_ready.csv` as the common input to all four model-training pipelines;
- use `client_id` and the precomputed `split` column for federated and on-device simulations.

The initial cleaning-only notebook can be run with:

```bash
jupyter notebook amazon_beauty_data_cleaning.ipynb
```

## Limitations

Amazon Beauty is a public review/rating dataset, not a complete record of purchases, browsing, impressions, or recommendations shown to users. It does not contain real device hardware measurements, federated network conditions, differential-privacy labels, or attack outcomes. Those measurements must be collected during the controlled experiments. The `client_id` mapping is a simulation in which each user is treated as one client; it should not be presented as evidence from real deployed devices.

## Citation

If this dataset is used in the report, cite the Amazon Review Data 2018 source and the associated paper:

> Jianmo Ni, Jiacheng Li, and Julian McAuley, “Justifying Recommendations Using Distantly-Labeled Reviews and Fine-Grained Aspects,” Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.

## Project next steps

1. Execute the all-model ETL notebook and commit the schema, manifest, and quality-check results.
2. Produce EDA for interaction counts, ratings, sequence lengths, item popularity, and temporal coverage.
3. Freeze the common split and evaluation protocol.
4. Reproduce centralized SASRec.
5. Implement FedSASRec with secure aggregation, compressed/on-device SASRec, and DP-SASRec.
6. Run accuracy, privacy, communication, and device-efficiency comparisons.
7. Update the Workbook 1 report, presentation, GitHub artifacts, and linked Linear issues with the executed evidence.
