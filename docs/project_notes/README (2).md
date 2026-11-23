# Product search index for the prototype

Written by Disha, Nov 22, 2026. Notebook: `notebooks/model_experiments/17_item_retrieval_index.ipynb`.
Code: `src/retrieval/item_index.py`. Tests: `tests/test_item_index.py`.

## What this is

The prototype needs a way to find products that match what a shopper is asking for. If someone types
"a gentle moisturizer for dry skin under $25", the *Preference and Retrieval* agent has to come back with
real products from our catalog, not made-up ones.

This index does that. It holds every product in our Skin Care catalog (25,903 of them) as a vector,
a list of 384 numbers that captures what the product is about. A search turns the request into a vector the
same way and returns the products whose vectors are closest. On top of that you can filter by price,
category and brand, and leave out products the shopper already owns.

It runs on an ordinary CPU, so the prototype's EC2 server doesn't need a GPU, and it needs no extra libraries
beyond numpy, pandas and sentence-transformers.

## What data it uses

- **Products:** the frozen `item_text.parquet`: the same 25,903 products the whole team uses.
- **Product text:** only the catalog description of each product: title, brand, category, features
  (first 80 words), description (first 120 words) and a price band like "$10-25".
- **Vectors:** Abhinita's verified MiniLM vectors (`all-MiniLM-L6-v2`, commit `1110a243…`). We didn't
  make new ones; we reuse hers so the whole project works in one embedding space.

**No reviews, no ratings and no purchase history go into the index.** That matters because it means nothing
from the validation or test data can leak into a search result.

## Where it is

On the shared Drive: `298A/retrieval_index/item_index_minilm_v1/`

| File | What's inside |
|---|---|
| `embeddings.npy` | the 25,903 product vectors (384 numbers each) |
| `items.parquet` | the product details shown in results: id, title, brand, category, price |
| `index_meta.json` | which model and data built it, plus checksums so a damaged file is caught |
| `build_report.json` | the results of every check below |

## How to use it

```python
from src.retrieval.item_index import ItemIndex

index = ItemIndex.load('<path to 298A>/retrieval_index/item_index_minilm_v1')

# search with words
index.search('gentle moisturizer for dry skin', k=10, price_max=25, categories=['Face'])

# search with a shopper's profile vector (the MiniLM profile files are in the same space)
index.search_vector(profile_vector, k=10, exclude_items=train_items)

# the category names the filter accepts, to show the agent
index.categories(level=2)
```

Each search returns a small table: rank, product id, similarity score, title, brand, category and price.

### The filters

| Filter | Example | How it behaves |
|---|---|---|
| Price | `price_min=10, price_max=40` | Inclusive. Products with no price are left out when you set a price, unless you add `include_unpriced=True`. |
| Category | `categories=['Face']` or `['Face > Creams & Moisturizers']` | Matches any level of the category, or a path. Not case-sensitive. |
| Brand | `brands=['CeraVe', 'Olay']` | Exact brand names, not case-sensitive. |
| Leave out | `exclude_items=[...]` | Product ids to skip, such as what the shopper already bought. |
| Minimum score | `min_score=0.5` | Drops weak matches. |

Two behaviours the agents can rely on:

- **If nothing fits, the result is empty.** That's the signal for the "no suitable match" answer.
- **A misspelled category or brand is rejected, not ignored.** Asking for "Facee" raises an error that suggests
  "face". Without this, a typo would quietly return nothing and look like "no products exist".

The level-2 categories are Face (12,817 products), Body (8,214), Sunscreens & Tanning Products (2,020),
Lip Care (1,294), Eyes (1,199), Sets & Kits (247) and Maternity (34).

## How we know it works

The notebook checks everything before and after building, and it stops with a clear message if something is
wrong instead of producing a bad index. Results from the run on the real data:

| Check | Result |
|---|---|
| The product file is the frozen version, and the vectors are the ones Abhinita verified | passed |
| Re-encoding 8 products with our query model gives the same vectors as hers (including the longest product, which tests the cut-off at 256 tokens) | cosine 1.0 |
| 50 random searches with random filters give exactly the same results as a slow brute-force search | 50 / 50 |
| Every result respects its filters | 50 / 50 |
| Searching for a product's own title finds that product | 85.6% at rank 1, 98.4% in the top 10 |
| Search speed on a CPU | about 85 ms with a text query, 9 ms with a vector |

We also read the example searches by hand, and they make sense:

- "vitamin c serum for dark spots" in Face returns vitamin C serums.
- "mineral sunscreen spf 50" under $30 returns mineral SPF 50 sunscreens.
- "lip balm" under $10 returns only lip balms.
- "exfoliating body scrub" between $10 and $40 returns body scrubs.

Using pilot profiles as queries also worked. A shopper who likes Olay and Radha Beauty cleansers got Olay and
Radha cleansers back, with the products they already own left out.

Some safety behaviour is built in too:

- The notebook refuses inputs that aren't the frozen data.
- It refuses vectors made with a different model or version.
- It refuses to overwrite an index that already exists with different content (a new version needs a new
  folder name, such as `item_index_minilm_v2`).
- Loading checks the checksums, so a damaged or edited file is caught.

## Things to know

- **About a third of the products (36%) have no price.** A search with a price filter leaves them out by
  default, so the agent never recommends something it can't confirm is within budget.
- **Some products appear twice.** The catalog has 236 products that share identical text under different ids
  (for example a Radha Beauty cleanser that shows up as both 7899 and 8491). They can both land in the same
  results. The prototype may want to drop results with an identical title.
- **Long product descriptions are cut at 256 tokens** when they're turned into vectors. That's how the item
  vectors were made, and queries use the same setting, so it's consistent.
- **Scores aren't probabilities.** A score of 0.7 means "closer than 0.6", not "70% relevant". If
  `min_score` is used, it should be chosen by looking at real examples.
- **This is the prototype version for 298A.** In 298B the plan extends it to the full skincare corpus with
  chunking and adds a proper retrieval evaluation (Recall@K and a manual relevance check). The self-retrieval
  number above is only a sanity check, not that evaluation.

## Rebuilding it

Open notebook 17 in Colab (a CPU runtime is enough), choose Run all, and allow Drive access. It takes about
5 minutes. If the index already exists with the same content, the notebook keeps it and changes nothing.
