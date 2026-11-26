# Model 1 user profiles: the full run (all 56,017 users)

Written by Disha, Nov 25, 2026. Notebook: `notebooks/model_experiments/16_model1_profiles_fullrun_qwen35.ipynb`. Audit: `audit_summary_full.md` in this folder.

## What this is

Model 1 learns from a short written description of each shopper, a "profile", alongside their purchase history.
After the 200-user pilot was approved, this run wrote a profile for every user in the dataset: all 56,017 of them.

Each profile is a few sentences an LLM writes after reading the products the shopper bought and rated in their
**training** history, for example: *"This shopper favors gentle, soothing cleansers and toners, specifically liking
Dove, Aveeno, Pond's, and Dickinson's brands…"*. Every profile is then turned into a vector (a list of numbers) by two
text encoders, and those vectors are what Ruchika's Model 1 training uses.

## How it was made

Everything is exactly as in the approved pilot:

- **Model:** `Qwen/Qwen3.5-9B` at commit `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, in bf16, with thinking off.
  Greedy decoding (no randomness), at most 160 new tokens.
- **Prompt:** the approved v2 prompt (hash `a18472f1…`). For each product it shows the title, brand, category and
  price (when the catalog has one), plus the shopper's own star rating, oldest to newest, using the 50 most recent
  training products.
- **Encoders:** multilingual-E5-large (`3d7cfbda…`, the main one, 1,024 numbers per profile) and MiniLM
  (`1110a243…`, 384 numbers), both pinned to a fixed version.
- **Where:** Ashley's home RTX 4090 (24 GB). It took **25.7 hours**, about 1.5 seconds per profile: 1.2 s for
  shoppers with 3–4 products, 3.4 s for those with 10 or more.

### Safety checks built into the run

Before writing a single profile, the notebook checked that:

- the three input files are the frozen versions (their checksums match the manifest);
- **every one of the 350,510 products going into the prompts is a training purchase of that same shopper**,
  looked up in `interactions.parquet`, so no validation or test purchase can reach a prompt;
- the prompt is the approved one, and thinking really is off.

**The pilot-match check.** Then, as a test before the long run, it regenerated 8 users from the approved pilot on
the 4090 and compared them with their pilot profiles. If they had come out very differently (average similarity
below 0.5), the run would have stopped there. They matched: 2 of 8 word for word, average similarity 0.785. The
others differ only in later wording, because a different GPU rounds numbers slightly differently.

During the run, profiles were saved in 29 parts of 2,000 users, so a crash would only have lost the current part.
Each part was copied to the shared Drive as soon as it finished, so Ruchika could start before the run ended.

## Results

| | |
|---|---|
| Profiles written | **56,017 of 56,017** |
| Failed users | **0** |
| Empty profiles | 0 |
| Profiles containing thinking text | 0 |
| Length | 62 words on average (median 62, longest 104) |
| Audit | **PASS: 0 failed, 39 passed, 1 warning** (explained below) |

### What the audit checked

The audit (`audit_summary_full.md`) checked the finished files independently of the run:

- all 94 files are unchanged since the run;
- both vector files load with Ruchika's `read_profile_file`;
- **every user's input was rebuilt from scratch** from `interactions.parquet` (using its own timestamps, not
  the run's lists) and matches the stored fingerprint (`input_sha256`) for **56,017 of 56,017** users, with
  **no validation or test item in any input**;
- every user is covered;
- 100 profiles per encoder were turned into vectors again and match the stored ones exactly;
- the 29 early parts agree with the final files.

### Things we looked at closely

The automatic flags only point at profiles worth reading; they don't reject anything. We read the flagged ones:

- **"Reasoning or meta text" (33, the audit's only warning):** all 33 are a false alarm. They mention the skin-care
  brand **"Okay"** (shea butters, black soaps), which the check mistook for "Okay," at the start of a sentence.
- **Identical profiles (21 users):** every group bought and rated exactly the same products, so identical text is
  expected. For example, 6 users bought the same 3 Brickell Men's products.
- **Brands from held-out purchases (266 profiles, 0.47%):** these profiles name a brand that the check thought
  appears only in the shopper's validation or test purchases. We traced every one of them
  (`leakage_check_explained.csv`): 151 are the same brand spelled differently in training (e.g. `L'Oreal Paris` and
  `L'Oréal Paris`), 113 are brands named in the title of a product the shopper bought in training, and 2 are ordinary
  words that happen to be brands ("simple", "cocoa butter"). **None is leakage.** The audit had already proven that no
  prompt contained a held-out product; this explains why the brands still match.
- **Possible sensitive words (162 by the run's list, 43 by the audit's stricter list):** they describe products,
  not people: "Asian beauty brands", "men's face wash", "baby lotion", "black soap", "acne treatment gel". No profile
  guesses a shopper's health, age, gender or ethnicity.

### Small wording issues (under 1% of profiles, left as they are)

- **7 profiles (0.01%) stop mid-sentence** at the 160-token limit. All belong to shoppers with 11–28 products; the
  text up to the cut is accurate.
- **About 200 (0.4%) repeat a rule from the prompt**, such as *"…without specifying gender or age"*. Awkward, but
  not a guess about the person.
- **389 (0.7%) use "effective"**, mostly as in *"affordable yet effective treatments"*, which the prompt discourages.
- **139 (0.25%) use "avoid"**, usually about low-rated products (*"avoid sunscreens that received low ratings"*),
  occasionally over-generalising.
- Rates of "all … price" statements (5.6%) and low ratings not mentioned (2.5%) are the same as in the pilot.

We kept these as they are: fixing them would mean either another day of GPU time or editing texts so they no longer
match their vectors and checksums, for very little gain.

## Where the files are

On the shared Drive in `298A/model1_profiles/full_v2_qwen35_9b/` (not in Git):

| File | What's inside |
|---|---|
| `profiles_e5_large.npz` | the main profile vectors (56,017 × 1,024) in the agreed `profile_file.py` format |
| `profiles_minilm.npz` | the lighter comparison vectors (56,017 × 384) |
| `user_profiles.parquet` | the profile texts, with the teacher tag, prompt version and each user's input fingerprint |
| `profiles_meta.json` | model, commit, prompt, encoders, input hashes and library versions |
| `run_report.json`, `profile_flags.csv`, `failed_users.json` | the run's summary, flags per user, and failures (none) |
| `checksums.json` | a fingerprint of every file, so changes or damage are caught |
| `chunks/` | the 29 parts handed over during the run |

The audit's full report is in `298A/model1_profiles/audits/full_v2_qwen35_9b/`.

## Things to know

- **Profiles are generated once and frozen.** Reproducibility comes from the saved files (the fingerprints,
  pinned versions and checksums), not from regenerating text: running the model again on another GPU would give
  slightly different wording, as the pilot-match check shows.
- **No cold-start profiles yet.** The Model 1 spec says shoppers in the cold-start variant need profiles regenerated
  from their reduced histories. This run covers the standard split only.
- **The automatic flag lists are rough.** They treat the brand "Okay" as meta text and exact brand spellings as
  different brands. They are meant to point at profiles to read, and we read them.
