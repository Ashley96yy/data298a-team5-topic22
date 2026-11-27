# Failure study: Model 0

Failure: true item outside the top 10 in every seed. 51,005 of 56,017 test users fail; 2,455 are hits in every seed. Random sample of 100 failures (seed 2026), one category each by the first rule that applies, in this order.

| Category | Count | % of failures (sample) | % of all failures | Example case | Definition |
|---|---|---|---|---|---|
| near_miss | 2 | 2% | 3.9% | case 87 | True item ranked 11 to 20 in at least one seed |
| unpopular_target | 34 | 34% | 39.6% | case 1 | True item outside the head of training popularity (mid or tail) |
| new_brand | 52 | 52% | 46.8% | case 3 | True item's brand not in the user's history |
| category_switch | 8 | 8% | 4.8% | case 2 | True item's category differs from the user's most recent item |
| long_gap | 0 | 0% | 1.4% | n/a | More than 180 days since the last known interaction |
| other | 4 | 4% | 3.5% | case 8 | Popular item of a familiar brand and category, bought soon after |
| total | 100 | 100% | 100% | | |

How common each condition is (not exclusive):

| Condition | Among failures | Among hits in every seed |
|---|---|---|
| near_miss | 3.9% | n/a |
| unpopular_target | 40.0% | 9.0% |
| new_brand | 82.8% | 27.3% |
| category_switch | 55.5% | 26.4% |
| long_gap | 42.1% | 16.5% |
| short_history | 56.7% | 67.9% |
