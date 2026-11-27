# Failure study: Model 0 (2026-11-26)

Produced by `src/evaluation/failure_study.py` from the saved test ranks of seeds 42, 43 and 44
(nothing retrained or rescored). See `failure_study.md` for the tables and `failure_study.json`
for all values, thresholds and input hashes.

- Failures (true item outside the top 10 in every seed): 51,005 of 56,017 test users.
  Hits in every seed: 2,455.
- Random sample of 100 failures, seed 2026, one category each by the first rule that applies.
  Counts sum to 100. The sample shares are close to the shares over all 51,005 failures.
- The categories were chosen after looking at Model 0 failures (exploratory) and will be applied
  unchanged to Model 1.

The case file `cases_for_review.csv` (titles and recent histories of the 100 sampled users, with
columns for a manual check) is kept in the shared Drive, not in Git.
