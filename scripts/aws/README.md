# AWS frozen dataset deployment helpers

Run these scripts from the repository root with Python 3.11+. They use the
standard library; downloading also requires AWS CLI v2 and an instance role
with S3 read access. No access keys belong in these files.

## Verify an existing download

```bash
python3 scripts/aws/verify_frozen_dataset.py \
  --processed "$HOME/data298-deployment/data/processed"
```

The manifest hash is pinned to the reviewed `amazon2023-skincare-v1` snapshot.
Every manifest-listed file must match both SHA-256 and byte count. This checks
integrity, not the warehouse's SQL/data quality checks. Extra files are ignored.
The recorded EC2 run on 2026-11-24 passed all 22 files.

## Download on a fresh machine

```bash
python3 scripts/aws/download_frozen_dataset.py \
  --aws-cli /usr/local/bin/aws \
  --processed "$HOME/data298-deployment/data/processed"
```

The default source is the team's frozen S3 prefix in `us-east-2`. A download
is staged beside the destination, verified, then renamed into place. Existing
destinations are refused. Failed staging downloads are cleaned up. Run only
one writer per destination. This performs S3 reads and local writes; it does
not create cloud resources, upload data, or configure credentials.

Our current EC2 download already passed, so only use the verifier there.

## Build the existing warehouse

The loader already lives in `src/warehouse/load.py`; do not duplicate it.
The EC2 environment has Python 3.11.13 and DuckDB 1.4.3 installed. From the
EC2 checkout, run:

```bash
cd "$HOME/data298-deployment/repo"
"$HOME/data298-deployment/warehouse-env/bin/python" -m src.warehouse.load \
  --processed "$HOME/data298-deployment/data/processed" \
  --manifest docs/data_run/manifest.json \
  --output "$HOME/data298-deployment/warehouse/aws_run_001"
```

Always select a new output directory for retries. See
[the warehouse README](../../src/warehouse/README.md) for the report and
publication contract. Do not upload a failed build.

## Upload and verify a passing output bundle

After the loader prints `PASS: 42 quality checks`, use this in Session Manager.
Change both run paths for subsequent builds; never reuse a published run prefix.
The three objects are uploaded separately, not as an atomic S3 transaction.
A failed upload may leave an incomplete prefix, which must not be consumed.
Consumers must require all three files, report status `PASS`, and a database
hash matching `database_sha256` in the report.

```bash
(
set -e
run_dir="$HOME/data298-deployment/warehouse/aws_run_001"
s3_dir="s3://data298-team5-artifacts-2026/warehouse/aws_run_001"
verify_dir=$(mktemp -d)

for file in warehouse.duckdb quality_report.json quality_checks.csv
do
  /usr/local/bin/aws s3 cp \
    "$run_dir/$file" "$s3_dir/$file" \
    --region us-east-2 --only-show-errors
  /usr/local/bin/aws s3 cp \
    "$s3_dir/$file" "$verify_dir/$file" \
    --region us-east-2 --only-show-errors
  cmp "$run_dir/$file" "$verify_dir/$file"
  echo "VERIFIED: $file"
  rm "$verify_dir/$file"
done

rmdir "$verify_dir"
echo "PASS: all three warehouse outputs saved and verified in S3"
)
```

If any command fails, inspect its error before proceeding; the block does not
print the final PASS. Failed readback may leave a temporary directory for
inspection. These commands use S3 storage/requests and must remain within the
team's approved credit budget.

## Recorded deployment: 2026-11-24

The following evidence comes from the operator's Session Manager output and
EC2 screenshots. It is a deployment summary, not a newly executed CI run.

| Step | Observed result |
| --- | --- |
| Runtime | Ubuntu 24.04, Python 3.11.13, DuckDB 1.4.3 |
| EC2 instance type / region | m7i-flex.large / us-east-2 |
| Access | Session Manager connected; STS returned the intended data298-team5-ec2-role |
| Frozen data | Pinned manifest and all 22 listed dataset files verified |
| Warehouse build | `PASS: 42 quality checks` for `aws_run_001` |
| Output publication | Database and both reports uploaded to `warehouse/aws_run_001/` in the team bucket |
| Readback | Each downloaded object matched its local original using `cmp` |
| Shutdown | EC2 console showed `Stopped` after publication |

The operator's final publication output was:

```text
VERIFIED: warehouse.duckdb
VERIFIED: quality_report.json
VERIFIED: quality_checks.csv
PASS: all three warehouse outputs saved and verified in S3
```

The original quality reports and database remain in controlled S3 storage;
this PR does not include their contents or assert an independently retrieved
database hash. The new helpers package the manual workflow for reuse; the
recorded cloud run used the terminal commands, not these newly added scripts.

Related Linear work: DAT-88 (warehouse), DAT-89 (AWS deployment), and DAT-91
(artifact validation/documentation). This record does not close those issues.
Airflow deployment, an end-to-end orchestrated run, Model 1 serving, and broader
artifact validation remain separate work.

## Stop between work sessions

Once output verification passes, use EC2 > Instances > the pipeline instance >
Instance state > Stop instance. Wait for `Stopped`. Closing Session Manager
does not stop the instance. Use Stop rather than Terminate to retain EBS data.
Compute usage stops while stopped, but EBS and S3 storage remain billable.
The automatically assigned public IPv4 address can change after restarting.

References: [AWS stop/start behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html)
and [AWS CLI S3 copy](https://docs.aws.amazon.com/cli/latest/reference/s3/cp.html).

EC2 setup used code commit `e2e1c924665ca8cddacf766997fd10de4656faf4`.
These new helper files are local additions and are not present in that commit
or the existing EC2 clone until transferred or included in a later reviewed commit.

Keep datasets, databases, environments, private keys, and credentials out of Git.
Only source scripts, tests, configuration without secrets, and documentation
belong in the changeset. Credit coverage still needs billing verification;
successful installation or dataset verification does not establish free usage.

## Local tests

```bash
python3 -m unittest discover -s tests -p test_aws_dataset_scripts.py -v
```
