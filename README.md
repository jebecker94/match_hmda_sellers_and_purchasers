# HMDA Seller–Purchaser Matching Tools

This repository bundles the scripts that the Becker research group uses to pair loan originations with subsequent purchases in the Home Mortgage Disclosure Act (HMDA) data. The tooling focuses on building reproducible crosswalks between seller and purchaser records in both the legacy (pre-2018) and expanded (post-2018) HMDA file formats.

## Repository layout

- `scripts/HMDALoader.py` – utilities for discovering and loading the most recent HMDA Loan/Application Register (LAR) files from a configurable data directory, with support for pandas, PyArrow, and Polars engines.【F:scripts/HMDALoader.py†L19-L162】
- `scripts/matching_support_functions.py` – shared data preparation helpers that harmonize schema differences, clean key variables, split originations from purchases, and apply demographic/numeric matching rules used throughout the pipeline.【F:scripts/matching_support_functions.py†L8-L515】
- `scripts/match_hmda_sellers_purchasers_pre2018.py` – end-to-end matching routine tailored to the narrower pre-2018 HMDA schema, producing staged crosswalks between sellers and purchasers.【F:scripts/match_hmda_sellers_purchasers_pre2018.py†L18-L575】
- `scripts/match_hmda_sellers_purchasers_post2018.py` – multi-round workflow for the post-2018 expanded HMDA dataset that incrementally refines candidate matches and exports parquet crosswalks and diagnostics.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L20-L1188】

## Data requirements

1. **HMDA data staging folder** – The scripts expect a `config.py` module that defines `DATA_DIR`, pointing to a directory containing the HMDA delivery folders and a `file_list_hmda.csv` manifest with the metadata used by the loader utilities.【F:scripts/HMDALoader.py†L17-L85】
2. **Parquet LAR files** – All loaders currently assume parquet-formatted LAR extracts; CSV, DTA, or gziped files are filtered out during discovery.【F:scripts/HMDALoader.py†L60-L85】

The matching routines will create `match_data` subdirectories beneath `DATA_DIR` to store intermediate parquet outputs. Ensure the executing user has write permissions within that location.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L1168-L1188】【F:scripts/match_hmda_sellers_purchasers_pre2018.py†L577-L587】

## Python environment

The project targets Python 3.13 and depends on pandas, NumPy, PyArrow, and (optionally) Polars for the loader utilities. Install the dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow polars numpy
```

Additional libraries may be required if you extend the scripts to produce visualizations or reporting artifacts.

## Running the match workflows

1. Populate `config.py` with the absolute path to your HMDA staging directory:

   ```python
   from pathlib import Path
   DATA_DIR = Path("/path/to/hmda")
   ```

2. Optionally update the `file_suffix` used to tag parquet artifacts in the post-2018 workflow to reflect the data vintage you are processing.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L1168-L1188】
3. Execute the desired matching script. For example, to run the post-2018 round-one match:

   ```bash
   cd scripts
   python match_hmda_sellers_purchasers_post2018.py
   ```

   Uncomment the desired function calls in the `__main__` section to control which rounds produce outputs.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L1168-L1188】

4. Inspect the generated parquet or CSV crosswalks in the `match_data` directory to verify match quality. Later rounds apply progressively stricter numeric tolerances, demographic checks, and uniqueness constraints to winnow the candidate list before saving the final crosswalks.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L41-L127】【F:scripts/matching_support_functions.py†L367-L515】

## Extending the pipeline

- **Adjust tolerances:** The numeric matching helpers accept column-specific tolerance dictionaries; tuning these thresholds is the primary lever for calibrating false positives versus false negatives.【F:scripts/match_hmda_sellers_purchasers_post2018.py†L72-L114】【F:scripts/matching_support_functions.py†L516-L573】
- **Incorporate new attributes:** When additional HMDA variables become available, update `get_match_columns` and the cleaning helpers to standardize their formats before matching.【F:scripts/matching_support_functions.py†L8-L162】
- **Document manual QA:** Because no automated tests exist yet, record manual validation notes in your commit messages or PR descriptions whenever you adjust the matching heuristics.【F:AGENTS.md†L5-L9】

## Contributing

Please review `AGENTS.md` for repository-specific expectations before modifying the scripts or documentation. In particular, expand docstrings when adding new functions and update this README whenever you introduce user-facing commands or configuration knobs.【F:AGENTS.md†L1-L9】
