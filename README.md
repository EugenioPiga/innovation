# innovation: compustat-patents replication + firm-level public enrichments

This repository contains a cleaner Python implementation of the `compustat-patents` replication pipeline, plus firm-year enrichments from publicly available datasets.

## Added improvements

1. Better `gvkey -> CIK` linkage:
   - exact normalized-name match
   - compact-name match
   - fuzzy match (`rapidfuzz`)
   - ticker-aware scoring via optional `gvkey,ticker` hint file
2. Link QA outputs:
   - top-N candidate table for every gvkey
   - conflict file for low-separation matches
3. Extra public datasets merged into firm-year output:
   - SEC `companyfacts` + submissions metadata + optional 10-K keyword features
   - FRED macro controls
   - BLS OEWS/OES series controls
   - Census CBP industry context (employment/payroll by NAICS bucket)
   - Optional BEA GDP-by-industry context (if API key provided)

## Install

```bash
cd C:\Users\eugen\OneDrive - UC San Diego\Documents\Playground\innovation
python -m pip install -e .
```

## Run

```bash
firm-patent-enrich \
  --data-dir ..\compustat-patents\data \
  --output-dir output \
  --cache-dir cache \
  --start-year 1997 \
  --end-year 2022 \
  --max-firms 500 \
  --sec-user-agent "Your Name your_email@domain.com"
```

### Optional linkage controls

```bash
firm-patent-enrich \
  --data-dir ..\compustat-patents\data \
  --ticker-hints-file config\gvkey_ticker_hints.template.csv \
  --manual-link-file config\gvkey_cik_manual.template.csv \
  --link-fuzzy-threshold 0.88 \
  --link-score-gap 0.02 \
  --link-top-n 7
```

### Optional external data controls

```bash
firm-patent-enrich \
  --data-dir ..\compustat-patents\data \
  --bls-series-file config\bls_oes_series.template.csv \
  --fred-api-key YOUR_FRED_KEY \
  --bls-api-key YOUR_BLS_KEY \
  --include-bea \
  --bea-api-key YOUR_BEA_KEY
```

## Output files

- `output/firm_year_enriched.csv`
- `output/patent_firm_year_panel.csv`
- `output/gvkey_cik_links.csv`
- `output/gvkey_cik_link_candidates.csv`
- `output/gvkey_cik_link_conflicts.csv`

## Notes

- `--manual-link-file` overrides auto linkage where provided.
- BLS/OES series IDs are fully user-overridable via `--bls-series-file`.
- FRED can run with `--fred-api-key` (recommended for reliability).
- BEA data requires an API key; if omitted, BEA columns are skipped.
- Census context is merged by year and a SIC->NAICS proxy bucket.
