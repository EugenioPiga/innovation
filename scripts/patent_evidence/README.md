# Patent Evidence Pipeline

## Scripts
1. `01_build_patent_backbone.py`
2. `02_score_patent_evidence.py`
3. `03_aggregate_patent_evidence_country_year.py`

## Run End-to-End

```powershell
python scripts/patent_evidence/01_build_patent_backbone.py --chunksize 250000
python scripts/patent_evidence/02_score_patent_evidence.py --chunksize 30000 --train-per-class 20000 --holdout-per-class 6000
python scripts/patent_evidence/03_aggregate_patent_evidence_country_year.py
```

## Resumable Scoring (if interrupted)

`02_score_patent_evidence.py` supports chunked append scoring:

```powershell
python scripts/patent_evidence/02_score_patent_evidence.py --score-only --chunksize 30000 --chunk-start 47 --chunk-end 66 --append-output --skip-diagnostics
```

After all chunks are appended, generate diagnostics without rescoring:

```powershell
python scripts/patent_evidence/02_score_patent_evidence.py --score-only --chunksize 30000 --chunk-start 1 --chunk-end 0 --append-output
```

## Main Outputs
- Patent backbone: `datasets/patent_evidence/output/patent_backbone.csv.gz`
- Patent-level scores: `datasets/patent_evidence/output/patent_evidence_patent_level.csv.gz`
- Country-year outputs: `datasets/patent_evidence/output/country_year/`
- Memo: `datasets/patent_evidence/measurement_memo.md`
