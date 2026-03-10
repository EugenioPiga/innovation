# Descriptive Package Generator

Generate referee-focused descriptive tables/figures for the social-returns-to-data paper:

```powershell
python scripts/descriptives/generate_descriptive_package.py
```

Generate the cross-industry package (NAICS 2/3/4-digit, mirroring country descriptives):

```powershell
python scripts/descriptives/generate_industry_descriptive_package.py
```

Outputs are written to:
- `outputs/descriptives/tables/main/`
- `outputs/descriptives/tables/appendix/`
- `outputs/descriptives/figures/main/`
- `outputs/descriptives/figures/appendix/`
- `outputs/descriptives/memo/descriptive_memo.md`
- `outputs/descriptives/index/descriptive_index.csv`

Industry outputs are written to:
- `outputs/descriptives_industry/data/`
- `outputs/descriptives_industry/tables/main/`
- `outputs/descriptives_industry/tables/appendix/`
- `outputs/descriptives_industry/figures/main/`
- `outputs/descriptives_industry/figures/appendix/`
- `outputs/descriptives_industry/memo/industry_descriptive_memo.md`
- `outputs/descriptives_industry/index/industry_descriptive_index.csv`
