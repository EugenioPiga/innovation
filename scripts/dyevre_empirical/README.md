# Dyèvre-Style Variable Package

Builds a Dyèvre (2024)-inspired variable set, adapted to the current repository and linked to the empirical/data-driven patent score.

Run:

```powershell
python scripts/dyevre_empirical/build_dyevre_empirical_package.py
```

Main data outputs:
- `datasets/dyevre_empirical/output/patent_dyevre_style_variables.csv.gz`
- `datasets/dyevre_empirical/output/country_year_dyevre_style_variables.csv`
- `datasets/dyevre_empirical/output/firm_year_dyevre_style_variables_with_d5.csv`
- `datasets/dyevre_empirical/output/paper_variable_crosswalk.csv`

Descriptive package outputs:
- `outputs/dyevre_empirical/tables/main/`
- `outputs/dyevre_empirical/tables/appendix/`
- `outputs/dyevre_empirical/figures/main/`
- `outputs/dyevre_empirical/figures/appendix/`
- `outputs/dyevre_empirical/memo/dyevre_empirical_memo.md`
- `outputs/dyevre_empirical/index/dyevre_empirical_index.csv`
