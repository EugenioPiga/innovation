# Dyevre-Style Variable and Graph Package Memo

## Objective
Construct as many variables as feasible from Dyevre (2024) using repository data plus PatentsView tables, and connect them to your empirical/data-driven patent measure.

## Core outputs
- Patent-level file: `datasets/dyevre_empirical/output/patent_dyevre_style_variables.csv.gz`
- Country-year file: `datasets/dyevre_empirical/output/country_year_dyevre_style_variables.csv`
- Firm-year file: `datasets/dyevre_empirical/output/firm_year_dyevre_style_variables_with_d5.csv`
- Variable crosswalk: `datasets/dyevre_empirical/output/paper_variable_crosswalk.csv`

## Coverage snapshot
- Patent observations: 3644430
- Country-year observations: 1847
- Firm-year observations: 42614
- Public-funded share (patents): 0.008
- Empirical-driven share (patents): 0.210

## What is exact vs approximate
- Exact from source tables: public-funding indicator, forward/backward patent citations, agency tags.
- Approximations: science-reference share (text heuristics), independent-claims proxy (total claims), class-opening/ahead-of-time (CPC analogue), small-firm citation share (linked Compustat subset).
- Not available from current data: inventor wage bill, full market value panel, historical SSIV shocks, examiner-leniency IV construction.

## Why this is useful for your empirical-driven question
The package enables direct descriptive links between empirical/data-driven innovation and:
1. public funding exposure,
2. science linkage,
3. patent influence and breadth of spillovers,
4. cross-country and agency heterogeneity,
with transparent coverage diagnostics.
