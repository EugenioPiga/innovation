# Industry Descriptive Memo

## Scope
This package replicates the country-style descriptive strategy at industry-year resolution using NAICS 2-, 3-, and 4-digit groupings.

## Assignment method
- Patents are linked to firms via `patent_id -> gvkey` (Compustat static tranches).
- Firm SIC codes come from `output/firm_year_enriched.csv` (year match with gvkey, fallback to gvkey modal SIC).
- SIC is mapped to NAICS using official Census concordance (`2002_NAICS_to_1987_SIC.xls`).
- Because SIC->NAICS is many-to-many, patents are fractionally assigned across mapped NAICS codes at each digit level (weights sum to 1 per patent per level).

## Selected outputs
- Main-paper candidates: Tables IM1-IM3, Figures IM1-IM6.
- Appendix descriptives: Tables IA1-IA3, Figure IA1.

## Why these outputs
- They quantify cross-industry concentration, trends, and heterogeneity at 2/3/4-digit detail.
- They distinguish level concentration from intensity differences.
- They preserve policy relevance through industry-level policy exposure metrics (patent-weighted country-policy exposure).

## Caveats
- Industry mapping quality depends on SIC availability in firm-year records.
- SIC->NAICS concordance is many-to-many; fractional assignment avoids arbitrary single-code picks but introduces dilution.
- Policy variables are country-level; industry-policy relationships are exposure-based descriptives, not causal industry policies.
