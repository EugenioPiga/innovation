# Patent-Based Data and Empirical Evidence Intensity: Measurement Memo

## 1) Concept
We measure whether a patent reflects **data-dependent, empirical-evidence-driven innovation**, not generic digital/software language. The target concept is reliance on data collection, measurement systems, experimentation/validation, data quality/calibration, statistical inference, and training-data-dependent ML.

## 2) Patent Inputs Used
From the merged Compustat patent universe (3,644,430 patents), we linked PatentsView and assembled a patent-level backbone with:
- `patent_id`, `app_year`, `grant_year`, `patent_date_year`
- `patent_title`, `patent_abstract`, `num_claims`
- assignee/inventor country assignments (mode + primary)
- CPC-derived metadata flags and counts (`has_measurement_cpc`, `has_testing_cpc`, `has_ml_cpc`, `has_statistics_cpc`, etc.)

Coverage in backbone:
- patents: 3,644,430
- with title: 2,989,034
- with abstract: 2,851,769
- with assignee country: 2,969,450
- with inventor country: 2,973,079
- with CPC: 2,839,447

## 3) Candidate Approaches Considered
Implemented and compared five candidates:
1. Dictionary/rules score (transparent lexical patterns + boilerplate penalty)
2. Metadata/CPC score (measurement/testing/ML/statistical CPC signals)
3. Semantic similarity score (LSA embedding vs positive/negative prototypes)
4. Supervised text classifier (TF-IDF + SGD logistic)
5. **Hybrid benchmark** (stacked logistic over dictionary + metadata + boilerplate + supervised text)

## 4) Preferred Benchmark and Why
Preferred benchmark: **hybrid stacked score** (`benchmark_score`) with binary flag (`benchmark_flag`) at an estimated threshold (`0.15`).

Why benchmark this approach:
- more robust than raw keywords (explicitly downweights generic digital boilerplate);
- uses complementary information from both text and CPC metadata;
- still transparent because each component score is retained for diagnostics and interpretation.

## 5) Patent-Level Dimensions Produced
Each patent gets component scores in [0,1]:
- `data_collection_score`
- `empirical_analysis_score`
- `experimental_validation_score`
- `measurement_instrumentation_score`
- `data_quality_calibration_score`
- `ml_training_data_score`
- `boilerplate_score` (generic digital/legal language)

Additional method scores:
- `dictionary_score`
- `metadata_score`
- `semantic_lsa_score`
- `text_supervised_score`
- `benchmark_score` (preferred)
- `benchmark_flag`, `benchmark_confidence`

## 6) Validation Strategy and Current Evidence
Validation assets created:
- candidate method comparison: `datasets/patent_evidence/output/validation/candidate_method_comparison.csv`
- threshold file: `datasets/patent_evidence/output/validation/candidate_method_thresholds.json`
- hand-coding template sample (stratified top/bottom/near-threshold/disagreement, n=1200): `datasets/patent_evidence/output/validation/manual_validation_sample.csv`

Current quantitative comparison (against weak labels):
- dictionary: AUC 1.000, F1 1.000
- hybrid benchmark: AUC 0.9999, F1 0.9958
- text supervised: AUC 0.9998, F1 0.9940
- semantic LSA: AUC 0.8413
- metadata: AUC 0.7062

Important caveat: these holdout metrics are against weak-supervision seed labels, so they are optimistic. The manual validation sample is included specifically to support referee-grade external validation and recalibration.

## 7) Country-Year Aggregation
Produced four assignment/year variants:
- assignee x application year
- assignee x grant year
- inventor x application year
- inventor x grant year

Countries are harmonized from PatentsView ISO-2 to ISO-3 before aggregation.

Benchmark aggregate for panel merge:
- `assignee_country_iso3 x app_year`
- file: `datasets/patent_evidence/output/country_year/patent_evidence_country_year_benchmark.csv`

Panel merged output:
- `datasets/patent_evidence/output/country_year/social_returns_country_year_panel_with_patent_evidence.csv`

## 8) Interpretation and Failure Modes
Interpretation:
- high `benchmark_score` means patent text+metadata indicate stronger reliance on empirical evidence generation/measurement/data quality/testing, conditional on boilerplate controls.

Key failure modes:
- weak-label dependence (label noise and leakage risk);
- missing abstract/title for older patents;
- class- and era-specific language drift;
- some CPC mappings capture related but not identical constructs.

Mitigations implemented:
- hybrid scoring (not single keyword count);
- explicit boilerplate penalties;
- multi-dimensional outputs (not forced single concept);
- preserved alternative method scores for robustness checks;
- manual validation file for transparent re-estimation.

## 9) Output Inventory
Patent-level:
- `datasets/patent_evidence/output/patent_evidence_patent_level.csv.gz`

Diagnostics:
- `datasets/patent_evidence/output/diagnostics/top_patents_by_benchmark_score.csv`
- `datasets/patent_evidence/output/diagnostics/bottom_patents_by_benchmark_score.csv`
- `datasets/patent_evidence/output/diagnostics/most_ambiguous_patents.csv`
- `datasets/patent_evidence/output/diagnostics/tech_class_breakdown.csv`
- `datasets/patent_evidence/output/diagnostics/yearly_score_distribution.csv`
- `datasets/patent_evidence/output/diagnostics/country_score_distribution.csv`

Country-year:
- `datasets/patent_evidence/output/country_year/patent_evidence_country_year_all_variants.csv`
- `datasets/patent_evidence/output/country_year/patent_evidence_country_year_variable_metadata.csv`
- merged panel file listed above.
