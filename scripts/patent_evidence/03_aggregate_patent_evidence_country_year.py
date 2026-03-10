#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pycountry


REPO_ROOT = Path(__file__).resolve().parents[2]
PATENT_LEVEL_PATH = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_evidence_patent_level.csv.gz"
OUT_DIR = REPO_ROOT / "datasets" / "patent_evidence" / "output"
CY_DIR = OUT_DIR / "country_year"
DIAG_DIR = OUT_DIR / "diagnostics"
PANEL_BASE_PATH = REPO_ROOT / "datasets" / "social_returns_data" / "processed" / "panel" / "social_returns_country_year_panel.csv"


def _agg_sql(country_col: str, year_col: str, out_table: str) -> str:
    return f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT
            {country_col} AS iso3c,
            CAST({year_col} AS INTEGER) AS year,
            COUNT(*) AS patent_count,
            SUM(benchmark_flag) AS data_driven_patent_count,
            AVG(benchmark_flag) AS data_driven_patent_share,
            AVG(benchmark_score) AS benchmark_score_mean,
            AVG(CASE WHEN benchmark_score >= q90.threshold THEN 1 ELSE 0 END) AS benchmark_top_decile_share,
            AVG(dictionary_score) AS dictionary_score_mean,
            AVG(metadata_score) AS metadata_score_mean,
            AVG(text_supervised_score) AS text_supervised_score_mean,
            AVG(semantic_lsa_score) AS semantic_lsa_score_mean,
            AVG(data_collection_score) AS data_collection_score_mean,
            AVG(empirical_analysis_score) AS empirical_analysis_score_mean,
            AVG(experimental_validation_score) AS experimental_validation_score_mean,
            AVG(measurement_instrumentation_score) AS measurement_instrumentation_score_mean,
            AVG(data_quality_calibration_score) AS data_quality_calibration_score_mean,
            AVG(ml_training_data_score) AS ml_training_data_score_mean,
            AVG(boilerplate_score) AS boilerplate_score_mean,
            SUM(CASE WHEN num_claims > 0 THEN benchmark_score * num_claims ELSE NULL END)
                / NULLIF(SUM(CASE WHEN num_claims > 0 THEN num_claims ELSE NULL END), 0)
                AS claims_weighted_benchmark_score_mean,
            AVG(benchmark_confidence) AS benchmark_confidence_mean
        FROM patents p
        CROSS JOIN q90
        WHERE {country_col} IS NOT NULL
          AND {country_col} ~ '^[A-Z]{{3}}$'
          AND {year_col} IS NOT NULL
        GROUP BY 1, 2
    """


def _build_iso2_to_iso3_map() -> pd.DataFrame:
    rows = []
    for c in pycountry.countries:
        if hasattr(c, "alpha_2") and hasattr(c, "alpha_3"):
            rows.append({"iso2": c.alpha_2.upper(), "iso3": c.alpha_3.upper()})
    # Common non-standard or historical codes observed in patents data.
    rows.extend(
        [
            {"iso2": "XK", "iso3": "XKX"},
            {"iso2": "SU", "iso3": "SUN"},
            {"iso2": "YU", "iso3": "YUG"},
            {"iso2": "AN", "iso3": "ANT"},
            {"iso2": "TP", "iso3": "TMP"},
            {"iso2": "ZR", "iso3": "ZAR"},
        ]
    )
    out = pd.DataFrame(rows).drop_duplicates(subset=["iso2"], keep="first")
    return out


def run_aggregation() -> None:
    CY_DIR.mkdir(parents=True, exist_ok=True)
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    iso_map = _build_iso2_to_iso3_map()
    con.register("iso_map_df", iso_map)
    con.execute("CREATE OR REPLACE TABLE iso_map AS SELECT * FROM iso_map_df")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW patents_raw AS
        SELECT
            patent_id,
            TRY_CAST(app_year AS INTEGER) AS app_year,
            TRY_CAST(grant_year AS INTEGER) AS grant_year,
            UPPER(TRIM(assignee_country_mode)) AS assignee_country_raw,
            UPPER(TRIM(inventor_country_mode)) AS inventor_country_raw,
            TRY_CAST(num_claims AS DOUBLE) AS num_claims,
            TRY_CAST(dictionary_score AS DOUBLE) AS dictionary_score,
            TRY_CAST(metadata_score AS DOUBLE) AS metadata_score,
            TRY_CAST(semantic_lsa_score AS DOUBLE) AS semantic_lsa_score,
            TRY_CAST(text_supervised_score AS DOUBLE) AS text_supervised_score,
            TRY_CAST(data_collection_score AS DOUBLE) AS data_collection_score,
            TRY_CAST(empirical_analysis_score AS DOUBLE) AS empirical_analysis_score,
            TRY_CAST(experimental_validation_score AS DOUBLE) AS experimental_validation_score,
            TRY_CAST(measurement_instrumentation_score AS DOUBLE) AS measurement_instrumentation_score,
            TRY_CAST(data_quality_calibration_score AS DOUBLE) AS data_quality_calibration_score,
            TRY_CAST(ml_training_data_score AS DOUBLE) AS ml_training_data_score,
            TRY_CAST(boilerplate_score AS DOUBLE) AS boilerplate_score,
            TRY_CAST(benchmark_score AS DOUBLE) AS benchmark_score,
            TRY_CAST(benchmark_flag AS INTEGER) AS benchmark_flag,
            TRY_CAST(benchmark_confidence AS DOUBLE) AS benchmark_confidence,
            TRY_CAST(has_measurement_cpc AS INTEGER) AS has_measurement_cpc,
            TRY_CAST(has_testing_cpc AS INTEGER) AS has_testing_cpc,
            TRY_CAST(has_ml_cpc AS INTEGER) AS has_ml_cpc,
            TRY_CAST(has_data_processing_cpc AS INTEGER) AS has_data_processing_cpc,
            TRY_CAST(has_bio_assay_cpc AS INTEGER) AS has_bio_assay_cpc,
            TRY_CAST(has_statistics_cpc AS INTEGER) AS has_statistics_cpc
        FROM read_csv_auto('{PATENT_LEVEL_PATH.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW patents AS
        SELECT
            p.*,
            CASE
                WHEN p.assignee_country_raw ~ '^[A-Z]{3}$' THEN p.assignee_country_raw
                ELSE m1.iso3
            END AS assignee_country_iso3,
            CASE
                WHEN p.inventor_country_raw ~ '^[A-Z]{3}$' THEN p.inventor_country_raw
                ELSE m2.iso3
            END AS inventor_country_iso3
        FROM patents_raw p
        LEFT JOIN iso_map m1 ON p.assignee_country_raw = m1.iso2
        LEFT JOIN iso_map m2 ON p.inventor_country_raw = m2.iso2
        """
    )
    con.execute("CREATE OR REPLACE TABLE q90 AS SELECT quantile_cont(benchmark_score, 0.9) AS threshold FROM patents")

    specs = [
        ("assignee_country_iso3", "app_year", "assignee_app"),
        ("assignee_country_iso3", "grant_year", "assignee_grant"),
        ("inventor_country_iso3", "app_year", "inventor_app"),
        ("inventor_country_iso3", "grant_year", "inventor_grant"),
    ]
    written_files = []
    for country_col, year_col, tag in specs:
        table = f"agg_{tag}"
        con.execute(_agg_sql(country_col, year_col, table))
        out = CY_DIR / f"patent_evidence_country_year_{tag}.csv"
        con.execute(f"COPY {table} TO '{out.as_posix()}' (HEADER, DELIMITER ',')")
        written_files.append(out)

    # Mergeable long table with assignment/year basis labels
    long_parts = []
    for _, _, tag in specs:
        df = pd.read_csv(CY_DIR / f"patent_evidence_country_year_{tag}.csv")
        assignment_basis, year_basis = tag.split("_")
        df["assignment_basis"] = assignment_basis
        df["year_basis"] = year_basis
        long_parts.append(df)
    mergeable = pd.concat(long_parts, ignore_index=True)
    mergeable.to_csv(CY_DIR / "patent_evidence_country_year_all_variants.csv", index=False)

    # Preferred benchmark variant: assignee country x application year
    benchmark_variant = pd.read_csv(CY_DIR / "patent_evidence_country_year_assignee_app.csv")
    benchmark_variant = benchmark_variant.rename(
        columns={c: f"pe_{c}" for c in benchmark_variant.columns if c not in {"iso3c", "year"}}
    )
    benchmark_variant.to_csv(CY_DIR / "patent_evidence_country_year_benchmark.csv", index=False)

    # Merge into existing country-year panel
    panel = pd.read_csv(PANEL_BASE_PATH)
    merged = panel.merge(benchmark_variant, on=["iso3c", "year"], how="left")
    merged_path = CY_DIR / "social_returns_country_year_panel_with_patent_evidence.csv"
    merged.to_csv(merged_path, index=False)

    # Variable metadata
    metadata_rows = []
    for c in benchmark_variant.columns:
        if c in {"iso3c", "year"}:
            continue
        description = {
            "pe_patent_count": "Total patents in iso3-year (assignee country, application year).",
            "pe_data_driven_patent_count": "Count of patents flagged data/evidence-driven by benchmark classifier.",
            "pe_data_driven_patent_share": "Share of patents flagged data/evidence-driven.",
            "pe_benchmark_score_mean": "Mean benchmark probability score.",
            "pe_benchmark_top_decile_share": "Share of patents above global 90th percentile benchmark score.",
            "pe_dictionary_score_mean": "Mean dictionary/rules score.",
            "pe_metadata_score_mean": "Mean metadata/CPC score.",
            "pe_text_supervised_score_mean": "Mean supervised text-model score.",
            "pe_semantic_lsa_score_mean": "Mean embedding-style semantic similarity score.",
            "pe_data_collection_score_mean": "Mean component: data collection intensity.",
            "pe_empirical_analysis_score_mean": "Mean component: empirical/statistical analysis intensity.",
            "pe_experimental_validation_score_mean": "Mean component: experimental/validation intensity.",
            "pe_measurement_instrumentation_score_mean": "Mean component: measurement/instrumentation intensity.",
            "pe_data_quality_calibration_score_mean": "Mean component: data quality/calibration intensity.",
            "pe_ml_training_data_score_mean": "Mean component: ML/training-data intensity.",
            "pe_boilerplate_score_mean": "Mean generic digital boilerplate score.",
            "pe_claims_weighted_benchmark_score_mean": "Claims-weighted mean benchmark score.",
            "pe_benchmark_confidence_mean": "Mean benchmark confidence (distance from 0.5).",
        }.get(c, "Patent evidence variable.")
        metadata_rows.append(
            {
                "variable": c,
                "source": "PatentsView + hybrid patent evidence classifier",
                "description": description,
                "benchmark_variant": "assignee_country_iso3 x app_year",
            }
        )
    metadata_path = CY_DIR / "patent_evidence_country_year_variable_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    # Diagnostics
    con.execute(
        """
        CREATE OR REPLACE TABLE tech_breakdown AS
        SELECT
            has_measurement_cpc,
            has_testing_cpc,
            has_ml_cpc,
            has_data_processing_cpc,
            has_bio_assay_cpc,
            has_statistics_cpc,
            COUNT(*) AS n_patents,
            AVG(benchmark_score) AS benchmark_score_mean,
            AVG(benchmark_flag) AS benchmark_flag_share
        FROM patents
        GROUP BY 1,2,3,4,5,6
        ORDER BY n_patents DESC
        LIMIT 200
        """
    )
    con.execute(
        f"COPY tech_breakdown TO '{(DIAG_DIR / 'tech_class_breakdown.csv').as_posix()}' (HEADER, DELIMITER ',')"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE yearly_distribution AS
        SELECT
            app_year AS year,
            COUNT(*) AS n_patents,
            AVG(benchmark_score) AS benchmark_mean,
            quantile_cont(benchmark_score, 0.1) AS p10,
            quantile_cont(benchmark_score, 0.5) AS p50,
            quantile_cont(benchmark_score, 0.9) AS p90
        FROM patents
        WHERE app_year IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    )
    con.execute(
        f"COPY yearly_distribution TO '{(DIAG_DIR / 'yearly_score_distribution.csv').as_posix()}' (HEADER, DELIMITER ',')"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE country_distribution AS
        SELECT
            assignee_country_iso3 AS iso3c,
            COUNT(*) AS n_patents,
            AVG(benchmark_score) AS benchmark_mean,
            AVG(benchmark_flag) AS benchmark_share
        FROM patents
        WHERE assignee_country_iso3 IS NOT NULL AND assignee_country_iso3 ~ '^[A-Z]{3}$'
        GROUP BY 1
        ORDER BY n_patents DESC
        """
    )
    con.execute(
        f"COPY country_distribution TO '{(DIAG_DIR / 'country_score_distribution.csv').as_posix()}' (HEADER, DELIMITER ',')"
    )

    summary = {
        "written_variant_files": [str(p) for p in written_files],
        "mergeable_all_variants_file": str(CY_DIR / "patent_evidence_country_year_all_variants.csv"),
        "benchmark_variant_file": str(CY_DIR / "patent_evidence_country_year_benchmark.csv"),
        "merged_panel_file": str(merged_path),
        "variable_metadata_file": str(metadata_path),
    }
    (CY_DIR / "aggregation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_aggregation()
