#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
PATENT_LEVEL = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_evidence_patent_level.csv.gz"
COUNTRY_PANEL = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "country_year" / "social_returns_country_year_panel_with_patent_evidence.csv"
FIRM_YEAR = REPO_ROOT / "output" / "firm_year_enriched.csv"
STATIC_GLOB = (REPO_ROOT / "datasets" / "compustat_patents_data" / "staticTranche*.csv").as_posix()
STATIC_FALLBACK = REPO_ROOT / "datasets" / "compustat_patents_data" / "static.csv"
XWALK_XLS = REPO_ROOT / "datasets" / "social_returns_data" / "raw" / "naics_crosswalk" / "2002_NAICS_to_1987_SIC.xls"
XWALK_CLEAN = REPO_ROOT / "datasets" / "social_returns_data" / "processed" / "industry" / "sic1987_to_naics2002_crosswalk_clean.csv"

OUT = REPO_ROOT / "outputs" / "descriptives_industry"
DATA_DIR = OUT / "data"
T_MAIN = OUT / "tables" / "main"
T_APP = OUT / "tables" / "appendix"
F_MAIN = OUT / "figures" / "main"
F_APP = OUT / "figures" / "appendix"
M_DIR = OUT / "memo"
I_DIR = OUT / "index"


def ensure_dirs() -> None:
    for p in [DATA_DIR, T_MAIN, T_APP, F_MAIN, F_APP, M_DIR, I_DIR, XWALK_CLEAN.parent]:
        p.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_table(
    df: pd.DataFrame,
    stem: str,
    out_dir: Path,
    float_cols: Iterable[str] | None = None,
    digits: int = 3,
) -> tuple[Path, Path]:
    x = df.copy()
    if float_cols:
        for c in float_cols:
            if c in x.columns:
                x[c] = x[c].round(digits)
    csv = out_dir / f"{stem}.csv"
    tex = out_dir / f"{stem}.tex"
    x.to_csv(csv, index=False)
    x.to_latex(tex, index=False, na_rep="", escape=False, float_format=lambda v: f"{v:.{digits}f}")
    return csv, tex


def save_fig(fig: plt.Figure, stem: str, out_dir: Path) -> tuple[Path, Path]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def normalize_digits(v: object) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"[^0-9]", "", str(v))
    if not s:
        return None
    return str(int(s))


def build_crosswalk() -> pd.DataFrame:
    if XWALK_CLEAN.exists() and XWALK_CLEAN.stat().st_size > 0:
        return pd.read_csv(XWALK_CLEAN, dtype=str)

    raw = pd.read_excel(XWALK_XLS)
    naics_col = raw.columns[0]
    sic_col = raw.columns[2]

    x = raw[[naics_col, sic_col]].copy()
    x.columns = ["naics_code_raw", "sic_code_raw"]
    x["sic_key"] = x["sic_code_raw"].map(normalize_digits)
    x["naics6"] = x["naics_code_raw"].map(normalize_digits)
    x = x.dropna(subset=["sic_key", "naics6"]).copy()
    x["naics6"] = x["naics6"].str.zfill(6)
    x = x[x["naics6"].str.match(r"^[0-9]{6}$")]
    x["naics2"] = x["naics6"].str[:2]
    x["naics3"] = x["naics6"].str[:3]
    x["naics4"] = x["naics6"].str[:4]
    x = x[["sic_key", "naics6", "naics2", "naics3", "naics4"]].drop_duplicates().sort_values(["sic_key", "naics6"])
    x.to_csv(XWALK_CLEAN, index=False)
    return x


def iso2_to_iso3_map() -> pd.DataFrame:
    rows = []
    for c in pycountry.countries:
        if hasattr(c, "alpha_2") and hasattr(c, "alpha_3"):
            rows.append({"iso2": c.alpha_2.upper(), "iso3": c.alpha_3.upper()})
    rows.extend([
        {"iso2": "XK", "iso3": "XKX"},
        {"iso2": "SU", "iso3": "SUN"},
        {"iso2": "YU", "iso3": "YUG"},
        {"iso2": "AN", "iso3": "ANT"},
    ])
    return pd.DataFrame(rows).drop_duplicates(subset=["iso2"], keep="first")


def summary_stats(df: pd.DataFrame, cols: list[str], labels: dict[str, str]) -> pd.DataFrame:
    out = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        out.append(
            {
                "Variable": labels.get(c, c),
                "N": int(s.shape[0]),
                "Mean": s.mean(),
                "SD": s.std(),
                "P10": s.quantile(0.10),
                "P25": s.quantile(0.25),
                "Median": s.quantile(0.50),
                "P75": s.quantile(0.75),
                "P90": s.quantile(0.90),
                "Min": s.min(),
                "Max": s.max(),
            }
        )
    return pd.DataFrame(out)


def var_decomp(df: pd.DataFrame, id_col: str, var: str) -> dict[str, float]:
    d = df[[id_col, var]].copy()
    d[var] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna()
    if d.empty:
        return {"N": 0, "Groups": 0, "Overall Var": np.nan, "Between Var": np.nan, "Within Var": np.nan, "Within/Overall": np.nan}
    overall = float(d[var].var(ddof=0))
    means = d.groupby(id_col)[var].mean()
    between = float(means.var(ddof=0))
    d = d.join(means.rename("_m"), on=id_col)
    within = float(((d[var] - d["_m"]) ** 2).mean())
    return {
        "N": int(len(d)),
        "Groups": int(d[id_col].nunique()),
        "Overall Var": overall,
        "Between Var": between,
        "Within Var": within,
        "Within/Overall": within / overall if overall > 0 else np.nan,
    }

def build_industry_panels() -> dict[str, pd.DataFrame]:
    xwalk = build_crosswalk()
    iso_map = iso2_to_iso3_map()

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    if list((REPO_ROOT / "datasets" / "compustat_patents_data").glob("staticTranche*.csv")):
        static_source = STATIC_GLOB
    else:
        static_source = STATIC_FALLBACK.as_posix()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE patent_gvkey_raw AS
        SELECT
          UPPER(TRIM(patent_id)) AS patent_id,
          TRY_CAST(appYear AS INTEGER) AS app_year,
          NULLIF(TRIM(COALESCE(gvkeyUO, gvkeyFR)), '') AS gvkey
        FROM read_csv_auto('{static_source}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        WHERE patent_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE patent_gvkey AS
        WITH ranked AS (
          SELECT
            patent_id,
            app_year,
            gvkey,
            ROW_NUMBER() OVER (
              PARTITION BY patent_id
              ORDER BY CASE WHEN gvkey IS NULL THEN 1 ELSE 0 END, app_year NULLS LAST
            ) AS rn
          FROM patent_gvkey_raw
        )
        SELECT patent_id, app_year, gvkey
        FROM ranked
        WHERE rn = 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE firm_year AS
        SELECT
          NULLIF(TRIM(gvkey), '') AS gvkey,
          TRY_CAST(year AS INTEGER) AS year,
          NULLIF(TRIM(CAST(sic AS VARCHAR)), '') AS sic
        FROM read_csv_auto('{FIRM_YEAR.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        WHERE gvkey IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE firm_gvkey_mode_sic AS
        WITH c AS (
          SELECT gvkey, sic, COUNT(*) AS n
          FROM firm_year
          WHERE sic IS NOT NULL
          GROUP BY 1,2
        ), r AS (
          SELECT gvkey, sic, n,
                 ROW_NUMBER() OVER (PARTITION BY gvkey ORDER BY n DESC, sic ASC) AS rn
          FROM c
        )
        SELECT gvkey, sic AS sic_mode
        FROM r WHERE rn = 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE patents AS
        SELECT
          UPPER(TRIM(patent_id)) AS patent_id,
          TRY_CAST(app_year AS INTEGER) AS app_year,
          NULLIF(TRIM(assignee_country_mode), '') AS assignee_iso2,
          TRY_CAST(benchmark_score AS DOUBLE) AS benchmark_score,
          TRY_CAST(benchmark_flag AS DOUBLE) AS benchmark_flag,
          TRY_CAST(dictionary_score AS DOUBLE) AS dictionary_score,
          TRY_CAST(metadata_score AS DOUBLE) AS metadata_score,
          TRY_CAST(text_supervised_score AS DOUBLE) AS text_supervised_score,
          TRY_CAST(semantic_lsa_score AS DOUBLE) AS semantic_lsa_score,
          TRY_CAST(data_collection_score AS DOUBLE) AS data_collection_score,
          TRY_CAST(empirical_analysis_score AS DOUBLE) AS empirical_analysis_score,
          TRY_CAST(experimental_validation_score AS DOUBLE) AS experimental_validation_score,
          TRY_CAST(measurement_instrumentation_score AS DOUBLE) AS measurement_instrumentation_score,
          TRY_CAST(data_quality_calibration_score AS DOUBLE) AS data_quality_calibration_score,
          TRY_CAST(ml_training_data_score AS DOUBLE) AS ml_training_data_score,
          TRY_CAST(boilerplate_score AS DOUBLE) AS boilerplate_score,
          TRY_CAST(has_measurement_cpc AS DOUBLE) AS has_measurement_cpc,
          TRY_CAST(has_testing_cpc AS DOUBLE) AS has_testing_cpc,
          TRY_CAST(has_ml_cpc AS DOUBLE) AS has_ml_cpc,
          TRY_CAST(has_data_processing_cpc AS DOUBLE) AS has_data_processing_cpc,
          TRY_CAST(has_statistics_cpc AS DOUBLE) AS has_statistics_cpc,
          TRY_CAST(has_bio_assay_cpc AS DOUBLE) AS has_bio_assay_cpc
        FROM read_csv_auto('{PATENT_LEVEL.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE patent_with_sic AS
        SELECT
          p.*,
          g.gvkey,
          COALESCE(fy.sic, fm.sic_mode) AS sic_raw
        FROM patents p
        LEFT JOIN patent_gvkey g USING (patent_id)
        LEFT JOIN firm_year fy ON g.gvkey = fy.gvkey AND p.app_year = fy.year
        LEFT JOIN firm_gvkey_mode_sic fm ON g.gvkey = fm.gvkey
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE patent_base AS
        SELECT
          *,
          CASE
            WHEN regexp_replace(COALESCE(sic_raw, ''), '[^0-9]', '', 'g') = '' THEN NULL
            ELSE CAST(CAST(regexp_replace(sic_raw, '[^0-9]', '', 'g') AS BIGINT) AS VARCHAR)
          END AS sic_key
        FROM patent_with_sic
        """
    )

    con.register("xwalk_df", xwalk)
    con.execute("CREATE OR REPLACE TABLE xwalk AS SELECT * FROM xwalk_df")

    con.execute(
        """
        CREATE OR REPLACE TABLE xwalk2 AS
        WITH t AS (
          SELECT DISTINCT sic_key, naics2 AS naics_code
          FROM xwalk
        ), d AS (
          SELECT sic_key, COUNT(*) AS n_codes
          FROM t
          GROUP BY 1
        )
        SELECT t.sic_key, t.naics_code, d.n_codes
        FROM t JOIN d USING (sic_key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE xwalk3 AS
        WITH t AS (
          SELECT DISTINCT sic_key, naics3 AS naics_code
          FROM xwalk
        ), d AS (
          SELECT sic_key, COUNT(*) AS n_codes
          FROM t
          GROUP BY 1
        )
        SELECT t.sic_key, t.naics_code, d.n_codes
        FROM t JOIN d USING (sic_key)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE xwalk4 AS
        WITH t AS (
          SELECT DISTINCT sic_key, naics4 AS naics_code
          FROM xwalk
        ), d AS (
          SELECT sic_key, COUNT(*) AS n_codes
          FROM t
          GROUP BY 1
        )
        SELECT t.sic_key, t.naics_code, d.n_codes
        FROM t JOIN d USING (sic_key)
        """
    )

    panel = pd.read_csv(
        COUNTRY_PANEL,
        usecols=[
            "iso3c",
            "year",
            "datagouv_privacy_level_ordinal",
            "unctad_privacy_and_data_protection_code",
            "ecipe_dte_n_measures_cum",
        ],
    )
    con.register("country_policy_df", panel)
    con.execute("CREATE OR REPLACE TABLE country_policy AS SELECT * FROM country_policy_df")

    iso = iso_map.copy()
    con.register("iso_map_df", iso)
    con.execute("CREATE OR REPLACE TABLE iso_map AS SELECT * FROM iso_map_df")

    con.execute(
        """
        CREATE OR REPLACE TABLE patent_policy AS
        SELECT
          p.*,
          m.iso3 AS assignee_iso3,
          cp.datagouv_privacy_level_ordinal,
          cp.unctad_privacy_and_data_protection_code,
          cp.ecipe_dte_n_measures_cum
        FROM patent_base p
        LEFT JOIN iso_map m ON UPPER(p.assignee_iso2) = m.iso2
        LEFT JOIN country_policy cp ON cp.iso3c = m.iso3 AND cp.year = p.app_year
        """
    )

    def agg_level(level: int) -> pd.DataFrame:
        x = f"xwalk{level}"
        q = f"""
        SELECT
          m.naics_code AS naics_code,
          p.app_year AS year,
          SUM(1.0 / m.n_codes) AS patent_count,
          SUM(p.benchmark_flag * (1.0 / m.n_codes)) AS data_driven_patent_count,
          SUM(p.benchmark_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS benchmark_score_mean,
          SUM(p.dictionary_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS dictionary_score_mean,
          SUM(p.metadata_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS metadata_score_mean,
          SUM(p.text_supervised_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS text_supervised_score_mean,
          SUM(p.semantic_lsa_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS semantic_lsa_score_mean,
          SUM(p.data_collection_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS data_collection_score_mean,
          SUM(p.empirical_analysis_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS empirical_analysis_score_mean,
          SUM(p.experimental_validation_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS experimental_validation_score_mean,
          SUM(p.measurement_instrumentation_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS measurement_instrumentation_score_mean,
          SUM(p.data_quality_calibration_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS data_quality_calibration_score_mean,
          SUM(p.ml_training_data_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS ml_training_data_score_mean,
          SUM(p.boilerplate_score * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS boilerplate_score_mean,
          SUM(p.has_measurement_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_measurement_cpc_share,
          SUM(p.has_testing_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_testing_cpc_share,
          SUM(p.has_ml_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_ml_cpc_share,
          SUM(p.has_data_processing_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_data_processing_cpc_share,
          SUM(p.has_statistics_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_statistics_cpc_share,
          SUM(p.has_bio_assay_cpc * (1.0 / m.n_codes)) / NULLIF(SUM(1.0 / m.n_codes), 0) AS has_bio_assay_cpc_share,
          SUM(COALESCE(p.datagouv_privacy_level_ordinal, NULL) * (1.0 / m.n_codes)) / NULLIF(SUM(CASE WHEN p.datagouv_privacy_level_ordinal IS NULL THEN 0 ELSE (1.0 / m.n_codes) END), 0) AS policy_privacy_level_mean,
          SUM(CASE WHEN p.unctad_privacy_and_data_protection_code = 1 THEN (1.0 / m.n_codes) ELSE 0 END) / NULLIF(SUM(CASE WHEN p.unctad_privacy_and_data_protection_code IS NULL THEN 0 ELSE (1.0 / m.n_codes) END), 0) AS policy_unctad_privacy_law_share,
          SUM(COALESCE(p.ecipe_dte_n_measures_cum, NULL) * (1.0 / m.n_codes)) / NULLIF(SUM(CASE WHEN p.ecipe_dte_n_measures_cum IS NULL THEN 0 ELSE (1.0 / m.n_codes) END), 0) AS policy_ecipe_cum_mean,
          COUNT(*) AS raw_links
        FROM patent_policy p
        JOIN {x} m ON p.sic_key = m.sic_key
        WHERE p.app_year IS NOT NULL
        GROUP BY 1,2
        ORDER BY 2,1
        """
        df = con.execute(q).fetchdf()
        df["data_driven_patent_share"] = df["data_driven_patent_count"] / df["patent_count"].replace(0, np.nan)
        df["naics_level"] = level
        return df

    ind2 = agg_level(2)
    ind3 = agg_level(3)
    ind4 = agg_level(4)

    ind2.to_csv(DATA_DIR / "industry_year_naics2.csv", index=False)
    ind3.to_csv(DATA_DIR / "industry_year_naics3.csv", index=False)
    ind4.to_csv(DATA_DIR / "industry_year_naics4.csv", index=False)

    coverage = con.execute(
        """
        SELECT
          COUNT(*) AS n_patents,
          SUM(CASE WHEN gvkey IS NOT NULL THEN 1 ELSE 0 END) AS n_with_gvkey,
          SUM(CASE WHEN sic_key IS NOT NULL THEN 1 ELSE 0 END) AS n_with_sic,
          SUM(CASE WHEN assignee_iso3 IS NOT NULL THEN 1 ELSE 0 END) AS n_with_assignee_iso3
        FROM patent_policy
        """
    ).fetchdf().iloc[0].to_dict()

    (DATA_DIR / "industry_mapping_coverage.json").write_text(json.dumps({k: float(v) for k, v in coverage.items()}, indent=2))

    return {"naics2": ind2, "naics3": ind3, "naics4": ind4, "coverage": pd.DataFrame([coverage])}

def table_dataset_overview(panel2: pd.DataFrame, panel3: pd.DataFrame, panel4: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    c = coverage_df.iloc[0]
    rows = []
    for level, df in [(2, panel2), (3, panel3), (4, panel4)]:
        rows.append(
            {
                "Dataset": f"Industry-year panel (NAICS {level}-digit)",
                "Unit": f"NAICS{level}-year",
                "Observations": int(len(df)),
                "Industries": int(df["naics_code"].nunique()),
                "Year start": int(df["year"].min()),
                "Year end": int(df["year"].max()),
            }
        )
    rows.append(
        {
            "Dataset": "Patent-level mapping coverage",
            "Unit": "Patent",
            "Observations": int(c["n_patents"]),
            "Industries": np.nan,
            "Year start": np.nan,
            "Year end": np.nan,
        }
    )
    out = pd.DataFrame(rows)
    out["Share with gvkey"] = [np.nan, np.nan, np.nan, c["n_with_gvkey"] / c["n_patents"]]
    out["Share with SIC"] = [np.nan, np.nan, np.nan, c["n_with_sic"] / c["n_patents"]]
    return out


def table_key_summary(ind4: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "patent_count": "Total patents (weighted)",
        "data_driven_patent_count": "Data/evidence patents (weighted)",
        "data_driven_patent_share": "Data/evidence share",
        "benchmark_score_mean": "Mean benchmark score",
        "policy_privacy_level_mean": "Patent-weighted data.gouv privacy level exposure",
        "policy_unctad_privacy_law_share": "Patent-weighted UNCTAD privacy-law exposure share",
        "policy_ecipe_cum_mean": "Patent-weighted ECIPE cumulative policy exposure",
        "has_measurement_cpc_share": "Measurement-CPC share",
        "has_ml_cpc_share": "ML-CPC share",
        "has_data_processing_cpc_share": "Data-processing-CPC share",
    }
    return summary_stats(ind4, list(labels.keys()), labels)


def table_top_industries(ind4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = (
        ind4.groupby("naics_code", as_index=False)
        .agg(
            total_data_patents=("data_driven_patent_count", "sum"),
            total_patents=("patent_count", "sum"),
            avg_data_share=("data_driven_patent_share", "mean"),
            avg_benchmark_score=("benchmark_score_mean", "mean"),
            years_observed=("year", "nunique"),
        )
        .sort_values("total_data_patents", ascending=False)
    )
    agg["global_data_patent_share"] = agg["total_data_patents"] / agg["total_data_patents"].sum()
    top_levels = agg.head(20).copy()
    top_intensity = agg[agg["total_patents"] >= 1000].sort_values("avg_data_share", ascending=False).head(20).copy()
    return top_levels, top_intensity


def table_within_between(ind2: pd.DataFrame, ind3: pd.DataFrame, ind4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    var_labels = {
        "data_driven_patent_share": "Data/evidence share",
        "benchmark_score_mean": "Mean benchmark score",
        "data_driven_patent_count": "Data/evidence count",
        "policy_privacy_level_mean": "Policy exposure: privacy level",
        "policy_ecipe_cum_mean": "Policy exposure: ECIPE cumulative",
    }
    for level, df in [(2, ind2), (3, ind3), (4, ind4)]:
        for v, label in var_labels.items():
            s = var_decomp(df, "naics_code", v)
            s["NAICS level"] = level
            s["Variable"] = label
            rows.append(s)
    out = pd.DataFrame(rows)
    return out[["NAICS level", "Variable", "N", "Groups", "Overall Var", "Between Var", "Within Var", "Within/Overall"]]


def table_policy_bins(ind4: pd.DataFrame) -> pd.DataFrame:
    d = ind4[["naics_code", "year", "data_driven_patent_share", "policy_privacy_level_mean", "policy_ecipe_cum_mean"]].copy()
    out_rows = []
    for var, label in [("policy_privacy_level_mean", "Privacy level exposure"), ("policy_ecipe_cum_mean", "ECIPE cumulative exposure")]:
        x = d[["data_driven_patent_share", var]].dropna().copy()
        if x.empty:
            continue
        nb = min(10, x[var].nunique())
        if nb < 2:
            continue
        x["bin"] = pd.qcut(x[var], q=nb, duplicates="drop")
        z = x.groupby("bin", observed=False, as_index=False).agg(policy_mean=(var, "mean"), outcome_mean=("data_driven_patent_share", "mean"), n=(var, "size"))
        z["Policy variable"] = label
        out_rows.append(z)
    if not out_rows:
        return pd.DataFrame(columns=["Policy variable", "policy_mean", "outcome_mean", "n"])
    out = pd.concat(out_rows, ignore_index=True)
    return out[["Policy variable", "policy_mean", "outcome_mean", "n"]]

def fig_global_trends_by_level(ind2: pd.DataFrame, ind3: pd.DataFrame, ind4: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, (lvl, df) in zip(axes, [(2, ind2), (3, ind3), (4, ind4)]):
        y = df.groupby("year", as_index=False).agg(total=("patent_count", "sum"), data=("data_driven_patent_count", "sum")).sort_values("year")
        y["share"] = y["data"] / y["total"]
        ax2 = ax.twinx()
        ax.plot(y["year"], y["total"], color="#6B7280", lw=1.8, label="Total")
        ax.plot(y["year"], y["data"], color="#0B7189", lw=2.0, label="Data/evidence")
        ax2.plot(y["year"], y["share"], color="#C73E1D", lw=1.8, linestyle="--", label="Share")
        ax.set_title(f"NAICS {lvl}-digit")
        ax.set_xlabel("Year")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"))
        ax2.yaxis.set_major_formatter(PercentFormatter(1))
        if lvl == 2:
            ax.set_ylabel("Weighted patent count")
            ax2.set_ylabel("Data/evidence share")
    fig.suptitle("Figure IM1. Industry-Year Trends in Total and Data/Evidence-Driven Patenting", y=1.05)
    return fig


def fig_concentration(ind2: pd.DataFrame, ind4: pd.DataFrame) -> plt.Figure:
    def curve(df: pd.DataFrame) -> pd.DataFrame:
        c = df.groupby("naics_code", as_index=False)["data_driven_patent_count"].sum().sort_values("data_driven_patent_count", ascending=False)
        c["x"] = np.arange(1, len(c) + 1) / len(c)
        c["y"] = c["data_driven_patent_count"].cumsum() / c["data_driven_patent_count"].sum()
        return c

    c2 = curve(ind2)
    c4 = curve(ind4)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(c2["x"], c2["y"], color="#0B7189", lw=2.2, label="NAICS 2-digit")
    ax.plot(c4["x"], c4["y"], color="#C73E1D", lw=2.0, label="NAICS 4-digit")
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1.3)
    ax.set_xlabel("Cumulative share of industries")
    ax.set_ylabel("Cumulative share of data/evidence patents")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Figure IM2. Cross-Industry Concentration")
    ax.legend(frameon=False, loc="lower right")
    return fig


def fig_levels_vs_intensity(top_levels: pd.DataFrame, top_intensity: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    left = top_levels.sort_values("total_data_patents", ascending=True).tail(12)
    axes[0].barh(left["naics_code"], left["total_data_patents"], color="#0B7189")
    axes[0].set_title("Top NAICS4 by Total Data/Evidence Patents")
    axes[0].set_xlabel("Weighted data/evidence patent count")

    right = top_intensity.sort_values("avg_data_share", ascending=True).tail(12)
    axes[1].barh(right["naics_code"], right["avg_data_share"], color="#C73E1D")
    axes[1].set_title("Top NAICS4 by Average Data/Evidence Share")
    axes[1].set_xlabel("Average data/evidence share")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle("Figure IM3. Levels vs Intensity Across Industries", y=1.03)
    return fig


def fig_naics2_heterogeneity(ind2: pd.DataFrame) -> plt.Figure:
    top = (
        ind2.groupby("naics_code", as_index=False)["data_driven_patent_count"]
        .sum()
        .sort_values("data_driven_patent_count", ascending=False)
        .head(8)["naics_code"]
        .tolist()
    )
    d = ind2[ind2["naics_code"].isin(top)].copy()

    y = d.groupby(["year", "naics_code"], as_index=False).agg(
        data=("data_driven_patent_count", "sum"),
        total=("patent_count", "sum"),
    )
    y["share"] = y["data"] / y["total"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for code in top:
        s = y[y["naics_code"] == code]
        axes[0].plot(s["year"], s["share"], lw=1.8, label=code)
        axes[1].plot(s["year"], s["data"], lw=1.8, label=code)

    axes[0].set_title("Data/Evidence Share by NAICS2 (Top Sectors)")
    axes[0].set_xlabel("Year")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Data/Evidence Patent Count by NAICS2")
    axes[1].set_xlabel("Year")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"))
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Figure IM4. Industry Heterogeneity Over Time", y=1.03)
    return fig

def fig_tech_by_industry(ind4: pd.DataFrame) -> plt.Figure:
    d = ind4.copy()
    d["year_decade"] = (d["year"] // 10) * 10
    top_ind = (
        d.groupby("naics_code", as_index=False)["data_driven_patent_count"]
        .sum()
        .sort_values("data_driven_patent_count", ascending=False)
        .head(12)["naics_code"]
        .tolist()
    )
    x = d[d["naics_code"].isin(top_ind)].groupby("naics_code", as_index=False).agg(
        measurement=("has_measurement_cpc_share", "mean"),
        ml=("has_ml_cpc_share", "mean"),
        data_proc=("has_data_processing_cpc_share", "mean"),
        testing=("has_testing_cpc_share", "mean"),
        data_share=("data_driven_patent_share", "mean"),
    )
    x = x.sort_values("data_share", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(x["naics_code"], x["measurement"], color="#0B7189", label="Measurement CPC")
    axes[0].barh(x["naics_code"], x["ml"], left=x["measurement"], color="#F4A259", label="ML CPC")
    axes[0].set_title("CPC Signal Composition in Top NAICS4 Industries")
    axes[0].set_xlabel("Average CPC signal share")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].barh(x["naics_code"], x["data_share"], color="#C73E1D")
    axes[1].set_title("Average Data/Evidence Patent Share")
    axes[1].set_xlabel("Share")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle("Figure IM5. Technology Signals Across Industries", y=1.03)
    return fig


def fig_policy_gradients(ind4: pd.DataFrame) -> plt.Figure:
    d = ind4[["data_driven_patent_share", "policy_privacy_level_mean", "policy_ecipe_cum_mean"]].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    p = d.dropna(subset=["policy_privacy_level_mean", "data_driven_patent_share"]).copy()
    if not p.empty:
        p["bin"] = pd.qcut(p["policy_privacy_level_mean"], q=min(8, p["policy_privacy_level_mean"].nunique()), duplicates="drop")
        b = p.groupby("bin", observed=False, as_index=False).agg(x=("policy_privacy_level_mean", "mean"), y=("data_driven_patent_share", "mean"), n=("policy_privacy_level_mean", "size"))
        axes[0].scatter(b["x"], b["y"], s=20 + 2 * np.sqrt(b["n"]), color="#0B7189")
        z = np.polyfit(b["x"], b["y"], 1)
        xl = np.linspace(b["x"].min(), b["x"].max(), 100)
        axes[0].plot(xl, z[0] * xl + z[1], "--", color="#C73E1D", lw=1.8)
    axes[0].set_title("Binned: Data Share vs Privacy-Level Exposure")
    axes[0].set_xlabel("Patent-weighted privacy-level exposure")
    axes[0].set_ylabel("Data/evidence patent share")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    e = d.dropna(subset=["policy_ecipe_cum_mean", "data_driven_patent_share"]).copy()
    if not e.empty:
        e["bin"] = pd.qcut(e["policy_ecipe_cum_mean"], q=min(10, e["policy_ecipe_cum_mean"].nunique()), duplicates="drop")
        b2 = e.groupby("bin", observed=False, as_index=False).agg(x=("policy_ecipe_cum_mean", "mean"), y=("data_driven_patent_share", "mean"), n=("policy_ecipe_cum_mean", "size"))
        axes[1].scatter(b2["x"], b2["y"], s=20 + 2 * np.sqrt(b2["n"]), color="#0B7189")
        z2 = np.polyfit(b2["x"], b2["y"], 1)
        xl2 = np.linspace(b2["x"].min(), b2["x"].max(), 100)
        axes[1].plot(xl2, z2[0] * xl2 + z2[1], "--", color="#C73E1D", lw=1.8)
    axes[1].set_title("Binned: Data Share vs ECIPE Policy Exposure")
    axes[1].set_xlabel("Patent-weighted ECIPE cumulative exposure")
    axes[1].set_ylabel("Data/evidence patent share")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))

    fig.suptitle("Figure IM6. Industry-Level Policy-Exposure Gradients (Descriptive)", y=1.03)
    return fig


def fig_level_comparison(ind2: pd.DataFrame, ind3: pd.DataFrame, ind4: pd.DataFrame) -> plt.Figure:
    def summarize_level(df: pd.DataFrame, level: int) -> pd.DataFrame:
        y = df.groupby("year", as_index=False).agg(
            share=("data_driven_patent_share", "mean"),
            weighted_share=("data_driven_patent_count", "sum"),
            total=("patent_count", "sum"),
        )
        y["weighted_share"] = y["weighted_share"] / y["total"]
        y["level"] = f"NAICS {level}-digit"
        return y

    x = pd.concat([summarize_level(ind2, 2), summarize_level(ind3, 3), summarize_level(ind4, 4)], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for lev, g in x.groupby("level"):
        axes[0].plot(g["year"], g["weighted_share"], lw=2.0, label=lev)
        axes[1].plot(g["year"], g["share"], lw=2.0, label=lev)
    axes[0].set_title("Patent-Weighted Share by NAICS Level")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlabel("Year")
    axes[1].set_title("Unweighted Industry Mean Share by NAICS Level")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlabel("Year")
    axes[0].legend(frameon=False)
    fig.suptitle("Figure IA1. Robustness Across NAICS Aggregation Levels", y=1.03)
    return fig

def write_index_and_memo(index_rows: list[dict[str, str]]) -> None:
    idx = pd.DataFrame(index_rows)
    idx_csv = I_DIR / "industry_descriptive_index.csv"
    idx_md = I_DIR / "industry_descriptive_index.md"
    idx.to_csv(idx_csv, index=False)
    idx_md.write_text(
        "# Industry Descriptive Output Index\n\n"
        + "\n".join(
            f"- **{r.Filename}** | {r.Title} | Sample: {r.Sample} | Variables: {r.Variables} | Interpretation: {r.Interpretation}"
            for _, r in idx.iterrows()
        )
    )

    memo = """# Industry Descriptive Memo

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
"""
    (M_DIR / "industry_descriptive_memo.md").write_text(memo)


def main() -> None:
    ensure_dirs()
    set_style()

    built = build_industry_panels()
    ind2 = built["naics2"].copy()
    ind3 = built["naics3"].copy()
    ind4 = built["naics4"].copy()
    coverage = built["coverage"].copy()

    index_rows: list[dict[str, str]] = []

    t1 = table_dataset_overview(ind2, ind3, ind4, coverage)
    t1_csv, t1_tex = save_table(t1, "table_im1_dataset_overview", T_MAIN, ["Share with gvkey", "Share with SIC"], 3)
    index_rows += [
        {"Section": "Main-paper candidates", "Filename": t1_csv.name, "Title": "Table IM1. Industry dataset overview and mapping coverage", "Sample": "NAICS2/3/4 industry-year panels + patent mapping", "Variables": "Obs/industry/year and mapping shares", "Interpretation": "Shows industry sample scope and mapping quality."},
        {"Section": "Main-paper candidates", "Filename": t1_tex.name, "Title": "Table IM1. Industry dataset overview (LaTeX)", "Sample": "Same as IM1", "Variables": "Same as IM1", "Interpretation": "Paper-ready table."},
    ]

    t2 = table_key_summary(ind4)
    t2_csv, t2_tex = save_table(t2, "table_im2_key_summary_stats_naics4", T_MAIN, ["Mean", "SD", "P10", "P25", "Median", "P75", "P90", "Min", "Max"], 3)
    index_rows += [
        {"Section": "Main-paper candidates", "Filename": t2_csv.name, "Title": "Table IM2. Key summary statistics (NAICS4-year)", "Sample": "NAICS4-year benchmark", "Variables": "Core outcomes, CPC composition, policy exposure", "Interpretation": "Characterizes cross-industry variation in key measures."},
        {"Section": "Main-paper candidates", "Filename": t2_tex.name, "Title": "Table IM2. Key summary statistics (LaTeX)", "Sample": "Same as IM2", "Variables": "Same as IM2", "Interpretation": "Paper-ready table."},
    ]

    top_levels, top_intensity = table_top_industries(ind4)
    t3_csv, t3_tex = save_table(top_levels[["naics_code", "total_data_patents", "global_data_patent_share", "avg_data_share", "avg_benchmark_score", "years_observed"]], "table_im3_top_industries_levels", T_MAIN, ["global_data_patent_share", "avg_data_share", "avg_benchmark_score"], 3)
    a2_csv, a2_tex = save_table(top_intensity[["naics_code", "total_patents", "avg_data_share", "avg_benchmark_score", "years_observed"]], "table_ia2_top_industries_intensity", T_APP, ["avg_data_share", "avg_benchmark_score"], 3)
    index_rows += [
        {"Section": "Main-paper candidates", "Filename": t3_csv.name, "Title": "Table IM3. Top industries by data/evidence patenting", "Sample": "NAICS4 totals over all years", "Variables": "Total contribution and intensity", "Interpretation": "Separates size from specialization across industries."},
        {"Section": "Main-paper candidates", "Filename": t3_tex.name, "Title": "Table IM3. Top industries by data/evidence patenting (LaTeX)", "Sample": "Same as IM3", "Variables": "Same as IM3", "Interpretation": "Paper-ready table."},
    ]
    index_rows += [
        {"Section": "Appendix descriptives", "Filename": a2_csv.name, "Title": "Table IA2. Top industries by data/evidence intensity", "Sample": "NAICS4 industries with at least 1,000 weighted patents", "Variables": "Average data/evidence share and benchmark score", "Interpretation": "Highlights specialization after filtering out very small industries."},
        {"Section": "Appendix descriptives", "Filename": a2_tex.name, "Title": "Table IA2. Top industries by data/evidence intensity (LaTeX)", "Sample": "Same as IA2", "Variables": "Same as IA2", "Interpretation": "Paper-ready table."},
    ]

    a1 = table_within_between(ind2, ind3, ind4)
    a1_csv, a1_tex = save_table(a1, "table_ia1_within_between_variance", T_APP, ["Overall Var", "Between Var", "Within Var", "Within/Overall"], 4)
    index_rows += [
        {"Section": "Appendix descriptives", "Filename": a1_csv.name, "Title": "Table IA1. Within-vs-between variance by NAICS level", "Sample": "NAICS2/3/4 industry-year", "Variables": "Outcomes and policy exposure vars", "Interpretation": "Checks within-industry temporal variation for panel credibility."},
        {"Section": "Appendix descriptives", "Filename": a1_tex.name, "Title": "Table IA1. Within-vs-between variance (LaTeX)", "Sample": "Same as IA1", "Variables": "Same as IA1", "Interpretation": "Paper-ready table."},
    ]

    a3 = table_policy_bins(ind4)
    a3_csv, a3_tex = save_table(a3, "table_ia3_policy_bin_means", T_APP, ["policy_mean", "outcome_mean"], 3)
    index_rows += [
        {"Section": "Appendix descriptives", "Filename": a3_csv.name, "Title": "Table IA3. Binned policy-exposure means", "Sample": "NAICS4-year with non-missing policy exposure", "Variables": "Binned policy means and data/evidence share", "Interpretation": "Supports policy-gradient figures with tabular values."},
        {"Section": "Appendix descriptives", "Filename": a3_tex.name, "Title": "Table IA3. Binned policy-exposure means (LaTeX)", "Sample": "Same as IA3", "Variables": "Same as IA3", "Interpretation": "Paper-ready table."},
    ]

    f1 = fig_global_trends_by_level(ind2, ind3, ind4)
    f1_png, f1_pdf = save_fig(f1, "figure_im1_global_trends_by_naics_level", F_MAIN)
    f2 = fig_concentration(ind2, ind4)
    f2_png, f2_pdf = save_fig(f2, "figure_im2_industry_concentration", F_MAIN)
    f3 = fig_levels_vs_intensity(top_levels, top_intensity)
    f3_png, f3_pdf = save_fig(f3, "figure_im3_levels_vs_intensity", F_MAIN)
    f4 = fig_naics2_heterogeneity(ind2)
    f4_png, f4_pdf = save_fig(f4, "figure_im4_naics2_heterogeneity", F_MAIN)
    f5 = fig_tech_by_industry(ind4)
    f5_png, f5_pdf = save_fig(f5, "figure_im5_technology_signals_by_industry", F_MAIN)
    f6 = fig_policy_gradients(ind4)
    f6_png, f6_pdf = save_fig(f6, "figure_im6_policy_exposure_gradients", F_MAIN)
    fa1 = fig_level_comparison(ind2, ind3, ind4)
    fa1_png, fa1_pdf = save_fig(fa1, "figure_ia1_level_comparison", F_APP)

    for fn, title, sample, vars_, interp, section in [
        (f1_png.name, "Figure IM1. Industry trends by NAICS level", "NAICS2/3/4 yearly aggregates", "Total/data patents and shares", "Shows evolution across aggregation levels.", "Main-paper candidates"),
        (f1_pdf.name, "Figure IM1. Industry trends by NAICS level (PDF)", "Same as IM1", "Same as IM1", "Vector format.", "Main-paper candidates"),
        (f2_png.name, "Figure IM2. Industry concentration curves", "NAICS2 and NAICS4 totals", "Cumulative industry shares", "Documents concentration across industries.", "Main-paper candidates"),
        (f2_pdf.name, "Figure IM2. Industry concentration curves (PDF)", "Same as IM2", "Same as IM2", "Vector format.", "Main-paper candidates"),
        (f3_png.name, "Figure IM3. Levels vs intensity across industries", "Top NAICS4 industries", "Counts and average shares", "Separates scale from specialization.", "Main-paper candidates"),
        (f3_pdf.name, "Figure IM3. Levels vs intensity (PDF)", "Same as IM3", "Same as IM3", "Vector format.", "Main-paper candidates"),
        (f4_png.name, "Figure IM4. NAICS2 heterogeneity over time", "Top NAICS2 sectors", "Sector trends in share and counts", "Highlights sectoral divergence.", "Main-paper candidates"),
        (f4_pdf.name, "Figure IM4. NAICS2 heterogeneity over time (PDF)", "Same as IM4", "Same as IM4", "Vector format.", "Main-paper candidates"),
        (f5_png.name, "Figure IM5. Technology signals by industry", "Top NAICS4 industries", "CPC-signal shares and data/evidence share", "Shows whether evidence-intensity is broad across technologies.", "Main-paper candidates"),
        (f5_pdf.name, "Figure IM5. Technology signals by industry (PDF)", "Same as IM5", "Same as IM5", "Vector format.", "Main-paper candidates"),
        (f6_png.name, "Figure IM6. Policy-exposure gradients", "NAICS4-year with policy exposure", "Data/evidence share vs policy exposure", "Descriptive policy links at industry level.", "Main-paper candidates"),
        (f6_pdf.name, "Figure IM6. Policy-exposure gradients (PDF)", "Same as IM6", "Same as IM6", "Vector format.", "Main-paper candidates"),
        (fa1_png.name, "Figure IA1. NAICS level robustness", "NAICS2/3/4 yearly aggregates", "Weighted/unweighted shares by level", "Checks aggregation-level robustness.", "Appendix descriptives"),
        (fa1_pdf.name, "Figure IA1. NAICS level robustness (PDF)", "Same as IA1", "Same as IA1", "Vector format.", "Appendix descriptives"),
    ]:
        index_rows.append({"Section": section, "Filename": fn, "Title": title, "Sample": sample, "Variables": vars_, "Interpretation": interp})

    write_index_and_memo(index_rows)

    summary = {
        "naics2_rows": int(len(ind2)),
        "naics3_rows": int(len(ind3)),
        "naics4_rows": int(len(ind4)),
        "naics2_industries": int(ind2["naics_code"].nunique()),
        "naics3_industries": int(ind3["naics_code"].nunique()),
        "naics4_industries": int(ind4["naics_code"].nunique()),
        "year_min": int(min(ind2["year"].min(), ind3["year"].min(), ind4["year"].min())),
        "year_max": int(max(ind2["year"].max(), ind3["year"].max(), ind4["year"].max())),
        "output_root": str(OUT),
    }
    (OUT / "industry_descriptive_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
