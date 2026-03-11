#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]

PATENT_BACKBONE = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_backbone.parquet"
PATENT_EVIDENCE = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_evidence_patent_level.csv.gz"
FIRM_YEAR = REPO_ROOT / "output" / "firm_year_enriched.csv"
STATIC_GLOB = (REPO_ROOT / "datasets" / "compustat_patents_data" / "staticTranche*.csv").as_posix()
STATIC_FALLBACK = REPO_ROOT / "datasets" / "compustat_patents_data" / "static.csv"

PATENTSVIEW_RAW = REPO_ROOT / "datasets" / "patentsview" / "raw"
PATENTSVIEW_BASE = "https://s3.amazonaws.com/data.patentsview.org/download/"
REQUIRED_RAW_FILES = [
    "g_application.tsv.zip",
    "g_us_patent_citation.tsv.zip",
    "g_other_reference.tsv.zip",
    "g_gov_interest.tsv.zip",
    "g_gov_interest_org.tsv.zip",
    "g_uspc_at_issue.tsv.zip",
    "g_cpc_at_issue.tsv.zip",
    "g_cpc_current.tsv.zip",
    "g_assignee_disambiguated.tsv.zip",
    "g_inventor_disambiguated.tsv.zip",
]

DATA_ROOT = REPO_ROOT / "datasets" / "dyevre_empirical"
INTERMEDIATE = DATA_ROOT / "intermediate"
DATA_OUT = DATA_ROOT / "output"
DB_PATH = INTERMEDIATE / "dyevre_empirical.duckdb"

OUT = REPO_ROOT / "outputs" / "dyevre_empirical"
FIG_MAIN = OUT / "figures" / "main"
FIG_APP = OUT / "figures" / "appendix"
TAB_MAIN = OUT / "tables" / "main"
TAB_APP = OUT / "tables" / "appendix"
MEMO_DIR = OUT / "memo"
INDEX_DIR = OUT / "index"


def ensure_dirs() -> None:
    for p in [PATENTSVIEW_RAW, INTERMEDIATE, DATA_OUT, FIG_MAIN, FIG_APP, TAB_MAIN, TAB_APP, MEMO_DIR, INDEX_DIR]:
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
                x[c] = pd.to_numeric(x[c], errors="coerce").round(digits)
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


def download_if_missing(file_name: str) -> Path:
    out = PATENTSVIEW_RAW / file_name
    if out.exists() and out.stat().st_size > 0:
        return out

    url = PATENTSVIEW_BASE + file_name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def ensure_raw_inputs() -> None:
    missing = []
    for fn in REQUIRED_RAW_FILES:
        p = PATENTSVIEW_RAW / fn
        if not (p.exists() and p.stat().st_size > 0):
            missing.append(fn)
    if not missing:
        return
    for fn in missing:
        download_if_missing(fn)


def configure_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='24GB'")
    return con


def sql_read(path: Path, delim: str = "\t", zipped: bool = True, cols: list[str] | None = None) -> str:
    del zipped
    if cols:
        cols_sql = ", ".join(f"'{c}': 'VARCHAR'" for c in cols)
        return (
            f"read_csv('{path.as_posix()}', delim='{delim}', header=TRUE, "
            f"columns={{ {cols_sql} }}, auto_detect=FALSE, quote='\"', escape='\"', "
            "ignore_errors=TRUE, strict_mode=FALSE, parallel=FALSE)"
        )
    return (
        f"read_csv('{path.as_posix()}', delim='{delim}', header=TRUE, "
        "all_varchar=TRUE, quote='\"', escape='\"', ignore_errors=TRUE, strict_mode=FALSE, parallel=FALSE)"
    )


def build_tables(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE universe AS
        SELECT
          UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
          TRY_CAST(app_year AS INTEGER) AS app_year,
          TRY_CAST(grant_year AS INTEGER) AS grant_year,
          UPPER(NULLIF(TRIM(assignee_country_mode), '')) AS assignee_country_mode,
          UPPER(NULLIF(TRIM(inventor_country_mode), '')) AS inventor_country_mode,
          TRY_CAST(num_claims AS DOUBLE) AS num_claims
        FROM read_parquet('{PATENT_BACKBONE.as_posix()}')
        WHERE patent_id IS NOT NULL
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE evidence AS
        SELECT
          UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
          TRY_CAST(benchmark_score AS DOUBLE) AS benchmark_score,
          TRY_CAST(benchmark_flag AS DOUBLE) AS benchmark_flag,
          TRY_CAST(benchmark_confidence AS DOUBLE) AS benchmark_confidence,
          TRY_CAST(data_collection_score AS DOUBLE) AS data_collection_score,
          TRY_CAST(empirical_analysis_score AS DOUBLE) AS empirical_analysis_score,
          TRY_CAST(experimental_validation_score AS DOUBLE) AS experimental_validation_score,
          TRY_CAST(measurement_instrumentation_score AS DOUBLE) AS measurement_instrumentation_score,
          TRY_CAST(data_quality_calibration_score AS DOUBLE) AS data_quality_calibration_score,
          TRY_CAST(ml_training_data_score AS DOUBLE) AS ml_training_data_score,
          TRY_CAST(dictionary_score AS DOUBLE) AS dictionary_score,
          TRY_CAST(metadata_score AS DOUBLE) AS metadata_score,
          TRY_CAST(text_supervised_score AS DOUBLE) AS text_supervised_score,
          TRY_CAST(semantic_lsa_score AS DOUBLE) AS semantic_lsa_score
        FROM read_csv_auto('{PATENT_EVIDENCE.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE app_year_map AS
        WITH raw AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            TRY_CAST(SUBSTR(TRIM(CAST(filing_date AS VARCHAR)), 1, 4) AS INTEGER) AS app_year
          FROM {sql_read(PATENTSVIEW_RAW / 'g_application.tsv.zip', cols=['patent_id', 'filing_date'])}
        )
        SELECT patent_id, MIN(app_year) AS app_year
        FROM raw
        WHERE patent_id IS NOT NULL AND app_year BETWEEN 1790 AND 2035
        GROUP BY 1
        """
    )

    gov_df = pd.read_csv(
        PATENTSVIEW_RAW / "g_gov_interest.tsv.zip",
        sep="\t",
        compression="zip",
        usecols=["patent_id", "gi_statement"],
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )
    gov_df["patent_id"] = gov_df["patent_id"].astype(str).str.strip().str.strip('"').str.upper()
    con.register("gov_interest_df", gov_df)

    gov_org_df = pd.read_csv(
        PATENTSVIEW_RAW / "g_gov_interest_org.tsv.zip",
        sep="\t",
        compression="zip",
        usecols=["patent_id", "fedagency_name", "level_one"],
        dtype=str,
        on_bad_lines="skip",
        low_memory=False,
    )
    gov_org_df["patent_id"] = gov_org_df["patent_id"].astype(str).str.strip().str.strip('"').str.upper()
    con.register("gov_org_df", gov_org_df)

    con.execute(
        """
        CREATE OR REPLACE TABLE gov_interest AS
        WITH g AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            NULLIF(TRIM(CAST(gi_statement AS VARCHAR)), '') AS gi_statement
          FROM gov_interest_df
        )
        SELECT
          g.patent_id,
          COUNT(*) AS gov_interest_statement_count,
          MAX(CASE WHEN gi_statement IS NOT NULL THEN 1 ELSE 0 END) AS is_publicly_funded
        FROM g
        JOIN universe u USING (patent_id)
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE gov_org_agg AS
        WITH g AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            NULLIF(TRIM(CAST(fedagency_name AS VARCHAR)), '') AS fedagency_name,
            LOWER(NULLIF(TRIM(CAST(level_one AS VARCHAR)), '')) AS level_one
          FROM gov_org_df
        ),
        c AS (
          SELECT patent_id, fedagency_name, COUNT(*) AS n
          FROM g
          WHERE fedagency_name IS NOT NULL
          GROUP BY 1,2
        ),
        top_org AS (
          SELECT patent_id, fedagency_name AS top_funding_agency
          FROM (
            SELECT
              patent_id,
              fedagency_name,
              n,
              ROW_NUMBER() OVER (PARTITION BY patent_id ORDER BY n DESC, fedagency_name ASC) AS rn
            FROM c
          )
          WHERE rn = 1
        )
        SELECT
          g.patent_id,
          COUNT(DISTINCT g.fedagency_name) AS gov_interest_org_count,
          MAX(CASE WHEN g.level_one LIKE '%defense%' OR LOWER(COALESCE(g.fedagency_name, '')) IN ('army', 'navy', 'air force') THEN 1 ELSE 0 END) AS has_dod_funding,
          MAX(CASE WHEN LOWER(COALESCE(g.fedagency_name, '')) LIKE '%nasa%' OR g.level_one LIKE '%nasa%' THEN 1 ELSE 0 END) AS has_nasa_funding,
          MAX(CASE WHEN LOWER(COALESCE(g.fedagency_name, '')) LIKE '%health%' OR g.level_one LIKE '%health%' THEN 1 ELSE 0 END) AS has_hhs_funding,
          MAX(CASE WHEN LOWER(COALESCE(g.fedagency_name, '')) LIKE '%energy%' OR g.level_one LIKE '%energy%' THEN 1 ELSE 0 END) AS has_doe_funding,
          MAX(CASE WHEN LOWER(COALESCE(g.fedagency_name, '')) LIKE '%science foundation%' OR LOWER(COALESCE(g.fedagency_name, '')) = 'nsf' THEN 1 ELSE 0 END) AS has_nsf_funding,
          t.top_funding_agency
        FROM g
        JOIN universe u USING (patent_id)
        LEFT JOIN top_org t USING (patent_id)
        GROUP BY 1, t.top_funding_agency
        """
    )

    science_regex = (
        r"doi\s*[:]|et\s+al|journal|vol\.|volume|pp\.|arxiv|pmid|"
        r"nature|science|ieee|proceedings|lancet|nejm|bioinformatics"
    )
    doi_regex = r"doi\s*[:]?10\."
    con.execute(
        f"""
        CREATE OR REPLACE TABLE other_ref_agg AS
        WITH refs AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            LOWER(COALESCE(CAST(other_reference_text AS VARCHAR), '')) AS ref_text
          FROM {sql_read(PATENTSVIEW_RAW / 'g_other_reference.tsv.zip', cols=['patent_id', 'other_reference_text'])}
        )
        SELECT
          r.patent_id,
          COUNT(*) AS other_reference_count,
          SUM(CASE WHEN regexp_matches(r.ref_text, '{science_regex}') THEN 1 ELSE 0 END) AS science_reference_count,
          SUM(CASE WHEN regexp_matches(r.ref_text, '{doi_regex}') THEN 1 ELSE 0 END) AS doi_reference_count
        FROM refs r
        JOIN universe u USING (patent_id)
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE citation_forward AS
        WITH c AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS citing_patent_id,
            UPPER(TRIM(CAST(citation_patent_id AS VARCHAR))) AS cited_patent_id,
            LOWER(TRIM(CAST(citation_category AS VARCHAR))) AS citation_category
          FROM {sql_read(PATENTSVIEW_RAW / 'g_us_patent_citation.tsv.zip', cols=['patent_id', 'citation_patent_id', 'citation_category'])}
          WHERE patent_id IS NOT NULL AND citation_patent_id IS NOT NULL
        )
        SELECT
          c.cited_patent_id AS patent_id,
          c.citing_patent_id,
          c.citation_category
        FROM c
        JOIN universe u ON c.cited_patent_id = u.patent_id
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE citation_backward AS
        WITH c AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS citing_patent_id,
            UPPER(TRIM(CAST(citation_patent_id AS VARCHAR))) AS cited_patent_id
          FROM {sql_read(PATENTSVIEW_RAW / 'g_us_patent_citation.tsv.zip', cols=['patent_id', 'citation_patent_id'])}
          WHERE patent_id IS NOT NULL AND citation_patent_id IS NOT NULL
        )
        SELECT
          c.citing_patent_id AS patent_id,
          c.cited_patent_id
        FROM c
        JOIN universe u ON c.citing_patent_id = u.patent_id
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE forward_metrics AS
        SELECT
          u.patent_id,
          COUNT(cf.citing_patent_id) AS forward_citation_count_all,
          COUNT(cf.citing_patent_id) FILTER (WHERE ay.app_year BETWEEN u.app_year AND u.app_year + 5) AS forward_citation_count_5y,
          COUNT(cf.citing_patent_id) FILTER (WHERE ay.app_year BETWEEN u.app_year AND u.app_year + 10) AS forward_citation_count_10y,
          COUNT(cf.citing_patent_id) FILTER (WHERE cf.citation_category = 'cited by examiner' AND ay.app_year BETWEEN u.app_year AND u.app_year + 10) AS forward_citation_examiner_10y,
          COUNT(cf.citing_patent_id) FILTER (WHERE cf.citation_category = 'cited by applicant' AND ay.app_year BETWEEN u.app_year AND u.app_year + 10) AS forward_citation_applicant_10y
        FROM universe u
        LEFT JOIN citation_forward cf ON u.patent_id = cf.patent_id
        LEFT JOIN app_year_map ay ON cf.citing_patent_id = ay.patent_id
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE backward_metrics AS
        SELECT
          patent_id,
          COUNT(*) AS backward_patent_citation_count
        FROM citation_backward
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE citation_forward_pairs AS
        SELECT DISTINCT patent_id, citing_patent_id
        FROM citation_forward
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE citing_patent_set AS
        SELECT DISTINCT citing_patent_id FROM citation_forward_pairs
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE citing_cpc AS
        WITH c AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS citing_patent_id,
            UPPER(TRIM(CAST(cpc_subclass AS VARCHAR))) AS cpc_subclass
          FROM {sql_read(PATENTSVIEW_RAW / 'g_cpc_current.tsv.zip', cols=['patent_id', 'cpc_subclass'])}
        )
        SELECT
          c.citing_patent_id,
          c.cpc_subclass,
          SUBSTR(c.cpc_subclass, 1, 4) AS cpc4
        FROM c
        JOIN citing_patent_set s ON c.citing_patent_id = s.citing_patent_id
        WHERE c.cpc_subclass IS NOT NULL AND LENGTH(c.cpc_subclass) >= 4
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE citation_breadth AS
        SELECT
          u.patent_id,
          COUNT(DISTINCT CASE WHEN ay.app_year BETWEEN u.app_year AND u.app_year + 10 THEN cc.cpc4 END) AS citing_cpc4_count_10y,
          COUNT(DISTINCT CASE WHEN ay.app_year BETWEEN u.app_year AND u.app_year + 10 THEN cc.cpc_subclass END) AS citing_cpc_subclass_count_10y
        FROM universe u
        LEFT JOIN citation_forward_pairs fp ON u.patent_id = fp.patent_id
        LEFT JOIN app_year_map ay ON fp.citing_patent_id = ay.patent_id
        LEFT JOIN citing_cpc cc ON fp.citing_patent_id = cc.citing_patent_id
        GROUP BY 1
        """
    )

    if list((REPO_ROOT / "datasets" / "compustat_patents_data").glob("staticTranche*.csv")):
        static_source = STATIC_GLOB
    else:
        static_source = STATIC_FALLBACK.as_posix()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE patent_gvkey AS
        WITH raw AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            TRY_CAST(appYear AS INTEGER) AS app_year,
            NULLIF(TRIM(COALESCE(gvkeyUO, gvkeyFR)), '') AS gvkey
          FROM read_csv_auto('{static_source}', HEADER=TRUE, ALL_VARCHAR=TRUE)
          WHERE patent_id IS NOT NULL
        ),
        ranked AS (
          SELECT
            patent_id,
            gvkey,
            app_year,
            ROW_NUMBER() OVER (
              PARTITION BY patent_id
              ORDER BY CASE WHEN gvkey IS NULL THEN 1 ELSE 0 END, app_year DESC NULLS LAST
            ) AS rn
          FROM raw
        )
        SELECT patent_id, gvkey, app_year
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
          TRY_CAST(employees AS DOUBLE) AS employees,
          TRY_CAST(revenue AS DOUBLE) AS revenue,
          TRY_CAST(rd_expense AS DOUBLE) AS rd_expense,
          TRY_CAST(capex AS DOUBLE) AS capex
        FROM read_csv_auto('{FIRM_YEAR.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        WHERE gvkey IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE firm_mode AS
        WITH c AS (
          SELECT gvkey, ROUND(employees) AS emp_bin, COUNT(*) AS n
          FROM firm_year
          WHERE employees IS NOT NULL
          GROUP BY 1,2
        ),
        r AS (
          SELECT
            gvkey,
            emp_bin,
            n,
            ROW_NUMBER() OVER (PARTITION BY gvkey ORDER BY n DESC, emp_bin ASC) AS rn
          FROM c
        )
        SELECT gvkey, emp_bin AS employees_mode
        FROM r
        WHERE rn = 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE citing_patent_size AS
        SELECT
          s.citing_patent_id,
          COALESCE(fy.employees, fm.employees_mode) AS employees
        FROM citing_patent_set s
        LEFT JOIN app_year_map ay ON s.citing_patent_id = ay.patent_id
        LEFT JOIN patent_gvkey pg ON s.citing_patent_id = pg.patent_id
        LEFT JOIN firm_year fy ON pg.gvkey = fy.gvkey AND ay.app_year = fy.year
        LEFT JOIN firm_mode fm ON pg.gvkey = fm.gvkey
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE citation_small_firm AS
        SELECT
          u.patent_id,
          COUNT(DISTINCT CASE WHEN ay.app_year BETWEEN u.app_year AND u.app_year + 10 THEN fp.citing_patent_id END) AS citing_patents_10y,
          COUNT(DISTINCT CASE WHEN ay.app_year BETWEEN u.app_year AND u.app_year + 10 AND cps.employees IS NOT NULL THEN fp.citing_patent_id END) AS citing_patents_with_size_10y,
          COUNT(DISTINCT CASE WHEN ay.app_year BETWEEN u.app_year AND u.app_year + 10 AND cps.employees < 500 THEN fp.citing_patent_id END) AS citing_small_firm_patents_10y
        FROM universe u
        LEFT JOIN citation_forward_pairs fp ON u.patent_id = fp.patent_id
        LEFT JOIN app_year_map ay ON fp.citing_patent_id = ay.patent_id
        LEFT JOIN citing_patent_size cps ON fp.citing_patent_id = cps.citing_patent_id
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE cpc_issue_intro AS
        WITH c AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            UPPER(TRIM(CAST(cpc_subclass AS VARCHAR))) AS cpc_subclass
          FROM {sql_read(PATENTSVIEW_RAW / 'g_cpc_at_issue.tsv.zip', cols=['patent_id', 'cpc_subclass'])}
          WHERE cpc_subclass IS NOT NULL
        )
        SELECT c.cpc_subclass, MIN(ay.app_year) AS cpc_intro_app_year
        FROM c
        JOIN app_year_map ay ON c.patent_id = ay.patent_id
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE universe_cpc_current AS
        WITH c AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            UPPER(TRIM(CAST(cpc_subclass AS VARCHAR))) AS cpc_subclass
          FROM {sql_read(PATENTSVIEW_RAW / 'g_cpc_current.tsv.zip', cols=['patent_id', 'cpc_subclass'])}
          WHERE cpc_subclass IS NOT NULL
        )
        SELECT c.patent_id, c.cpc_subclass
        FROM c
        JOIN universe u USING (patent_id)
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE cpc_ahead AS
        SELECT
          u.patent_id,
          MAX(GREATEST(ci.cpc_intro_app_year - u.app_year, 0)) AS years_ahead_of_cpc_intro,
          MAX(CASE WHEN ci.cpc_intro_app_year > u.app_year THEN 1 ELSE 0 END) AS opens_new_cpc_field_flag
        FROM universe u
        LEFT JOIN universe_cpc_current uc ON u.patent_id = uc.patent_id
        LEFT JOIN cpc_issue_intro ci ON uc.cpc_subclass = ci.cpc_subclass
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE uspc_patent AS
        WITH x AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            NULLIF(TRIM(CAST(uspc_mainclass_id AS VARCHAR)), '') AS uspc_mainclass_id
          FROM {sql_read(PATENTSVIEW_RAW / 'g_uspc_at_issue.tsv.zip', cols=['patent_id', 'uspc_mainclass_id'])}
          WHERE uspc_mainclass_id IS NOT NULL
        )
        SELECT x.patent_id, x.uspc_mainclass_id
        FROM x
        JOIN universe u USING (patent_id)
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE uspc_intro AS
        SELECT
          up.uspc_mainclass_id,
          MIN(u.app_year) AS uspc_mainclass_first_app_year
        FROM uspc_patent up
        JOIN universe u USING (patent_id)
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE uspc_features AS
        SELECT
          u.patent_id,
          MIN(ui.uspc_mainclass_first_app_year) AS patent_uspc_first_app_year_min,
          MAX(CASE WHEN u.app_year = ui.uspc_mainclass_first_app_year THEN 1 ELSE 0 END) AS is_first_gen_uspc_class
        FROM universe u
        LEFT JOIN uspc_patent up ON u.patent_id = up.patent_id
        LEFT JOIN uspc_intro ui ON up.uspc_mainclass_id = ui.uspc_mainclass_id
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE inventor_links AS
        WITH i AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            NULLIF(TRIM(CAST(inventor_id AS VARCHAR)), '') AS inventor_id
          FROM {sql_read(PATENTSVIEW_RAW / 'g_inventor_disambiguated.tsv.zip', cols=['patent_id', 'inventor_id'])}
        )
        SELECT i.patent_id, i.inventor_id
        FROM i
        JOIN universe u USING (patent_id)
        WHERE i.inventor_id IS NOT NULL
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE assignee_links AS
        WITH a AS (
          SELECT
            UPPER(TRIM(CAST(patent_id AS VARCHAR))) AS patent_id,
            NULLIF(TRIM(CAST(assignee_id AS VARCHAR)), '') AS assignee_id
          FROM {sql_read(PATENTSVIEW_RAW / 'g_assignee_disambiguated.tsv.zip', cols=['patent_id', 'assignee_id'])}
        )
        SELECT a.patent_id, a.assignee_id
        FROM a
        JOIN universe u USING (patent_id)
        WHERE a.assignee_id IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE inventor_prior AS
        WITH p AS (
          SELECT DISTINCT il.inventor_id, il.patent_id, u.app_year
          FROM inventor_links il
          JOIN universe u USING (patent_id)
          WHERE u.app_year IS NOT NULL
        )
        SELECT
          inventor_id,
          patent_id,
          COUNT(*) OVER (
            PARTITION BY inventor_id
            ORDER BY app_year, patent_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
          ) AS inventor_prior_patent_count
        FROM p
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE assignee_prior AS
        WITH p AS (
          SELECT DISTINCT al.assignee_id, al.patent_id, u.app_year
          FROM assignee_links al
          JOIN universe u USING (patent_id)
          WHERE u.app_year IS NOT NULL
        )
        SELECT
          assignee_id,
          patent_id,
          COUNT(*) OVER (
            PARTITION BY assignee_id
            ORDER BY app_year, patent_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
          ) AS assignee_prior_patent_count
        FROM p
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE inventor_prod AS
        SELECT
          patent_id,
          COUNT(*) AS inventor_count,
          AVG(inventor_prior_patent_count) AS inventor_productivity_mean_prior_count,
          MAX(inventor_prior_patent_count) AS inventor_productivity_max_prior_count
        FROM inventor_prior
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE assignee_prod AS
        SELECT
          patent_id,
          COUNT(*) AS assignee_count,
          AVG(assignee_prior_patent_count) AS assignee_productivity_mean_prior_count,
          MAX(assignee_prior_patent_count) AS assignee_productivity_max_prior_count
        FROM assignee_prior
        GROUP BY 1
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE patent_dyevre_style AS
        SELECT
          u.patent_id,
          u.app_year,
          u.grant_year,
          u.assignee_country_mode,
          u.inventor_country_mode,
          u.num_claims,
          LN(COALESCE(u.num_claims, 0) + 1) AS ln_num_claims_plus1,

          e.benchmark_score,
          e.benchmark_flag,
          e.benchmark_confidence,
          e.data_collection_score,
          e.empirical_analysis_score,
          e.experimental_validation_score,
          e.measurement_instrumentation_score,
          e.data_quality_calibration_score,
          e.ml_training_data_score,
          e.dictionary_score,
          e.metadata_score,
          e.text_supervised_score,
          e.semantic_lsa_score,

          COALESCE(gi.is_publicly_funded, 0) AS is_publicly_funded,
          COALESCE(gi.gov_interest_statement_count, 0) AS gov_interest_statement_count,
          COALESCE(go.gov_interest_org_count, 0) AS gov_interest_org_count,
          COALESCE(go.has_dod_funding, 0) AS has_dod_funding,
          COALESCE(go.has_nasa_funding, 0) AS has_nasa_funding,
          COALESCE(go.has_hhs_funding, 0) AS has_hhs_funding,
          COALESCE(go.has_doe_funding, 0) AS has_doe_funding,
          COALESCE(go.has_nsf_funding, 0) AS has_nsf_funding,
          go.top_funding_agency,

          COALESCE(r.other_reference_count, 0) AS other_reference_count,
          COALESCE(r.science_reference_count, 0) AS science_reference_count,
          COALESCE(r.doi_reference_count, 0) AS doi_reference_count,
          COALESCE(r.science_reference_count, 0) / NULLIF(COALESCE(r.other_reference_count, 0), 0) AS science_reference_share,

          COALESCE(b.backward_patent_citation_count, 0) AS backward_patent_citation_count,
          COALESCE(f.forward_citation_count_all, 0) AS forward_citation_count_all,
          COALESCE(f.forward_citation_count_5y, 0) AS forward_citation_count_5y,
          COALESCE(f.forward_citation_count_10y, 0) AS forward_citation_count_10y,
          COALESCE(f.forward_citation_examiner_10y, 0) AS forward_citation_examiner_10y,
          COALESCE(f.forward_citation_applicant_10y, 0) AS forward_citation_applicant_10y,
          LN(COALESCE(f.forward_citation_count_10y, 0) + 1) AS ln_forward_citation_count_10y_plus1,

          COALESCE(cb.citing_cpc4_count_10y, 0) AS citing_cpc4_count_10y,
          COALESCE(cb.citing_cpc_subclass_count_10y, 0) AS citing_cpc_subclass_count_10y,

          COALESCE(sf.citing_patents_10y, 0) AS citing_patents_10y,
          COALESCE(sf.citing_patents_with_size_10y, 0) AS citing_patents_with_size_10y,
          COALESCE(sf.citing_small_firm_patents_10y, 0) AS citing_small_firm_patents_10y,
          COALESCE(sf.citing_small_firm_patents_10y, 0) / NULLIF(COALESCE(sf.citing_patents_with_size_10y, 0), 0) AS small_firm_citation_share_10y_linked,
          COALESCE(sf.citing_patents_with_size_10y, 0) / NULLIF(COALESCE(sf.citing_patents_10y, 0), 0) AS small_firm_citation_size_coverage_10y,

          COALESCE(ca.opens_new_cpc_field_flag, 0) AS opens_new_cpc_field_flag,
          COALESCE(ca.years_ahead_of_cpc_intro, 0) AS years_ahead_of_cpc_intro,
          COALESCE(uf.is_first_gen_uspc_class, 0) AS is_first_gen_uspc_class,
          uf.patent_uspc_first_app_year_min,

          COALESCE(ip.inventor_count, 0) AS inventor_count,
          COALESCE(ip.inventor_productivity_mean_prior_count, 0) AS inventor_productivity_mean_prior_count,
          COALESCE(ip.inventor_productivity_max_prior_count, 0) AS inventor_productivity_max_prior_count,
          COALESCE(ap.assignee_count, 0) AS assignee_count,
          COALESCE(ap.assignee_productivity_mean_prior_count, 0) AS assignee_productivity_mean_prior_count,
          COALESCE(ap.assignee_productivity_max_prior_count, 0) AS assignee_productivity_max_prior_count
        FROM universe u
        LEFT JOIN evidence e USING (patent_id)
        LEFT JOIN gov_interest gi USING (patent_id)
        LEFT JOIN gov_org_agg go USING (patent_id)
        LEFT JOIN other_ref_agg r USING (patent_id)
        LEFT JOIN backward_metrics b USING (patent_id)
        LEFT JOIN forward_metrics f USING (patent_id)
        LEFT JOIN citation_breadth cb USING (patent_id)
        LEFT JOIN citation_small_firm sf USING (patent_id)
        LEFT JOIN cpc_ahead ca USING (patent_id)
        LEFT JOIN uspc_features uf USING (patent_id)
        LEFT JOIN inventor_prod ip USING (patent_id)
        LEFT JOIN assignee_prod ap USING (patent_id)
        """
    )

    patent_out_csv = DATA_OUT / "patent_dyevre_style_variables.csv.gz"
    patent_out_parq = DATA_OUT / "patent_dyevre_style_variables.parquet"
    con.execute(f"COPY patent_dyevre_style TO '{patent_out_csv.as_posix()}' (HEADER, DELIMITER ',', COMPRESSION GZIP)")
    con.execute(f"COPY patent_dyevre_style TO '{patent_out_parq.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    con.execute(
        """
        CREATE OR REPLACE TABLE country_year_dyevre AS
        SELECT
          assignee_country_mode AS iso2c,
          app_year AS year,
          COUNT(*) AS patent_count,
          AVG(benchmark_score) AS benchmark_score_mean,
          AVG(COALESCE(benchmark_flag, 0)) AS empirical_driven_share,
          AVG(is_publicly_funded) AS public_funded_share,
          AVG(science_reference_share) AS science_reference_share_mean,
          AVG(ln_forward_citation_count_10y_plus1) AS ln_forward_citation_10y_plus1_mean,
          AVG(citing_cpc4_count_10y) AS citing_cpc4_count_10y_mean,
          AVG(opens_new_cpc_field_flag) AS opens_new_cpc_field_share,
          AVG(years_ahead_of_cpc_intro) AS years_ahead_cpc_mean
        FROM patent_dyevre_style
        WHERE assignee_country_mode IS NOT NULL AND app_year IS NOT NULL
        GROUP BY 1,2
        """
    )
    con.execute(
        f"COPY country_year_dyevre TO '{(DATA_OUT / 'country_year_dyevre_style_variables.csv').as_posix()}' (HEADER, DELIMITER ',')"
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE firm_patent_agg AS
        WITH links AS (
          SELECT
            pg.gvkey,
            p.app_year AS year,
            p.patent_id,
            p.benchmark_score,
            p.benchmark_flag,
            p.is_publicly_funded,
            p.science_reference_share,
            p.forward_citation_count_10y,
            p.citing_cpc4_count_10y,
            p.opens_new_cpc_field_flag
          FROM patent_dyevre_style p
          JOIN patent_gvkey pg ON p.patent_id = pg.patent_id
          WHERE pg.gvkey IS NOT NULL AND p.app_year IS NOT NULL
        )
        SELECT
          gvkey,
          year,
          COUNT(*) AS patent_count_linked,
          AVG(benchmark_score) AS benchmark_score_mean,
          AVG(COALESCE(benchmark_flag, 0)) AS empirical_driven_share,
          AVG(is_publicly_funded) AS public_funded_share,
          AVG(science_reference_share) AS science_reference_share_mean,
          AVG(forward_citation_count_10y) AS forward_citation_count_10y_mean,
          AVG(citing_cpc4_count_10y) AS citing_cpc4_count_10y_mean,
          AVG(opens_new_cpc_field_flag) AS opens_new_cpc_field_share
        FROM links
        GROUP BY 1,2
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE firm_year_dyevre AS
        SELECT
          fy.gvkey,
          fy.year,
          fy.revenue,
          fy.capex,
          fy.rd_expense,
          fy.employees,
          fpa.patent_count_linked,
          fpa.benchmark_score_mean,
          fpa.empirical_driven_share,
          fpa.public_funded_share,
          fpa.science_reference_share_mean,
          fpa.forward_citation_count_10y_mean,
          fpa.citing_cpc4_count_10y_mean,
          fpa.opens_new_cpc_field_share,
          LN(GREATEST(COALESCE(fy.revenue, 0), 0) + 1) AS ln_sales_plus1,
          LN(GREATEST(COALESCE(fy.rd_expense, 0), 0) + 1) AS ln_rd_plus1,
          LN(GREATEST(COALESCE(fy.capex, 0), 0) + 1) AS ln_capital_proxy_plus1,
          LN(GREATEST(COALESCE(fpa.patent_count_linked, 0), 0) + 1) AS ln_patent_count_plus1,
          LN(GREATEST(COALESCE(fy.revenue / NULLIF(fy.employees, 0), 0), 0) + 1) AS ln_sales_per_worker_plus1
        FROM firm_year fy
        LEFT JOIN firm_patent_agg fpa ON fy.gvkey = fpa.gvkey AND fy.year = fpa.year
        """
    )
    con.execute(
        f"COPY firm_year_dyevre TO '{(DATA_OUT / 'firm_year_dyevre_style_variables.csv').as_posix()}' (HEADER, DELIMITER ',')"
    )

    stats = con.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM patent_dyevre_style) AS n_patent_rows,
          (SELECT COUNT(*) FROM country_year_dyevre) AS n_country_year_rows,
          (SELECT COUNT(*) FROM firm_year_dyevre) AS n_firm_year_rows,
          (SELECT AVG(is_publicly_funded) FROM patent_dyevre_style) AS public_funded_share,
          (SELECT AVG(COALESCE(benchmark_flag, 0)) FROM patent_dyevre_style) AS empirical_driven_share
        """
    ).fetchdf().iloc[0].to_dict()
    return {k: (float(v) if v is not None else None) for k, v in stats.items()}


def add_d5_changes() -> pd.DataFrame:
    path = DATA_OUT / "firm_year_dyevre_style_variables.csv"
    df = pd.read_csv(path)
    df = df.sort_values(["gvkey", "year"]).copy()
    for col in [
        "ln_sales_plus1",
        "ln_rd_plus1",
        "ln_capital_proxy_plus1",
        "ln_patent_count_plus1",
        "ln_sales_per_worker_plus1",
        "empirical_driven_share",
        "public_funded_share",
        "science_reference_share_mean",
    ]:
        dcol = f"d5_{col}"
        df[dcol] = df.groupby("gvkey")[col].shift(-5) - df[col]

    out = DATA_OUT / "firm_year_dyevre_style_variables_with_d5.csv"
    df.to_csv(out, index=False)
    return df


def build_variable_crosswalk() -> pd.DataFrame:
    rows = [
        {
            "paper_variable": "Patent is publicly-funded",
            "constructed_variable": "is_publicly_funded",
            "status": "Exact",
            "notes": "Directly from PatentsView government-interest statements.",
        },
        {
            "paper_variable": "Share of citations directed to scientific papers",
            "constructed_variable": "science_reference_share",
            "status": "Approximation",
            "notes": "Heuristic classification on non-patent references (DOI/journal patterns).",
        },
        {
            "paper_variable": "Log number of independent claims",
            "constructed_variable": "ln_num_claims_plus1",
            "status": "Approximation",
            "notes": "PatentsView has total claims in this pipeline, not independent-claim flags.",
        },
        {
            "paper_variable": "Probability of opening a new technological class",
            "constructed_variable": "opens_new_cpc_field_flag",
            "status": "Approximation",
            "notes": "CPC-based analogue: patent uses subclass whose at-issue introduction year is later than filing year.",
        },
        {
            "paper_variable": "Years ahead of creation of patent class",
            "constructed_variable": "years_ahead_of_cpc_intro",
            "status": "Approximation",
            "notes": "CPC-based analogue to USPC timing measure.",
        },
        {
            "paper_variable": "Log forward citations",
            "constructed_variable": "ln_forward_citation_count_10y_plus1",
            "status": "Exact",
            "notes": "From PatentsView US patent citations and filing-year windows.",
        },
        {
            "paper_variable": "Count of classes citing focal patent",
            "constructed_variable": "citing_cpc4_count_10y / citing_cpc_subclass_count_10y",
            "status": "Approximation",
            "notes": "Computed with CPC classes of citing patents over a 10-year window.",
        },
        {
            "paper_variable": "Share of citations from small firms (<500 employees)",
            "constructed_variable": "small_firm_citation_share_10y_linked",
            "status": "Approximation",
            "notes": "Computed on subset of citing patents linked to Compustat firms with observed employment.",
        },
        {
            "paper_variable": "Inventor productivity",
            "constructed_variable": "inventor_productivity_mean_prior_count",
            "status": "Approximation",
            "notes": "Prior patent counts by inventor within observed patent universe.",
        },
        {
            "paper_variable": "Assignee productivity",
            "constructed_variable": "assignee_productivity_mean_prior_count",
            "status": "Approximation",
            "notes": "Prior patent counts by assignee within observed patent universe.",
        },
        {
            "paper_variable": "Wage bill of inventors",
            "constructed_variable": "",
            "status": "Not available",
            "notes": "No inventor-level wage data in current repository sources.",
        },
        {
            "paper_variable": "Market value (firm-level)",
            "constructed_variable": "",
            "status": "Not available",
            "notes": "Not present in current firm-year enriched panel.",
        },
        {
            "paper_variable": "Cobb-Douglas / translog productivity",
            "constructed_variable": "ln_sales_per_worker_plus1 (proxy)",
            "status": "Approximation",
            "notes": "Proxy available, but full productivity estimation inputs are not all present.",
        },
        {
            "paper_variable": "Public/private spillovers and IV instruments",
            "constructed_variable": "",
            "status": "Not available",
            "notes": "Requires historical agency-shock and examiner-level instrument construction outside current pipeline.",
        },
    ]
    x = pd.DataFrame(rows)
    out_csv = DATA_OUT / "paper_variable_crosswalk.csv"
    x.to_csv(out_csv, index=False)
    return x


def build_descriptive_tables(con: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    idx_rows: list[dict[str, str]] = []

    t1 = con.execute(
        """
        SELECT
          COUNT(*) AS patent_observations,
          COUNT(DISTINCT assignee_country_mode) AS countries_assignee,
          MIN(app_year) AS app_year_min,
          MAX(app_year) AS app_year_max,
          AVG(CASE WHEN benchmark_score IS NULL THEN 1 ELSE 0 END) AS missing_benchmark_score_share,
          AVG(CASE WHEN science_reference_share IS NULL THEN 1 ELSE 0 END) AS missing_science_share,
          AVG(small_firm_citation_size_coverage_10y) AS avg_small_firm_citation_coverage_10y
        FROM patent_dyevre_style
        """
    ).fetchdf()
    t1_csv, t1_tex = save_table(
        t1,
        "table_dm1_dataset_overview",
        TAB_MAIN,
        [
            "missing_benchmark_score_share",
            "missing_science_share",
            "avg_small_firm_citation_coverage_10y",
        ],
        3,
    )
    idx_rows += [
        {
            "Section": "Main-paper candidates",
            "Filename": t1_csv.name,
            "Title": "Table DM1. Dataset coverage for Dyevre-style variables",
            "Sample": "Patent-level Dyevre-style file",
            "Variables": "Coverage and missingness metrics",
            "Interpretation": "Shows feasibility and limits of variable construction.",
        },
        {
            "Section": "Main-paper candidates",
            "Filename": t1_tex.name,
            "Title": "Table DM1. Dataset coverage (LaTeX)",
            "Sample": "Same as DM1",
            "Variables": "Same as DM1",
            "Interpretation": "Paper-ready table.",
        },
    ]

    t2 = con.execute(
        """
        SELECT
          'benchmark_score' AS variable, AVG(benchmark_score) AS mean, STDDEV_SAMP(benchmark_score) AS sd,
          QUANTILE_CONT(benchmark_score, 0.1) AS p10, QUANTILE_CONT(benchmark_score, 0.5) AS p50, QUANTILE_CONT(benchmark_score, 0.9) AS p90
        FROM patent_dyevre_style
        UNION ALL
        SELECT
          'science_reference_share', AVG(science_reference_share), STDDEV_SAMP(science_reference_share),
          QUANTILE_CONT(science_reference_share, 0.1), QUANTILE_CONT(science_reference_share, 0.5), QUANTILE_CONT(science_reference_share, 0.9)
        FROM patent_dyevre_style
        UNION ALL
        SELECT
          'ln_forward_citation_count_10y_plus1', AVG(ln_forward_citation_count_10y_plus1), STDDEV_SAMP(ln_forward_citation_count_10y_plus1),
          QUANTILE_CONT(ln_forward_citation_count_10y_plus1, 0.1), QUANTILE_CONT(ln_forward_citation_count_10y_plus1, 0.5), QUANTILE_CONT(ln_forward_citation_count_10y_plus1, 0.9)
        FROM patent_dyevre_style
        UNION ALL
        SELECT
          'citing_cpc4_count_10y', AVG(citing_cpc4_count_10y), STDDEV_SAMP(citing_cpc4_count_10y),
          QUANTILE_CONT(citing_cpc4_count_10y, 0.1), QUANTILE_CONT(citing_cpc4_count_10y, 0.5), QUANTILE_CONT(citing_cpc4_count_10y, 0.9)
        FROM patent_dyevre_style
        UNION ALL
        SELECT
          'years_ahead_of_cpc_intro', AVG(years_ahead_of_cpc_intro), STDDEV_SAMP(years_ahead_of_cpc_intro),
          QUANTILE_CONT(years_ahead_of_cpc_intro, 0.1), QUANTILE_CONT(years_ahead_of_cpc_intro, 0.5), QUANTILE_CONT(years_ahead_of_cpc_intro, 0.9)
        FROM patent_dyevre_style
        UNION ALL
        SELECT
          'is_publicly_funded', AVG(is_publicly_funded), STDDEV_SAMP(is_publicly_funded),
          QUANTILE_CONT(is_publicly_funded, 0.1), QUANTILE_CONT(is_publicly_funded, 0.5), QUANTILE_CONT(is_publicly_funded, 0.9)
        FROM patent_dyevre_style
        """
    ).fetchdf()
    t2_csv, t2_tex = save_table(t2, "table_dm2_patent_variable_summary", TAB_MAIN, ["mean", "sd", "p10", "p50", "p90"], 3)
    idx_rows += [
        {
            "Section": "Main-paper candidates",
            "Filename": t2_csv.name,
            "Title": "Table DM2. Summary statistics of constructed patent variables",
            "Sample": "Patent-level Dyevre-style file",
            "Variables": "Core Dyevre-style and empirical-driven measures",
            "Interpretation": "Benchmark moments for the combined variable family.",
        },
        {
            "Section": "Main-paper candidates",
            "Filename": t2_tex.name,
            "Title": "Table DM2. Summary statistics (LaTeX)",
            "Sample": "Same as DM2",
            "Variables": "Same as DM2",
            "Interpretation": "Paper-ready table.",
        },
    ]

    t3 = pd.read_csv(DATA_OUT / "paper_variable_crosswalk.csv")
    t3_csv, t3_tex = save_table(t3, "table_da1_paper_variable_crosswalk", TAB_APP, None, 3)
    idx_rows += [
        {
            "Section": "Appendix descriptives",
            "Filename": t3_csv.name,
            "Title": "Table DA1. Dyevre paper variable crosswalk",
            "Sample": "Conceptual variable mapping",
            "Variables": "Paper variable, constructed field, status, notes",
            "Interpretation": "Clarifies exact vs approximate vs unavailable constructs.",
        },
        {
            "Section": "Appendix descriptives",
            "Filename": t3_tex.name,
            "Title": "Table DA1. Variable crosswalk (LaTeX)",
            "Sample": "Same as DA1",
            "Variables": "Same as DA1",
            "Interpretation": "Paper-ready table.",
        },
    ]

    return idx_rows


def fig_score_distribution(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        SELECT benchmark_score, is_publicly_funded
        FROM patent_dyevre_style
        WHERE benchmark_score IS NOT NULL
        USING SAMPLE 300000 ROWS
        """
    ).fetchdf()
    d["Funding"] = np.where(d["is_publicly_funded"] == 1, "Publicly funded", "Not publicly funded")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    sns.kdeplot(data=d, x="benchmark_score", hue="Funding", common_norm=False, fill=True, alpha=0.2, linewidth=1.8, ax=ax)
    ax.set_title("Figure DM1. Distribution of Empirical-Driven Score by Public Funding")
    ax.set_xlabel("Empirical/data-driven benchmark score")
    ax.set_ylabel("Density")
    return fig


def fig_time_trends(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        SELECT
          app_year AS year,
          is_publicly_funded,
          COUNT(*) AS patent_count,
          AVG(COALESCE(benchmark_flag, 0)) AS empirical_share,
          AVG(benchmark_score) AS benchmark_score_mean
        FROM patent_dyevre_style
        WHERE app_year BETWEEN 1976 AND 2022
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    d["Funding"] = np.where(d["is_publicly_funded"] == 1, "Publicly funded", "Not publicly funded")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.lineplot(data=d, x="year", y="empirical_share", hue="Funding", ax=axes[0], linewidth=2)
    axes[0].set_title("Empirical-Driven Share Over Time")
    axes[0].set_xlabel("Application year")
    axes[0].set_ylabel("Share of patents")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False)

    sns.lineplot(data=d, x="year", y="benchmark_score_mean", hue="Funding", ax=axes[1], linewidth=2)
    axes[1].set_title("Mean Benchmark Score Over Time")
    axes[1].set_xlabel("Application year")
    axes[1].set_ylabel("Mean benchmark score")
    axes[1].legend(frameon=False)

    fig.suptitle("Figure DM2. Public vs Non-Public Patent Trends in Empirical-Driven Innovation", y=1.03)
    return fig


def fig_decile_relationships(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        WITH x AS (
          SELECT
            benchmark_score,
            science_reference_share,
            forward_citation_count_10y,
            citing_cpc4_count_10y,
            is_publicly_funded,
            NTILE(10) OVER (ORDER BY benchmark_score) AS decile
          FROM patent_dyevre_style
          WHERE benchmark_score IS NOT NULL
        )
        SELECT
          decile,
          AVG(science_reference_share) AS science_share_mean,
          AVG(forward_citation_count_10y) AS fwd10_mean,
          AVG(citing_cpc4_count_10y) AS citing_class_mean,
          AVG(is_publicly_funded) AS public_share_mean
        FROM x
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    axes = axes.ravel()
    sns.lineplot(data=d, x="decile", y="science_share_mean", marker="o", ax=axes[0], color="#0B7189")
    axes[0].set_title("Science-Reference Share")
    axes[0].set_xlabel("Empirical-driven score decile")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))

    sns.lineplot(data=d, x="decile", y="fwd10_mean", marker="o", ax=axes[1], color="#C73E1D")
    axes[1].set_title("Forward Citations (10y)")
    axes[1].set_xlabel("Empirical-driven score decile")
    axes[1].set_ylabel("Average count")

    sns.lineplot(data=d, x="decile", y="citing_class_mean", marker="o", ax=axes[2], color="#4B7F52")
    axes[2].set_title("Citing CPC4 Class Breadth (10y)")
    axes[2].set_xlabel("Empirical-driven score decile")
    axes[2].set_ylabel("Average distinct classes")

    sns.lineplot(data=d, x="decile", y="public_share_mean", marker="o", ax=axes[3], color="#8E5C42")
    axes[3].set_title("Public-Funded Share")
    axes[3].set_xlabel("Empirical-driven score decile")
    axes[3].yaxis.set_major_formatter(PercentFormatter(1))
    axes[3].set_ylabel("Share")

    fig.suptitle("Figure DM3. Paper-Style Outcomes Across Empirical-Driven Patent Deciles", y=1.02)
    return fig


def fig_agency_leaders(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        SELECT
          COALESCE(top_funding_agency, 'Unknown') AS agency,
          COUNT(*) AS patent_count,
          AVG(COALESCE(benchmark_flag, 0)) AS empirical_share,
          AVG(benchmark_score) AS benchmark_score_mean
        FROM patent_dyevre_style
        WHERE is_publicly_funded = 1
        GROUP BY 1
        HAVING COUNT(*) >= 100
        ORDER BY patent_count DESC
        LIMIT 20
        """
    ).fetchdf()
    d = d.sort_values("patent_count", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].barh(d["agency"], d["patent_count"], color="#0B7189")
    axes[0].set_title("Top Funding Agencies by Patent Count")
    axes[0].set_xlabel("Patents")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{int(v)}"))

    axes[1].barh(d["agency"], d["empirical_share"], color="#C73E1D")
    axes[1].set_title("Empirical-Driven Share by Agency")
    axes[1].set_xlabel("Share")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Figure DM4. Public Funding Agencies and Empirical-Driven Innovation", y=1.03)
    return fig


def fig_country_patterns(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        WITH c AS (
          SELECT
            assignee_country_mode AS iso2,
            COUNT(*) AS patent_count,
            AVG(COALESCE(benchmark_flag, 0)) AS empirical_share,
            AVG(is_publicly_funded) AS public_share
          FROM patent_dyevre_style
          WHERE assignee_country_mode IS NOT NULL
          GROUP BY 1
          HAVING COUNT(*) >= 5000
        )
        SELECT *
        FROM c
        """
    ).fetchdf()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        d["public_share"],
        d["empirical_share"],
        s=np.clip(np.sqrt(d["patent_count"]) * 2.3, 20, 260),
        c=d["patent_count"],
        cmap="viridis",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Public-funded patent share")
    ax.set_ylabel("Empirical-driven patent share")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Figure DM5. Cross-Country Public-Funding vs Empirical-Driven Intensity")
    for _, row in d.nlargest(10, "patent_count").iterrows():
        ax.text(row["public_share"], row["empirical_share"], row["iso2"], fontsize=8, ha="left", va="bottom")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Patent count")
    return fig


def fig_small_firm_citations(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        WITH x AS (
          SELECT
            benchmark_score,
            small_firm_citation_share_10y_linked,
            small_firm_citation_size_coverage_10y,
            NTILE(10) OVER (ORDER BY benchmark_score) AS decile
          FROM patent_dyevre_style
          WHERE benchmark_score IS NOT NULL
        )
        SELECT
          decile,
          AVG(small_firm_citation_share_10y_linked) AS small_share,
          AVG(small_firm_citation_size_coverage_10y) AS coverage
        FROM x
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    sns.lineplot(data=d, x="decile", y="small_share", marker="o", ax=axes[0], color="#0B7189")
    axes[0].set_title("Linked Small-Firm Citation Share (10y)")
    axes[0].set_xlabel("Empirical-driven score decile")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Share")

    sns.lineplot(data=d, x="decile", y="coverage", marker="o", ax=axes[1], color="#C73E1D")
    axes[1].set_title("Coverage of Firm-Size Link for Citing Patents")
    axes[1].set_xlabel("Empirical-driven score decile")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_ylabel("Share with observed size")
    fig.suptitle("Figure DA1. Small-Firm Spillover Proxy and Coverage", y=1.03)
    return fig


def fig_cpc_ahead(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        WITH x AS (
          SELECT
            benchmark_score,
            opens_new_cpc_field_flag,
            years_ahead_of_cpc_intro,
            NTILE(20) OVER (ORDER BY benchmark_score) AS ventile
          FROM patent_dyevre_style
          WHERE benchmark_score IS NOT NULL
        )
        SELECT
          ventile,
          AVG(opens_new_cpc_field_flag) AS open_share,
          AVG(years_ahead_of_cpc_intro) AS years_ahead_mean
        FROM x
        GROUP BY 1
        ORDER BY 1
        """
    ).fetchdf()
    fig, ax1 = plt.subplots(figsize=(7.3, 5.0))
    ax2 = ax1.twinx()
    ax1.plot(d["ventile"], d["open_share"], color="#0B7189", lw=2.0, marker="o", label="Open-share")
    ax2.plot(d["ventile"], d["years_ahead_mean"], color="#C73E1D", lw=1.8, linestyle="--", marker="s", label="Years-ahead")
    ax1.set_xlabel("Empirical-driven score ventile")
    ax1.set_ylabel("Share opening new CPC field")
    ax1.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_ylabel("Mean years ahead")
    ax1.set_title("Figure DA2. CPC-Based 'Ahead-of-Time' Patterns by Empirical Intensity")
    return fig


def fig_country_trends(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        WITH agg AS (
          SELECT
            assignee_country_mode AS iso2,
            app_year AS year,
            SUM(COALESCE(benchmark_flag, 0)) AS empirical_patents
          FROM patent_dyevre_style
          WHERE assignee_country_mode IS NOT NULL AND app_year BETWEEN 1976 AND 2022
          GROUP BY 1,2
        ),
        top AS (
          SELECT iso2
          FROM agg
          GROUP BY 1
          ORDER BY SUM(empirical_patents) DESC
          LIMIT 8
        )
        SELECT a.iso2, a.year, a.empirical_patents
        FROM agg a
        JOIN top t ON a.iso2 = t.iso2
        ORDER BY a.year, a.iso2
        """
    ).fetchdf()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    sns.lineplot(data=d, x="year", y="empirical_patents", hue="iso2", linewidth=2.0, ax=ax)
    ax.set_title("Figure DA3. Time Trends in Empirical-Driven Patents (Top Countries)")
    ax.set_xlabel("Application year")
    ax.set_ylabel("Empirical-driven patent count")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.legend(frameon=False, ncol=2, fontsize=8, title="")
    return fig


def fig_science_forward_scatter(con: duckdb.DuckDBPyConnection) -> plt.Figure:
    d = con.execute(
        """
        SELECT
          science_reference_share,
          ln_forward_citation_count_10y_plus1,
          benchmark_score
        FROM patent_dyevre_style
        WHERE science_reference_share IS NOT NULL
          AND ln_forward_citation_count_10y_plus1 IS NOT NULL
          AND benchmark_score IS NOT NULL
        USING SAMPLE 250000 ROWS
        """
    ).fetchdf()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    hb = ax.hexbin(
        d["science_reference_share"],
        d["ln_forward_citation_count_10y_plus1"],
        C=d["benchmark_score"],
        reduce_C_function=np.mean,
        gridsize=35,
        cmap="magma",
        mincnt=20,
    )
    ax.set_xlabel("Science-reference share")
    ax.set_ylabel("ln(1 + forward citations, 10y)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Figure DA4. Science Linkage, Impact, and Empirical-Driven Score")
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Mean benchmark score")
    return fig


def build_figures(con: duckdb.DuckDBPyConnection) -> list[dict[str, str]]:
    idx_rows: list[dict[str, str]] = []
    figs = [
        (fig_score_distribution(con), "figure_dm1_score_distribution_public_vs_private", FIG_MAIN, "Figure DM1. Distribution of empirical-driven score by public funding", "Patent sample (300k random draws)", "Benchmark score, public-funded flag", "Tests whether public-funded patents differ in empirical-data intensity."),
        (fig_time_trends(con), "figure_dm2_time_trends_public_private", FIG_MAIN, "Figure DM2. Time trends by public-funding status", "Patent-year 1976-2022", "Empirical share and benchmark means", "Shows dynamic divergence/convergence across funding source."),
        (fig_decile_relationships(con), "figure_dm3_decile_relationships", FIG_MAIN, "Figure DM3. Paper-style outcomes across empirical-score deciles", "All patents with benchmark score", "Science links, citations, breadth, public share", "Connects empirical-driven innovation to impact/spillover proxies."),
        (fig_agency_leaders(con), "figure_dm4_agency_leaders", FIG_MAIN, "Figure DM4. Agency heterogeneity in empirical-driven patenting", "Public-funded patents (agencies with >=100 patents)", "Counts and empirical share", "Identifies which agencies are most associated with empirical-driven innovation."),
        (fig_country_patterns(con), "figure_dm5_country_patterns", FIG_MAIN, "Figure DM5. Country pattern: public funding vs empirical intensity", "Countries with >=5,000 patents", "Public-funded share, empirical share", "Cross-country fact linking funding composition and empirical-driven innovation."),
        (fig_small_firm_citations(con), "figure_da1_small_firm_citation_proxy", FIG_APP, "Figure DA1. Small-firm citation proxy by empirical decile", "All patents with benchmark score", "Small-firm citation share + coverage", "Checks spillover distribution toward small firms under data constraints."),
        (fig_cpc_ahead(con), "figure_da2_cpc_ahead_proxy", FIG_APP, "Figure DA2. CPC ahead-of-time proxy by empirical intensity", "All patents with benchmark score", "Opens-new-field share and years-ahead mean", "Shows novelty field-opening relation with empirical-driven score."),
        (fig_country_trends(con), "figure_da3_country_trends_top", FIG_APP, "Figure DA3. Top-country empirical-driven trends", "Top 8 countries by cumulative empirical-driven patents", "Empirical-driven patent counts", "Highlights geographic/time concentration."),
        (fig_science_forward_scatter(con), "figure_da4_science_forward_scatter", FIG_APP, "Figure DA4. Science linkage, impact, and empirical score", "250k patent sample", "Science share, forward citations, benchmark score", "Triangulates fundamentalness, influence, and empirical-data intensity."),
    ]
    for fig, stem, out_dir, title, sample, vars_, interp in figs:
        png, pdf = save_fig(fig, stem, out_dir)
        sec = "Main-paper candidates" if out_dir == FIG_MAIN else "Appendix descriptives"
        idx_rows += [
            {"Section": sec, "Filename": png.name, "Title": title, "Sample": sample, "Variables": vars_, "Interpretation": interp},
            {"Section": sec, "Filename": pdf.name, "Title": f"{title} (PDF)", "Sample": sample, "Variables": vars_, "Interpretation": "Vector format."},
        ]
    return idx_rows


def write_index_and_memo(index_rows: list[dict[str, str]], stats: dict[str, float]) -> None:
    idx = pd.DataFrame(index_rows)
    idx_csv = INDEX_DIR / "dyevre_empirical_index.csv"
    idx_md = INDEX_DIR / "dyevre_empirical_index.md"
    idx.to_csv(idx_csv, index=False)
    idx_md.write_text(
        "# Dyevre-Style Empirical Package Index\n\n"
        + "\n".join(
            f"- **{r.Filename}** | {r.Title} | Sample: {r.Sample} | Variables: {r.Variables} | Interpretation: {r.Interpretation}"
            for _, r in idx.iterrows()
        )
    )

    memo = f"""# Dyevre-Style Variable and Graph Package Memo

## Objective
Construct as many variables as feasible from Dyevre (2024) using repository data plus PatentsView tables, and connect them to your empirical/data-driven patent measure.

## Core outputs
- Patent-level file: `datasets/dyevre_empirical/output/patent_dyevre_style_variables.csv.gz`
- Country-year file: `datasets/dyevre_empirical/output/country_year_dyevre_style_variables.csv`
- Firm-year file: `datasets/dyevre_empirical/output/firm_year_dyevre_style_variables_with_d5.csv`
- Variable crosswalk: `datasets/dyevre_empirical/output/paper_variable_crosswalk.csv`

## Coverage snapshot
- Patent observations: {int(stats.get("n_patent_rows", 0))}
- Country-year observations: {int(stats.get("n_country_year_rows", 0))}
- Firm-year observations: {int(stats.get("n_firm_year_rows", 0))}
- Public-funded share (patents): {stats.get("public_funded_share", float("nan")):.3f}
- Empirical-driven share (patents): {stats.get("empirical_driven_share", float("nan")):.3f}

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
"""
    (MEMO_DIR / "dyevre_empirical_memo.md").write_text(memo)


def main() -> None:
    ensure_dirs()
    set_style()
    ensure_raw_inputs()
    con = configure_duckdb()

    stats = build_tables(con)
    add_d5_changes()
    build_variable_crosswalk()

    index_rows: list[dict[str, str]] = []
    index_rows += build_descriptive_tables(con)
    index_rows += build_figures(con)

    write_index_and_memo(index_rows, stats)
    summary_path = OUT / "dyevre_empirical_run_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

