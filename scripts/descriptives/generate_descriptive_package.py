#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "country_year" / "social_returns_country_year_panel_with_patent_evidence.csv"
VARIANTS = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "country_year" / "patent_evidence_country_year_all_variants.csv"
PATENT_LEVEL = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "patent_evidence_patent_level.csv.gz"
METHOD_CMP = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "validation" / "candidate_method_comparison.csv"
TOP = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "diagnostics" / "top_patents_by_benchmark_score.csv"
BOTTOM = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "diagnostics" / "bottom_patents_by_benchmark_score.csv"
AMB = REPO_ROOT / "datasets" / "patent_evidence" / "output" / "diagnostics" / "most_ambiguous_patents.csv"

OUT = REPO_ROOT / "outputs" / "descriptives"
T_MAIN = OUT / "tables" / "main"
T_APP = OUT / "tables" / "appendix"
F_MAIN = OUT / "figures" / "main"
F_APP = OUT / "figures" / "appendix"
M_DIR = OUT / "memo"
I_DIR = OUT / "index"


def ensure_dirs() -> None:
    for p in [T_MAIN, T_APP, F_MAIN, F_APP, M_DIR, I_DIR]:
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


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    z = a / b.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def summarize(df: pd.DataFrame, cols: list[str], labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "Variable": labels.get(c, c),
                "N": int(s.shape[0]),
                "Mean": s.mean(),
                "SD": s.std(),
                "P10": s.quantile(0.1),
                "P25": s.quantile(0.25),
                "Median": s.quantile(0.5),
                "P75": s.quantile(0.75),
                "P90": s.quantile(0.9),
                "Min": s.min(),
                "Max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def var_decomp(df: pd.DataFrame, var: str) -> dict[str, float]:
    d = df[["iso3c", var]].copy()
    d[var] = pd.to_numeric(d[var], errors="coerce")
    d = d.dropna()
    overall = float(d[var].var(ddof=0))
    means = d.groupby("iso3c")[var].mean()
    between = float(means.var(ddof=0))
    d = d.join(means.rename("_m"), on="iso3c")
    within = float(((d[var] - d["_m"]) ** 2).mean())
    return {
        "N": int(len(d)),
        "Countries": int(d["iso3c"].nunique()),
        "Overall Var": overall,
        "Between Var": between,
        "Within Var": within,
        "Within/Overall": within / overall if overall > 0 else np.nan,
    }

def build_tables(panel: pd.DataFrame, a: pd.DataFrame, idx: list[dict[str, str]]) -> None:
    con = duckdb.connect()
    pstats = con.execute(
        f"""
        SELECT COUNT(*) n, COUNT(DISTINCT patent_id) nu, COUNT(DISTINCT assignee_country_mode) nc,
               MIN(TRY_CAST(app_year AS INTEGER)) y0, MAX(TRY_CAST(app_year AS INTEGER)) y1
        FROM read_csv_auto('{PATENT_LEVEL.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    ).fetchdf().iloc[0]
    v = pd.read_csv(VARIANTS)

    t1 = pd.DataFrame(
        [
            ["Merged country-year panel", "Country-year", len(panel), panel.iso3c.nunique(), panel.year.min(), panel.year.max(), "Full macro-policy merge"],
            ["Benchmark panel with pe_*", "Country-year", len(a), a.iso3c.nunique(), a.year.min(), a.year.max(), "Assignee x application-year"],
            ["Inventor-app variant", "Country-year", len(v[(v.assignment_basis == "inventor") & (v.year_basis == "app")]), v[(v.assignment_basis == "inventor") & (v.year_basis == "app")].iso3c.nunique(), v.year.min(), v.year.max(), "Alternative assignment"],
            ["Patent-level scored file", "Patent", int(pstats["n"]), int(pstats["nc"]), int(pstats["y0"]), int(pstats["y1"]), "Benchmark + components + CPC"],
            ["Benchmark with UNCTAD observed", "Country-year", int(a.unctad_privacy_and_data_protection_code.notna().sum()), int(a.loc[a.unctad_privacy_and_data_protection_code.notna(), "iso3c"].nunique()), int(a.loc[a.unctad_privacy_and_data_protection_code.notna(), "year"].min()), int(a.loc[a.unctad_privacy_and_data_protection_code.notna(), "year"].max()), "Regulation overlap"],
            ["Benchmark with ECIPE observed", "Country-year", int(a.ecipe_dte_n_measures_cum.notna().sum()), int(a.loc[a.ecipe_dte_n_measures_cum.notna(), "iso3c"].nunique()), int(a.loc[a.ecipe_dte_n_measures_cum.notna(), "year"].min()), int(a.loc[a.ecipe_dte_n_measures_cum.notna(), "year"].max()), "Policy-intensity overlap"],
        ],
        columns=["Dataset", "Unit", "Observations", "Countries", "Year start", "Year end", "Coverage note"],
    )
    t1["Merge coverage vs base panel"] = t1["Observations"] / len(panel)
    t1.loc[t1["Unit"] != "Country-year", "Merge coverage vs base panel"] = np.nan
    t1_csv, t1_tex = save_table(t1, "table_m1_dataset_overview", T_MAIN, ["Merge coverage vs base panel"], 3)
    idx += [
        {"Section": "Main-paper candidates", "Filename": t1_csv.name, "Title": "Table M1. Dataset overview and merge coverage", "Sample": "Panel + variants + patent-level", "Variables": "Obs/country/year coverage", "Interpretation": "Shows where analysis sample narrows."},
        {"Section": "Main-paper candidates", "Filename": t1_tex.name, "Title": "Table M1. Dataset overview and merge coverage (LaTeX)", "Sample": "Same as M1", "Variables": "Same as M1", "Interpretation": "Paper-ready table."},
    ]

    labels = {
        "pe_patent_count": "Total patents",
        "pe_data_driven_patent_count": "Data/evidence patent count",
        "pe_data_driven_patent_share": "Data/evidence patent share",
        "pe_benchmark_score_mean": "Mean benchmark score",
        "pe_data_patents_per_million": "Data/evidence patents per million",
        "pe_data_patents_per_trillion_gdp": "Data/evidence patents per trillion GDP",
        "pe_data_patents_per_billion_rd": "Data/evidence patents per USD 1bn R&D",
        "datagouv_privacy_level_ordinal": "Data.gouv privacy ordinal",
        "unctad_privacy_law": "UNCTAD privacy law indicator",
        "ecipe_dte_n_measures_cum": "ECIPE cumulative measures",
        "IT.NET.USER.ZS": "Internet users (%)",
        "NY.GDP.PCAP.KD": "GDP per capita (const. USD)",
    }
    t2 = summarize(a, list(labels.keys()), labels)
    t2_csv, t2_tex = save_table(t2, "table_m2_key_summary_stats", T_MAIN, ["Mean", "SD", "P10", "P25", "Median", "P75", "P90", "Min", "Max"], 3)
    idx += [
        {"Section": "Main-paper candidates", "Filename": t2_csv.name, "Title": "Table M2. Key summary statistics", "Sample": "Country-year benchmark", "Variables": "Outcomes/policy/controls", "Interpretation": "Core distributional facts for the paper."},
        {"Section": "Main-paper candidates", "Filename": t2_tex.name, "Title": "Table M2. Key summary statistics (LaTeX)", "Sample": "Same as M2", "Variables": "Same as M2", "Interpretation": "Paper-ready table."},
    ]

    c = (
        a.groupby("iso3c", as_index=False)
        .agg(
            total_data_patents=("pe_data_driven_patent_count", "sum"),
            total_patents=("pe_patent_count", "sum"),
            avg_data_share=("pe_data_driven_patent_share", "mean"),
            avg_data_patents_per_million=("pe_data_patents_per_million", "mean"),
            years_observed=("year", "nunique"),
        )
        .sort_values("total_data_patents", ascending=False)
    )
    c["global_data_patent_share"] = c["total_data_patents"] / c["total_data_patents"].sum()
    t3 = c.head(15)[["iso3c", "total_data_patents", "global_data_patent_share", "avg_data_share", "avg_data_patents_per_million", "years_observed"]]
    t3_csv, t3_tex = save_table(t3, "table_m3_top_country_contributors", T_MAIN, ["global_data_patent_share", "avg_data_share", "avg_data_patents_per_million"], 3)
    idx += [
        {"Section": "Main-paper candidates", "Filename": t3_csv.name, "Title": "Table M3. Top country contributors", "Sample": "Country totals over benchmark years", "Variables": "Levels + intensity", "Interpretation": "Separates scale from specialization."},
        {"Section": "Main-paper candidates", "Filename": t3_tex.name, "Title": "Table M3. Top country contributors (LaTeX)", "Sample": "Same as M3", "Variables": "Same as M3", "Interpretation": "Paper-ready table."},
    ]
    save_table(
        c[c.total_patents >= 5000]
        .sort_values("avg_data_share", ascending=False)
        .head(15)[["iso3c", "total_patents", "avg_data_share", "avg_data_patents_per_million", "years_observed"]],
        "table_a2_top_country_intensity",
        T_APP,
        ["avg_data_share", "avg_data_patents_per_million"],
        3,
    )

    dvars = {
        "pe_data_driven_patent_share": "Data/evidence share",
        "pe_data_driven_patent_count": "Data/evidence count",
        "pe_data_patents_per_million": "Data/evidence per million",
        "pe_benchmark_score_mean": "Mean benchmark score",
        "ecipe_dte_n_measures_cum": "ECIPE cumulative measures",
        "unctad_privacy_law": "UNCTAD privacy law",
    }
    dtab = pd.DataFrame([{**{"Variable": lab}, **var_decomp(a, var)} for var, lab in dvars.items()])
    a1_csv, a1_tex = save_table(
        dtab[["Variable", "N", "Countries", "Overall Var", "Between Var", "Within Var", "Within/Overall"]],
        "table_a1_within_between_variance",
        T_APP,
        ["Overall Var", "Between Var", "Within Var", "Within/Overall"],
        4,
    )
    idx += [
        {"Section": "Appendix descriptives", "Filename": a1_csv.name, "Title": "Table A1. Within-between variance decomposition", "Sample": "Country-year benchmark", "Variables": "Outcomes + policy", "Interpretation": "Panel identifying variation check."},
        {"Section": "Appendix descriptives", "Filename": a1_tex.name, "Title": "Table A1. Within-between variance decomposition (LaTeX)", "Sample": "Same as A1", "Variables": "Same as A1", "Interpretation": "Paper-ready table."},
    ]

    qtab = con.execute(
        f"""
        WITH b AS (
          SELECT TRY_CAST(benchmark_flag AS INTEGER) flag, TRY_CAST(num_claims AS DOUBLE) claims
          FROM read_csv_auto('{PATENT_LEVEL.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
          WHERE num_claims IS NOT NULL
        ), p AS (SELECT quantile_cont(claims, 0.9) q90 FROM b)
        SELECT
          CASE WHEN flag=1 THEN 'Data/evidence patent (flag=1)' ELSE 'Other patent (flag=0)' END AS group_name,
          COUNT(*) n_patents, AVG(claims) mean_claims, quantile_cont(claims,0.5) median_claims,
          AVG(CASE WHEN claims>=p.q90 THEN 1 ELSE 0 END) top_claim_decile_share
        FROM b CROSS JOIN p GROUP BY 1
        """
    ).fetchdf()
    a3_csv, a3_tex = save_table(
        qtab,
        "table_a3_claims_quality_proxy",
        T_APP,
        ["mean_claims", "median_claims", "top_claim_decile_share"],
        3,
    )
    idx += [
        {"Section": "Appendix descriptives", "Filename": a3_csv.name, "Title": "Table A3. Claims-based quality proxy", "Sample": "Patent-level non-missing claims", "Variables": "Claims moments by benchmark flag", "Interpretation": "Limited quality lens when citation data are absent."},
        {"Section": "Appendix descriptives", "Filename": a3_tex.name, "Title": "Table A3. Claims-based quality proxy (LaTeX)", "Sample": "Same as A3", "Variables": "Same as A3", "Interpretation": "Paper-ready table."},
    ]

    tval = pd.concat(
        [
            pd.read_csv(TOP).head(8).assign(Bucket="Top score"),
            pd.read_csv(BOTTOM).head(8).assign(Bucket="Bottom score"),
            pd.read_csv(AMB).head(8).assign(Bucket="Most ambiguous"),
        ],
        ignore_index=True,
    )[["Bucket", "patent_id", "benchmark_score", "dictionary_score", "metadata_score", "patent_title"]]
    a4_csv, a4_tex = save_table(
        tval,
        "table_a4_validation_examples",
        T_APP,
        ["benchmark_score", "dictionary_score", "metadata_score"],
        3,
    )
    idx += [
        {"Section": "Appendix descriptives", "Filename": a4_csv.name, "Title": "Table A4. Validation patent examples", "Sample": "Patent-level diagnostics", "Variables": "Top/bottom/ambiguous patents", "Interpretation": "Human-readable sanity check."},
        {"Section": "Appendix descriptives", "Filename": a4_tex.name, "Title": "Table A4. Validation patent examples (LaTeX)", "Sample": "Same as A4", "Variables": "Same as A4", "Interpretation": "Paper-ready table."},
    ]

    m = pd.read_csv(METHOD_CMP).rename(columns={"method": "Method", "auc": "AUC vs weak-label holdout", "best_f1": "Best F1", "best_threshold": "Best threshold"})
    a5_csv, a5_tex = save_table(m, "table_a5_method_comparison", T_APP, ["AUC vs weak-label holdout", "Best F1", "Best threshold"], 4)
    idx += [
        {"Section": "Appendix descriptives", "Filename": a5_csv.name, "Title": "Table A5. Candidate method comparison", "Sample": "Weak-label holdout", "Variables": "AUC/F1/threshold", "Interpretation": "Documents benchmark choice."},
        {"Section": "Appendix descriptives", "Filename": a5_tex.name, "Title": "Table A5. Candidate method comparison (LaTeX)", "Sample": "Same as A5", "Variables": "Same as A5", "Interpretation": "Paper-ready table."},
    ]

def build_figures(a: pd.DataFrame, idx: list[dict[str, str]]) -> None:
    y = a.groupby("year", as_index=False).agg(total=("pe_patent_count", "sum"), data=("pe_data_driven_patent_count", "sum")).sort_values("year")
    y["share"] = y["data"] / y["total"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax[0].plot(y.year, y.total, color="#6B7280", lw=2, label="Total patents")
    ax[0].plot(y.year, y.data, color="#0B7189", lw=2.2, label="Data/evidence patents")
    ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax[0].set_title("Global patent levels")
    ax[0].set_xlabel("Year")
    ax[0].legend(frameon=False)
    ax[1].plot(y.year, y.share, color="#C73E1D", lw=2.2)
    ax[1].set_title("Global data/evidence share")
    ax[1].set_xlabel("Year")
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Figure M1. Time-Series Evolution of Data/Evidence-Driven Innovation")
    p, q = save_fig(fig, "figure_m1_global_trends", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M1. Global trends", "Sample": "Benchmark country-year aggregated to year", "Variables": "Total/data patents and share", "Interpretation": "Core trend fact."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M1. Global trends (PDF)", "Sample": "Same as M1", "Variables": "Same as M1", "Interpretation": "Vector format."},
    ]

    c = a.groupby("iso3c", as_index=False)["pe_data_driven_patent_count"].sum().sort_values("pe_data_driven_patent_count", ascending=False)
    c["x"] = np.arange(1, len(c) + 1) / len(c)
    c["y"] = c.pe_data_driven_patent_count.cumsum() / c.pe_data_driven_patent_count.sum()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.plot(c.x, c.y, color="#0B7189", lw=2.3, label="Observed")
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1.5, label="Equal-share")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Cumulative share of countries")
    ax.set_ylabel("Cumulative share of data/evidence patents")
    ax.set_title("Figure M2. Cross-Country Concentration")
    ax.legend(frameon=False, loc="lower right")
    p, q = save_fig(fig, "figure_m2_country_concentration", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M2. Country concentration", "Sample": "Country totals over benchmark years", "Variables": "Cumulative shares", "Interpretation": "Concentration and skewness."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M2. Country concentration (PDF)", "Sample": "Same as M2", "Variables": "Same as M2", "Interpretation": "Vector format."},
    ]

    by_c = a.groupby("iso3c", as_index=False).agg(total_data=("pe_data_driven_patent_count", "sum"), total_pat=("pe_patent_count", "sum"), avg_share=("pe_data_driven_patent_share", "mean"))
    left = by_c.sort_values("total_data", ascending=False).head(10).sort_values("total_data")
    right = by_c[by_c.total_pat >= 5000].sort_values("avg_share", ascending=False).head(10).sort_values("avg_share")
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.8))
    ax[0].barh(left.iso3c, left.total_data, color="#0B7189")
    ax[0].set_title("Top by total data/evidence patents")
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k"))
    ax[1].barh(right.iso3c, right.avg_share, color="#C73E1D")
    ax[1].set_title("Top by average data/evidence share")
    ax[1].xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Figure M3. Levels vs Intensity")
    p, q = save_fig(fig, "figure_m3_levels_vs_intensity", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M3. Levels vs intensity", "Sample": "Country totals/means", "Variables": "Total counts and shares", "Interpretation": "Scale vs intensity contrast."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M3. Levels vs intensity (PDF)", "Sample": "Same as M3", "Variables": "Same as M3", "Interpretation": "Vector format."},
    ]
    d = a.copy()
    d["income"] = d["wb_income_level"].fillna("Unknown")
    keep = ["High income", "Upper middle income", "Lower middle income", "Low income"]
    d = d[d.income.isin(keep)]
    g = d.groupby(["year", "income"], as_index=False).agg(data=("pe_data_driven_patent_count", "sum"), total=("pe_patent_count", "sum"), med_pm=("pe_data_patents_per_million", "median"))
    g["share"] = g["data"] / g["total"]
    pal = {"High income": "#0B7189", "Upper middle income": "#F4A259", "Lower middle income": "#7C9885", "Low income": "#C73E1D"}
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.4))
    for k in keep:
        x = g[g.income == k]
        ax[0].plot(x.year, x.share, lw=2, label=k, color=pal[k])
        ax[1].plot(x.year, x.med_pm, lw=2, label=k, color=pal[k])
    ax[0].set_title("Weighted share by income group")
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].legend(frameon=False)
    ax[1].set_title("Median per-million intensity")
    fig.suptitle("Figure M4. Income-Group Heterogeneity")
    p, q = save_fig(fig, "figure_m4_income_group_heterogeneity", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M4. Income-group heterogeneity", "Sample": "Benchmark sample with income groups", "Variables": "Shares and per-capita intensity", "Interpretation": "Cross-development differences."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M4. Income-group heterogeneity (PDF)", "Sample": "Same as M4", "Variables": "Same as M4", "Interpretation": "Vector format."},
    ]

    con = duckdb.connect()
    tech = con.execute(
        f"""
        WITH b AS (
          SELECT TRY_CAST(benchmark_flag AS INTEGER) flag, TRY_CAST(has_measurement_cpc AS INTEGER) m,
                 TRY_CAST(has_testing_cpc AS INTEGER) t, TRY_CAST(has_ml_cpc AS INTEGER) ml,
                 TRY_CAST(has_data_processing_cpc AS INTEGER) dp, TRY_CAST(cpc_total_count AS INTEGER) n
          FROM read_csv_auto('{PATENT_LEVEL.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
          WHERE cpc_total_count IS NOT NULL
        )
        SELECT CASE
                 WHEN m=1 AND ml=1 THEN 'Measurement + ML'
                 WHEN m=1 THEN 'Measurement only'
                 WHEN ml=1 THEN 'ML only'
                 WHEN t=1 THEN 'Testing only'
                 WHEN dp=1 THEN 'Data-processing only'
                 ELSE 'Other/none'
               END grp,
               COUNT(*) n_patents, AVG(flag) AS share_flag
        FROM b GROUP BY 1 ORDER BY n_patents DESC
        """
    ).fetchdf().sort_values("n_patents")
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax[0].barh(tech.grp, tech.n_patents, color="#6B7280")
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax[0].set_title("Patent counts by technology group")
    ax[1].barh(tech.grp, tech.share_flag, color="#0B7189")
    ax[1].xaxis.set_major_formatter(PercentFormatter(1))
    ax[1].set_title("Data/evidence share by technology group")
    fig.suptitle("Figure M5. Technology Composition")
    p, q = save_fig(fig, "figure_m5_technology_composition", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M5. Technology composition", "Sample": "Patent-level with CPC flags", "Variables": "Counts and shares by CPC group", "Interpretation": "Shows concept extends beyond generic digital classes."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M5. Technology composition (PDF)", "Sample": "Same as M5", "Variables": "Same as M5", "Interpretation": "Vector format."},
    ]

    r = a.copy()
    r["datagouv_label"] = pd.to_numeric(r["datagouv_privacy_level_ordinal"], errors="coerce").round().astype("Int64").map({1: "Level 1", 2: "Level 2", 3: "Level 3", 4: "Level 4", 5: "Level 5 (EU/EEA)"})
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    sns.boxplot(data=r.dropna(subset=["datagouv_label", "pe_data_driven_patent_share"]), x="datagouv_label", y="pe_data_driven_patent_share", ax=ax[0], color="#7C9885", fliersize=1.5)
    ax[0].tick_params(axis="x", rotation=25)
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].set_title("Share by privacy regime (data.gouv)")
    s = r.dropna(subset=["ecipe_dte_n_measures_cum", "pe_data_driven_patent_share"]).copy()
    if not s.empty:
        s["bin"] = pd.qcut(s["ecipe_dte_n_measures_cum"], q=min(15, s["ecipe_dte_n_measures_cum"].nunique()), duplicates="drop")
        b = s.groupby("bin", as_index=False, observed=False).agg(
            x=("ecipe_dte_n_measures_cum", "mean"),
            y=("pe_data_driven_patent_share", "mean"),
            n=("iso3c", "size"),
        )
        ax[1].scatter(b.x, b.y, s=15 + 2 * np.sqrt(b.n), color="#0B7189")
        z = np.polyfit(b.x, b.y, 1)
        xl = np.linspace(b.x.min(), b.x.max(), 100)
        ax[1].plot(xl, z[0] * xl + z[1], "--", color="#C73E1D", lw=1.8)
    ax[1].yaxis.set_major_formatter(PercentFormatter(1))
    ax[1].set_title("Binned relation with ECIPE intensity")
    ax[1].set_xlabel("ECIPE cumulative measures")
    fig.suptitle("Figure M6. Regulation Gradients (Descriptive)")
    p, q = save_fig(fig, "figure_m6_regulation_gradients", F_MAIN)
    idx += [
        {"Section": "Main-paper candidates", "Filename": p.name, "Title": "Figure M6. Regulation gradients", "Sample": "Benchmark sample (ECIPE overlap for right panel)", "Variables": "Share vs data.gouv and ECIPE", "Interpretation": "Motivates policy-side empirical design."},
        {"Section": "Main-paper candidates", "Filename": q.name, "Title": "Figure M6. Regulation gradients (PDF)", "Sample": "Same as M6", "Variables": "Same as M6", "Interpretation": "Vector format."},
    ]
    v = pd.read_csv(VARIANTS)
    ass = v[(v.assignment_basis == "assignee") & (v.year_basis == "app")][["iso3c", "year", "data_driven_patent_share"]].rename(columns={"data_driven_patent_share": "ass"})
    inv = v[(v.assignment_basis == "inventor") & (v.year_basis == "app")][["iso3c", "year", "data_driven_patent_share"]].rename(columns={"data_driven_patent_share": "inv"})
    m = ass.merge(inv, on=["iso3c", "year"])
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.7))
    hb = ax[0].hexbin(m.ass, m.inv, gridsize=35, cmap="Blues", mincnt=1)
    lim = [0, max(m.ass.max(), m.inv.max()) * 1.02]
    ax[0].plot(lim, lim, "--", color="#C73E1D")
    ax[0].set_xlim(lim)
    ax[0].set_ylim(lim)
    ax[0].xaxis.set_major_formatter(PercentFormatter(1))
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].set_title("Assignee vs inventor share")
    fig.colorbar(hb, ax=ax[0], fraction=0.046, pad=0.04)
    diff = m.ass - m.inv
    ax[1].hist(diff, bins=40, color="#7C9885", edgecolor="white")
    ax[1].axvline(0, ls="--", color="#C73E1D")
    ax[1].xaxis.set_major_formatter(PercentFormatter(1))
    ax[1].set_title("Assignee minus inventor share")
    fig.suptitle("Figure A1. Assignment Basis Sensitivity")
    p, q = save_fig(fig, "figure_a1_assignee_vs_inventor", F_APP)
    idx += [
        {"Section": "Appendix descriptives", "Filename": p.name, "Title": "Figure A1. Assignee vs inventor comparison", "Sample": "Overlap of app-year variants", "Variables": "Country-year shares", "Interpretation": "Assesses benchmark assignment choice."},
        {"Section": "Appendix descriptives", "Filename": q.name, "Title": "Figure A1. Assignee vs inventor comparison (PDF)", "Sample": "Same as A1", "Variables": "Same as A1", "Interpretation": "Vector format."},
    ]

    h = con.execute(
        f"""
        WITH b AS (
          SELECT TRY_CAST(benchmark_score AS DOUBLE) score,
                 TRY_CAST(has_data_processing_cpc AS INTEGER) dp,
                 TRY_CAST(has_measurement_cpc AS INTEGER) m,
                 TRY_CAST(has_testing_cpc AS INTEGER) t,
                 TRY_CAST(has_ml_cpc AS INTEGER) ml,
                 TRY_CAST(has_statistics_cpc AS INTEGER) st,
                 TRY_CAST(has_bio_assay_cpc AS INTEGER) bio
          FROM read_csv_auto('{PATENT_LEVEL.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        )
        SELECT CASE
                 WHEN dp=1 AND m=0 AND t=0 AND ml=0 AND st=0 AND bio=0 THEN 'Generic digital-only CPC'
                 WHEN m=1 OR t=1 OR ml=1 OR st=1 OR bio=1 THEN 'Evidence-linked CPC'
                 ELSE 'Other patents'
               END grp,
               FLOOR(score*20)/20.0 bin,
               COUNT(*) n
        FROM b
        WHERE score IS NOT NULL
        GROUP BY 1,2
        ORDER BY 1,2
        """
    ).fetchdf()
    h["density"] = h["n"] / h.groupby("grp")["n"].transform("sum")
    piv = h.pivot(index="bin", columns="grp", values="density").fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for g, ccol in [("Other patents", "#7C9885"), ("Generic digital-only CPC", "#C73E1D"), ("Evidence-linked CPC", "#0B7189")]:
        if g in piv.columns:
            ax.plot(piv.index, piv[g], lw=2, label=g, color=ccol)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Benchmark score bin")
    ax.set_ylabel("Density within group")
    ax.set_title("Figure A2. Validation Score Distributions by Patent Type")
    ax.legend(frameon=False)
    p, q = save_fig(fig, "figure_a2_validation_score_distribution", F_APP)
    idx += [
        {"Section": "Appendix descriptives", "Filename": p.name, "Title": "Figure A2. Validation score distributions", "Sample": "Patent-level sample", "Variables": "Benchmark score by patent-type subsets", "Interpretation": "Checks benchmark is not just generic digital language."},
        {"Section": "Appendix descriptives", "Filename": q.name, "Title": "Figure A2. Validation score distributions (PDF)", "Sample": "Same as A2", "Variables": "Same as A2", "Interpretation": "Vector format."},
    ]


def write_index_and_memo(index_rows: list[dict[str, str]]) -> None:
    idx = pd.DataFrame(index_rows)
    csv = I_DIR / "descriptive_index.csv"
    md = I_DIR / "descriptive_index.md"
    idx.to_csv(csv, index=False)
    md.write_text(
        "# Descriptive Output Index\n\n"
        + "\n".join(
            f"- **{r.Filename}** | {r.Title} | Sample: {r.Sample} | Variables: {r.Variables} | Interpretation: {r.Interpretation}"
            for _, r in idx.iterrows()
        )
    )

    memo = """# Descriptive Memo

## Candidate descriptives considered
- Dataset overview and merge/missingness coverage
- Key variable summary statistics
- Global trends in levels and shares
- Country concentration and contribution rankings
- Levels-vs-intensity comparisons
- Normalized intensity metrics (population/GDP/R&D)
- Income-group heterogeneity
- Technology composition by CPC signal groups
- Regulation gradients (data.gouv and ECIPE)
- Within-vs-between variance decomposition
- Assignee-vs-inventor sensitivity
- Patent-level validation examples and score distributions
- Claims-based quality proxies
- Alternative measure comparison

## Selected benchmark package
- Main-paper: Tables M1-M3, Figures M1-M6.
- Appendix: Tables A1-A5, Figures A1-A2.

## Why these outputs
- They map directly to the core questions on variation, concentration, composition, regulation links, and panel credibility.
- They separate size effects from intensity effects.
- They include explicit validation diagnostics for the benchmark patent measure.

## Caveats
- Benchmark country-year coverage is narrower than the full merged panel.
- ECIPE policy-intensity overlap is partial.
- UNCTAD privacy code has limited within-sample switching.
- Citation-based quality outcomes are not currently merged; claims proxies are reported with caution.
"""
    (M_DIR / "descriptive_memo.md").write_text(memo)


def main() -> None:
    ensure_dirs()
    set_style()

    panel = pd.read_csv(PANEL)
    a = panel[panel["pe_patent_count"].notna()].copy()
    a["pe_patents_per_million"] = safe_div(pd.to_numeric(a["pe_patent_count"], errors="coerce"), pd.to_numeric(a["pwt_pop_millions"], errors="coerce"))
    a["pe_data_patents_per_million"] = safe_div(pd.to_numeric(a["pe_data_driven_patent_count"], errors="coerce"), pd.to_numeric(a["pwt_pop_millions"], errors="coerce"))
    gdp = pd.to_numeric(a["NY.GDP.MKTP.KD"], errors="coerce")
    a["pe_data_patents_per_trillion_gdp"] = safe_div(pd.to_numeric(a["pe_data_driven_patent_count"], errors="coerce"), gdp / 1e12)
    rd = gdp * pd.to_numeric(a["GB.XPD.RSDV.GD.ZS"], errors="coerce") / 100.0
    a["pe_data_patents_per_billion_rd"] = safe_div(pd.to_numeric(a["pe_data_driven_patent_count"], errors="coerce"), rd / 1e9)
    a["unctad_privacy_law"] = (pd.to_numeric(a["unctad_privacy_and_data_protection_code"], errors="coerce") == 1).astype(float)

    index_rows: list[dict[str, str]] = []
    build_tables(panel, a, index_rows)
    build_figures(a, index_rows)
    write_index_and_memo(index_rows)

    summary = {
        "analysis_rows": int(len(a)),
        "analysis_countries": int(a["iso3c"].nunique()),
        "analysis_year_min": int(a["year"].min()),
        "analysis_year_max": int(a["year"].max()),
        "output_root": str(OUT),
    }
    (OUT / "descriptive_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
