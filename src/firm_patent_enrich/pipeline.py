from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .external_data import (
    fetch_bea_gdp_by_industry_panel,
    fetch_bls_oes_panel,
    fetch_census_cbp_industry_panel,
    fetch_fred_macro_panel,
    load_bls_series_map,
    map_sic_to_naics_bucket,
)
from .linking import (
    LinkConfig,
    auto_link_gvkey_to_cik,
    fetch_sec_tickers,
    load_manual_link_file,
    load_ticker_hints_file,
)
from .patents import build_patent_firm_year_panel, build_primary_name_map, load_static_patents
from .sec import build_sec_firm_year_panel


def _clean_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if df.empty or "year" not in df.columns:
        return df
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()


def _merge_year_context(context_frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in context_frames if not f.empty and "year" in f.columns]
    if not frames:
        return pd.DataFrame(columns=["year"])
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="year", how="outer")
    return out


def _apply_manual_overrides(
    links: pd.DataFrame,
    manual_links: pd.DataFrame,
    sec_tickers: pd.DataFrame,
) -> pd.DataFrame:
    if manual_links.empty:
        return links

    manual = manual_links.copy().rename(columns={"cik": "manual_cik"})
    out = links.merge(manual[["gvkey", "manual_cik"]], on="gvkey", how="left")

    has_manual = out["manual_cik"].notna()
    out.loc[has_manual, "cik"] = out.loc[has_manual, "manual_cik"]
    out.loc[has_manual, "link_method"] = "manual_override"
    out.loc[has_manual, "link_score"] = 1.0
    out.loc[has_manual, "link_conflict"] = False

    sec_ref = sec_tickers[["cik", "ticker", "sec_name", "exchange"]].drop_duplicates(subset=["cik"])
    out = out.drop(columns=["manual_cik", "ticker", "sec_name", "exchange"], errors="ignore")
    out = out.merge(sec_ref, on="cik", how="left")

    return out


def run_pipeline(
    config: PipelineConfig,
    manual_link_file: Path | None = None,
    ticker_hints_file: Path | None = None,
    bls_series_file: Path | None = None,
) -> dict[str, Path]:
    config.ensure_dirs()

    static = load_static_patents(config.data_dir)
    patent_panel = build_patent_firm_year_panel(static)
    primary_names = build_primary_name_map(static)

    if config.max_firms is not None:
        top_firms = primary_names.sort_values("patent_count", ascending=False).head(config.max_firms)
        patent_panel = patent_panel[patent_panel["gvkey"].isin(set(top_firms["gvkey"]))]
        primary_names = top_firms

    sec_tickers = fetch_sec_tickers(config.cache_dir, user_agent=config.sec_user_agent)
    ticker_hints = load_ticker_hints_file(ticker_hints_file) if ticker_hints_file else None

    links, link_candidates, link_conflicts = auto_link_gvkey_to_cik(
        primary_names=primary_names,
        sec_tickers=sec_tickers,
        ticker_hints=ticker_hints,
        config=LinkConfig(
            fuzzy_threshold=config.link_fuzzy_threshold,
            score_gap_for_conflict=config.link_score_gap,
            top_n_candidates=config.link_top_n,
        ),
    )

    if manual_link_file is not None:
        manual_links = load_manual_link_file(manual_link_file)
        links = _apply_manual_overrides(links=links, manual_links=manual_links, sec_tickers=sec_tickers)

    linked = links[links["cik"].notna()].copy()
    ciks = sorted(set(linked["cik"].astype(str)))

    sec_panel = build_sec_firm_year_panel(
        ciks=ciks,
        cache_dir=config.cache_dir,
        user_agent=config.sec_user_agent,
        include_text_features=config.include_10k_text_features,
    )
    sec_panel = _clean_year_range(sec_panel, config.start_year, config.end_year)

    year_context_frames: list[pd.DataFrame] = []
    if config.include_fred_macro:
        year_context_frames.append(
            fetch_fred_macro_panel(
                cache_dir=config.cache_dir,
                start_year=config.start_year,
                end_year=config.end_year,
                api_key=config.fred_api_key,
            )
        )

    if config.include_bls_oes:
        bls_series_map = load_bls_series_map(bls_series_file)
        year_context_frames.append(
            fetch_bls_oes_panel(
                cache_dir=config.cache_dir,
                start_year=config.start_year,
                end_year=config.end_year,
                series_map=bls_series_map,
                api_key=config.bls_api_key,
            )
        )

    year_context = _merge_year_context(year_context_frames)

    if not sec_panel.empty:
        sec_panel["naics2_proxy"] = sec_panel["sic"].map(map_sic_to_naics_bucket)

        if config.include_census_context:
            census_panel = fetch_census_cbp_industry_panel(
                cache_dir=config.cache_dir,
                start_year=config.start_year,
                end_year=config.end_year,
            )
            sec_panel = sec_panel.merge(census_panel, on=["year", "naics2_proxy"], how="left")

        if config.include_bea_context:
            bea_panel = fetch_bea_gdp_by_industry_panel(
                cache_dir=config.cache_dir,
                start_year=config.start_year,
                end_year=config.end_year,
                api_key=config.bea_api_key,
            )
            sec_panel = sec_panel.merge(bea_panel, on=["year", "naics2_proxy"], how="left")

        if not year_context.empty:
            sec_panel = sec_panel.merge(year_context, on="year", how="left")

    patent_panel = _clean_year_range(patent_panel, config.start_year, config.end_year)

    if sec_panel.empty:
        sec_with_gvkey = linked[["gvkey", "cik", "link_method", "link_score", "link_conflict"]].drop_duplicates()
        out = patent_panel.merge(sec_with_gvkey, on=["gvkey"], how="left")
        if not year_context.empty:
            out = out.merge(year_context, on="year", how="left")
    else:
        sec_with_gvkey = linked[
            [
                "gvkey",
                "cik",
                "link_method",
                "link_score",
                "link_conflict",
                "candidate_count",
                "ticker_hint",
            ]
        ].drop_duplicates().merge(
            sec_panel,
            on="cik",
            how="left",
        )
        out = patent_panel.merge(sec_with_gvkey, on=["gvkey", "year"], how="left")

    out_path = config.output_dir / "firm_year_enriched.csv"
    patent_path = config.output_dir / "patent_firm_year_panel.csv"
    link_path = config.output_dir / "gvkey_cik_links.csv"
    link_candidates_path = config.output_dir / "gvkey_cik_link_candidates.csv"
    link_conflicts_path = config.output_dir / "gvkey_cik_link_conflicts.csv"

    out.to_csv(out_path, index=False)
    patent_panel.to_csv(patent_path, index=False)
    links.to_csv(link_path, index=False)
    link_candidates.to_csv(link_candidates_path, index=False)
    link_conflicts.to_csv(link_conflicts_path, index=False)

    return {
        "firm_year_enriched": out_path,
        "patent_firm_year_panel": patent_path,
        "gvkey_cik_links": link_path,
        "gvkey_cik_link_candidates": link_candidates_path,
        "gvkey_cik_link_conflicts": link_conflicts_path,
    }
