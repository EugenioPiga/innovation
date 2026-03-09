from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .patents import write_combined_static_file
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build firm-year patent + public-data enrichment panel.")
    p.add_argument("--data-dir", type=Path, default=Path("..\\compustat-patents\\data"), help="Directory with static/dynamic patent CSVs.")
    p.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory.")
    p.add_argument("--cache-dir", type=Path, default=Path("cache"), help="API cache directory.")
    p.add_argument("--start-year", type=int, default=1997)
    p.add_argument("--end-year", type=int, default=2022)
    p.add_argument("--max-firms", type=int, default=250, help="Optional speed cap for top patenting firms.")
    p.add_argument(
        "--sec-user-agent",
        type=str,
        default="research@example.com",
        help="SEC requires a descriptive User-Agent with contact info.",
    )

    p.add_argument("--manual-link-file", type=Path, default=None, help="Optional CSV with columns gvkey,cik.")
    p.add_argument("--ticker-hints-file", type=Path, default=None, help="Optional CSV with columns gvkey,ticker.")
    p.add_argument("--bls-series-file", type=Path, default=None, help="Optional CSV with columns series_id,metric_name.")

    p.add_argument("--link-fuzzy-threshold", type=float, default=0.86, help="Minimum fuzzy score to keep (0-1).")
    p.add_argument("--link-score-gap", type=float, default=0.03, help="Top-1 minus top-2 score gap under which link is flagged as conflict.")
    p.add_argument("--link-top-n", type=int, default=5, help="How many top candidates to keep per gvkey.")

    p.add_argument("--bls-api-key", type=str, default=None, help="Optional BLS API key.")
    p.add_argument("--bea-api-key", type=str, default=None, help="Optional BEA API key.")
    p.add_argument("--fred-api-key", type=str, default=None, help="Optional FRED API key.")

    p.add_argument("--skip-10k-text", action="store_true", help="Skip 10-K text keyword features (faster).")
    p.add_argument("--skip-fred", action="store_true", help="Skip FRED macro controls.")
    p.add_argument("--skip-bls-oes", action="store_true", help="Skip BLS OEWS/OES controls.")
    p.add_argument("--skip-census", action="store_true", help="Skip Census CBP industry context.")
    p.add_argument("--include-bea", action="store_true", help="Include BEA industry context (requires --bea-api-key).")
    p.add_argument("--combined-static-output", type=Path, default=None, help="Optional path to write one combined patent-level static CSV.")
    p.add_argument("--combine-static-only", action="store_true", help="Only write the combined patent-level static CSV and exit.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.combined_static_output is not None:
        stats = write_combined_static_file(data_dir=args.data_dir, output_path=args.combined_static_output)
        print(f"combined_static: {args.combined_static_output}")
        print(f"combined_static_files: {stats['files']}")
        print(f"combined_static_rows: {stats['rows']}")
        if args.combine_static_only:
            return

    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        max_firms=args.max_firms,
        sec_user_agent=args.sec_user_agent,
        include_10k_text_features=not args.skip_10k_text,
        include_fred_macro=not args.skip_fred,
        include_bls_oes=not args.skip_bls_oes,
        include_census_context=not args.skip_census,
        include_bea_context=args.include_bea,
        fred_api_key=args.fred_api_key,
        bls_api_key=args.bls_api_key,
        bea_api_key=args.bea_api_key,
        link_fuzzy_threshold=args.link_fuzzy_threshold,
        link_score_gap=args.link_score_gap,
        link_top_n=args.link_top_n,
    )
    outputs = run_pipeline(
        config=config,
        manual_link_file=args.manual_link_file,
        ticker_hints_file=args.ticker_hints_file,
        bls_series_file=args.bls_series_file,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
