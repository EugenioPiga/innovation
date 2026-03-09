from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz, process
from tqdm import tqdm

from .utils import compact_name, normalize_name, to_cik_str


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


@dataclass
class LinkConfig:
    fuzzy_threshold: float = 0.86
    score_gap_for_conflict: float = 0.03
    top_n_candidates: int = 5


def fetch_sec_tickers(cache_dir: Path, user_agent: str) -> pd.DataFrame:
    cache_file = cache_dir / "sec_company_tickers_exchange.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        resp = requests.get(SEC_TICKERS_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        cache_file.write_text(resp.text, encoding="utf-8")
        payload = resp.json()

    columns = payload.get("fields", [])
    rows = payload.get("data", [])
    if not columns or not rows:
        raise ValueError("Unexpected schema from SEC tickers file")

    df = pd.DataFrame(rows, columns=columns)
    df = df.rename(columns={"name": "sec_name", "ticker": "ticker", "cik": "cik"})
    df["cik"] = df["cik"].map(to_cik_str)
    df["sec_name_norm"] = df["sec_name"].map(normalize_name)
    df["sec_name_compact"] = df["sec_name"].map(compact_name)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper()
    return df[["cik", "ticker", "sec_name", "sec_name_norm", "sec_name_compact", "exchange"]]


def load_manual_link_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"gvkey": "string", "cik": "string"})
    required = {"gvkey", "cik"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manual link file missing columns: {sorted(missing)}")
    df["cik"] = df["cik"].map(to_cik_str)
    df["link_method"] = "manual_override"
    return df[["gvkey", "cik", "link_method"]]


def load_ticker_hints_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"gvkey": "string", "ticker": "string"})
    required = {"gvkey", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ticker hints file missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].fillna("").str.upper().str.strip()
    df = df[df["ticker"] != ""]
    return df[["gvkey", "ticker"]]


def _build_candidates(
    firm_name_norm: str,
    firm_name_compact: str,
    sec_tickers: pd.DataFrame,
    sec_name_values: list[str],
    sec_by_name: dict[str, pd.DataFrame],
    ticker_hint: str | None,
    config: LinkConfig,
) -> pd.DataFrame:
    rows: list[dict] = []

    if firm_name_norm:
        exact_norm = sec_tickers[sec_tickers["sec_name_norm"] == firm_name_norm]
        for r in exact_norm.itertuples(index=False):
            rows.append(
                {
                    "cik": r.cik,
                    "ticker": r.ticker,
                    "sec_name": r.sec_name,
                    "exchange": r.exchange,
                    "score": 1.0,
                    "method": "exact_normalized_name",
                }
            )

    if firm_name_compact:
        exact_compact = sec_tickers[sec_tickers["sec_name_compact"] == firm_name_compact]
        for r in exact_compact.itertuples(index=False):
            rows.append(
                {
                    "cik": r.cik,
                    "ticker": r.ticker,
                    "sec_name": r.sec_name,
                    "exchange": r.exchange,
                    "score": 0.985,
                    "method": "exact_compact_name",
                }
            )

    if ticker_hint:
        ticker_rows = sec_tickers[sec_tickers["ticker"] == ticker_hint]
        for r in ticker_rows.itertuples(index=False):
            fuzzy_score = fuzz.WRatio(firm_name_norm, r.sec_name_norm) / 100 if firm_name_norm else 0.0
            rows.append(
                {
                    "cik": r.cik,
                    "ticker": r.ticker,
                    "sec_name": r.sec_name,
                    "exchange": r.exchange,
                    "score": min(1.0, fuzzy_score + 0.08),
                    "method": "ticker_hint",
                }
            )

    if firm_name_norm:
        fuzzy_hits = process.extract(
            query=firm_name_norm,
            choices=sec_name_values,
            scorer=fuzz.WRatio,
            limit=max(20, config.top_n_candidates * 4),
        )
        for name_match, raw_score, _ in fuzzy_hits:
            score = raw_score / 100
            if score < config.fuzzy_threshold:
                continue
            for r in sec_by_name[name_match].itertuples(index=False):
                rows.append(
                    {
                        "cik": r.cik,
                        "ticker": r.ticker,
                        "sec_name": r.sec_name,
                        "exchange": r.exchange,
                        "score": score,
                        "method": "fuzzy_name",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["cik", "ticker", "sec_name", "exchange", "score", "method"])

    out = pd.DataFrame(rows)
    method_rank = {
        "exact_normalized_name": 0,
        "exact_compact_name": 1,
        "ticker_hint": 2,
        "fuzzy_name": 3,
    }
    out["method_rank"] = out["method"].map(method_rank).fillna(9)
    out = out.sort_values(["score", "method_rank"], ascending=[False, True])
    out = out.drop_duplicates(subset=["cik"], keep="first")
    out = out.sort_values(["score", "method_rank"], ascending=[False, True]).reset_index(drop=True)
    return out


def auto_link_gvkey_to_cik(
    primary_names: pd.DataFrame,
    sec_tickers: pd.DataFrame,
    ticker_hints: pd.DataFrame | None = None,
    config: LinkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config is None:
        config = LinkConfig()

    firms = primary_names.copy()
    firms["firm_name_norm"] = firms["firm_name"].map(normalize_name)
    firms["firm_name_compact"] = firms["firm_name"].map(compact_name)

    if ticker_hints is not None and not ticker_hints.empty:
        firms = firms.merge(ticker_hints, on="gvkey", how="left")
        firms["ticker_hint"] = firms["ticker"].fillna("").astype(str).str.upper().str.strip()
        firms = firms.drop(columns=["ticker"])
    else:
        firms["ticker_hint"] = ""

    sec_name_values = sorted(set(sec_tickers["sec_name_norm"].dropna().astype(str)))
    sec_by_name = {name: sec_tickers[sec_tickers["sec_name_norm"] == name] for name in sec_name_values}

    link_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for firm in tqdm(firms.itertuples(index=False), total=len(firms), desc="Linking gvkey->CIK"):
        candidates = _build_candidates(
            firm_name_norm=firm.firm_name_norm,
            firm_name_compact=firm.firm_name_compact,
            sec_tickers=sec_tickers,
            sec_name_values=sec_name_values,
            sec_by_name=sec_by_name,
            ticker_hint=firm.ticker_hint if firm.ticker_hint else None,
            config=config,
        )

        if candidates.empty:
            link_rows.append(
                {
                    "gvkey": firm.gvkey,
                    "firm_name": firm.firm_name,
                    "patent_count": firm.patent_count,
                    "cik": pd.NA,
                    "ticker": pd.NA,
                    "sec_name": pd.NA,
                    "exchange": pd.NA,
                    "link_method": "unlinked",
                    "link_score": 0.0,
                    "link_conflict": False,
                    "candidate_count": 0,
                    "ticker_hint": firm.ticker_hint,
                }
            )
            continue

        top = candidates.iloc[0]
        second = candidates.iloc[1] if len(candidates) > 1 else None
        conflict = False
        if second is not None:
            conflict = bool((float(top["score"]) - float(second["score"])) < config.score_gap_for_conflict)

        candidate_count = int(len(candidates))
        link_rows.append(
            {
                "gvkey": firm.gvkey,
                "firm_name": firm.firm_name,
                "patent_count": firm.patent_count,
                "cik": top["cik"],
                "ticker": top["ticker"],
                "sec_name": top["sec_name"],
                "exchange": top["exchange"],
                "link_method": top["method"],
                "link_score": float(top["score"]),
                "link_conflict": conflict,
                "candidate_count": candidate_count,
                "ticker_hint": firm.ticker_hint,
            }
        )

        top_n = min(config.top_n_candidates, len(candidates))
        for rank in range(top_n):
            row = candidates.iloc[rank]
            candidate_rows.append(
                {
                    "gvkey": firm.gvkey,
                    "firm_name": firm.firm_name,
                    "patent_count": firm.patent_count,
                    "candidate_rank": rank + 1,
                    "candidate_cik": row["cik"],
                    "candidate_ticker": row["ticker"],
                    "candidate_sec_name": row["sec_name"],
                    "candidate_exchange": row["exchange"],
                    "candidate_method": row["method"],
                    "candidate_score": float(row["score"]),
                    "ticker_hint": firm.ticker_hint,
                }
            )

    links = pd.DataFrame(link_rows)
    candidates = pd.DataFrame(candidate_rows)
    conflicts = links[links["link_conflict"]].copy()

    if not conflicts.empty and not candidates.empty:
        conflicts = conflicts.merge(candidates, on=["gvkey", "firm_name", "patent_count", "ticker_hint"], how="left")

    return links, candidates, conflicts
