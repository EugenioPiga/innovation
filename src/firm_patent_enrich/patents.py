from __future__ import annotations

from pathlib import Path

import pandas as pd


STATIC_COLS = [
    "patent_id",
    "appYear",
    "gvkeyUO",
    "gvkeyFR",
    "clean_name",
    "privateSubsidiary",
    "grantYear",
]


def discover_static_files(data_dir: Path) -> list[Path]:
    full_static = data_dir / "static.csv"
    if full_static.exists():
        return [full_static]
    tranche_files = sorted(data_dir.glob("staticTranche*.csv"))
    if not tranche_files:
        raise FileNotFoundError(f"No static patent files found in {data_dir}")
    return tranche_files


def load_static_patents(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in discover_static_files(data_dir):
        frame = pd.read_csv(path, usecols=STATIC_COLS, dtype={"gvkeyUO": "string", "gvkeyFR": "string"})
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["gvkeyUO", "appYear"])  # firm-year granularity uses owner gvkey
    return out


def build_patent_firm_year_panel(static_patents: pd.DataFrame) -> pd.DataFrame:
    panel = (
        static_patents.groupby(["gvkeyUO", "appYear"], as_index=False)
        .agg(
            patents_applied=("patent_id", "nunique"),
            patents_granted=("grantYear", lambda s: int(s.notna().sum())),
            private_subsidiary_patents=("privateSubsidiary", lambda s: int((s == 1).sum())),
            distinct_assignee_names=("clean_name", "nunique"),
        )
        .rename(columns={"gvkeyUO": "gvkey", "appYear": "year"})
    )
    panel["private_subsidiary_share"] = panel["private_subsidiary_patents"] / panel["patents_applied"]
    return panel


def build_primary_name_map(static_patents: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        static_patents.groupby(["gvkeyUO", "clean_name"], as_index=False)
        .agg(patent_count=("patent_id", "nunique"))
        .sort_values(["gvkeyUO", "patent_count"], ascending=[True, False])
    )
    top = ranked.drop_duplicates(subset=["gvkeyUO"]).rename(columns={"gvkeyUO": "gvkey", "clean_name": "firm_name"})
    return top[["gvkey", "firm_name", "patent_count"]]