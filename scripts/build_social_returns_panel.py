#!/usr/bin/env python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "datasets" / "social_returns_data"


@dataclass(frozen=True)
class MetadataRow:
    variable: str
    source: str
    description: str


def normalize_name(value: str) -> str:
    value = str(value).upper().strip()
    value = value.replace("&", "AND")
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def fetch_world_bank_country_metadata() -> pd.DataFrame:
    response = requests.get(
        "https://api.worldbank.org/v2/country",
        params={"format": "json", "per_page": 400},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload[1]
    rows = []
    for rec in records:
        iso3 = rec.get("id")
        iso2 = rec.get("iso2Code")
        if not iso3 or len(iso3) != 3 or iso2 == "NA":
            continue
        if (rec.get("region") or {}).get("id") == "NA":
            continue
        rows.append(
            {
                "iso3c": iso3,
                "iso2c": iso2,
                "wb_country_name": rec.get("name"),
                "wb_region": (rec.get("region") or {}).get("value"),
                "wb_income_level": (rec.get("incomeLevel") or {}).get("value"),
                "wb_lending_type": (rec.get("lendingType") or {}).get("value"),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["iso3c"])
    out["wb_name_norm"] = out["wb_country_name"].map(normalize_name)
    return out


def to_year(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.round().astype("Int64")


def load_wdi_panel() -> tuple[pd.DataFrame, List[MetadataRow]]:
    path = DATA_ROOT / "processed" / "world_bank" / "wdi_selected_indicators_wide.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"countryiso3code": "iso3c"})
    df["iso3c"] = df["iso3c"].astype(str).str.upper()
    df["year"] = to_year(df["year"])
    df = df[df["iso3c"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    df = df.drop_duplicates(subset=["iso3c", "year"])

    id_to_name = dict(
        pd.read_csv(
            DATA_ROOT / "processed" / "world_bank" / "wdi_selected_indicator_metadata.csv"
        ).values
    )
    meta = [
        MetadataRow("country_name", "World Bank WDI", "Country name from WDI."),
    ]
    for col in df.columns:
        if col in {"iso3c", "year", "country_name"}:
            continue
        description = id_to_name.get(col, "WDI indicator")
        meta.append(MetadataRow(col, "World Bank WDI", description))
    return df, meta


def load_pwt_panel() -> tuple[pd.DataFrame, List[MetadataRow]]:
    pwt_path = DATA_ROOT / "raw" / "pwt" / "pwt100_main_xlsx.xlsx"
    selected = [
        "countrycode",
        "country",
        "year",
        "rgdpna",
        "ctfp",
        "cwtfp",
        "rtfpna",
        "rwtfpna",
        "pop",
        "emp",
        "avh",
        "hc",
        "rnna",
        "rkna",
    ]
    df = pd.read_excel(pwt_path, sheet_name="Data", usecols=selected)
    rename_map = {
        "countrycode": "iso3c",
        "country": "pwt_country_name",
        "year": "year",
        "rgdpna": "pwt_rgdpna_mil_2017usd",
        "ctfp": "pwt_ctfp_level",
        "cwtfp": "pwt_cwtfp_level",
        "rtfpna": "pwt_rtfpna_level",
        "rwtfpna": "pwt_rwtfpna_level",
        "pop": "pwt_pop_millions",
        "emp": "pwt_emp_millions",
        "avh": "pwt_avh_hours",
        "hc": "pwt_human_capital_index",
        "rnna": "pwt_natural_capital_stock_mil_2017usd",
        "rkna": "pwt_capital_stock_mil_2017usd",
    }
    df = df.rename(columns=rename_map)
    df["iso3c"] = df["iso3c"].astype(str).str.upper()
    df["year"] = to_year(df["year"])
    df = df[df["iso3c"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    df = df.drop_duplicates(subset=["iso3c", "year"])

    meta = [
        MetadataRow("pwt_country_name", "Penn World Table 10.0", "Country name in PWT."),
        MetadataRow(
            "pwt_rgdpna_mil_2017usd",
            "Penn World Table 10.0",
            "Real GDP at constant 2017 national prices (millions).",
        ),
        MetadataRow("pwt_ctfp_level", "Penn World Table 10.0", "TFP level at current PPPs."),
        MetadataRow(
            "pwt_cwtfp_level",
            "Penn World Table 10.0",
            "Welfare-relevant TFP level at current PPPs.",
        ),
        MetadataRow(
            "pwt_rtfpna_level",
            "Penn World Table 10.0",
            "TFP level at constant national prices.",
        ),
        MetadataRow(
            "pwt_rwtfpna_level",
            "Penn World Table 10.0",
            "Welfare-relevant TFP at constant national prices.",
        ),
        MetadataRow("pwt_pop_millions", "Penn World Table 10.0", "Population (millions)."),
        MetadataRow("pwt_emp_millions", "Penn World Table 10.0", "Employment (millions)."),
        MetadataRow(
            "pwt_avh_hours",
            "Penn World Table 10.0",
            "Average annual hours worked by engaged person.",
        ),
        MetadataRow(
            "pwt_human_capital_index",
            "Penn World Table 10.0",
            "Human capital index based on years of schooling and returns.",
        ),
        MetadataRow(
            "pwt_natural_capital_stock_mil_2017usd",
            "Penn World Table 10.0",
            "Natural capital stock at constant 2017 national prices (millions).",
        ),
        MetadataRow(
            "pwt_capital_stock_mil_2017usd",
            "Penn World Table 10.0",
            "Capital stock at constant 2017 national prices (millions).",
        ),
    ]
    return df, meta


def load_oecd_file(file_name: str) -> pd.DataFrame:
    path = DATA_ROOT / "raw" / "oecd" / file_name
    df = pd.read_csv(path, low_memory=False)
    df["iso3c"] = df["REF_AREA"].astype(str).str.upper()
    df["year"] = to_year(df["TIME_PERIOD"])
    df["obs_value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df[df["iso3c"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
    df = df.dropna(subset=["year", "obs_value"])
    return df


def extract_oecd_series(
    df: pd.DataFrame,
    variable: str,
    filters: Dict[str, Any],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if isinstance(val, (list, tuple, set)):
            mask &= df[col].isin(list(val))
        else:
            mask &= df[col] == val
    out = df.loc[mask, ["iso3c", "year", "obs_value"]].copy()
    out = (
        out.groupby(["iso3c", "year"], as_index=False)["obs_value"]
        .mean()
        .rename(columns={"obs_value": variable})
    )
    return out


def build_oecd_panel() -> tuple[pd.DataFrame, List[MetadataRow], pd.DataFrame]:
    metadata: List[MetadataRow] = []
    series_frames: List[pd.DataFrame] = []

    gerd = load_oecd_file("oecd_gerd_toe.csv")
    gerd_specs = [
        (
            "oecd_gerd_total_usd_ppp_const2015",
            {"MEASURE": "G", "SECT_PERF": "_T", "TYPE_COST": "_T", "UNIT_MEASURE": "USD_PPP", "PRICE_BASE": "Q"},
            "Gross domestic expenditure on R&D, all performing sectors; constant PPP USD (base 2015).",
        ),
        (
            "oecd_gerd_business_usd_ppp_const2015",
            {"MEASURE": "G", "SECT_PERF": "BES", "TYPE_COST": "_T", "UNIT_MEASURE": "USD_PPP", "PRICE_BASE": "Q"},
            "GERD performed by business enterprise sector; constant PPP USD (base 2015).",
        ),
        (
            "oecd_gerd_government_usd_ppp_const2015",
            {"MEASURE": "G", "SECT_PERF": "GOV", "TYPE_COST": "_T", "UNIT_MEASURE": "USD_PPP", "PRICE_BASE": "Q"},
            "GERD performed by government sector; constant PPP USD (base 2015).",
        ),
        (
            "oecd_gerd_highered_usd_ppp_const2015",
            {"MEASURE": "G", "SECT_PERF": "HES", "TYPE_COST": "_T", "UNIT_MEASURE": "USD_PPP", "PRICE_BASE": "Q"},
            "GERD performed by higher education sector; constant PPP USD (base 2015).",
        ),
    ]
    for var, filt, desc in gerd_specs:
        series_frames.append(extract_oecd_series(gerd, var, filt))
        metadata.append(MetadataRow(var, "OECD SDMX (DF_GERD_TOE)", desc))

    pers = load_oecd_file("oecd_pers_ford.csv")
    common_pers = {
        "MEASURE": "T_RD",
        "FORD": "_T",
        "ACTIVITY": "_T",
        "FUNCTION": "_T",
        "EDUCATION_LEV": "_T",
        "SEX": "_T",
        "EMP_STATUS": "INT",
        "UNIT_MEASURE": "PS_FTE",
    }
    for sect, label in [("_T", "total"), ("BES", "business"), ("HES", "highered"), ("GOV", "government")]:
        var = f"oecd_rd_personnel_fte_{label}"
        filt = dict(common_pers)
        filt["SECT_PERF"] = sect
        series_frames.append(extract_oecd_series(pers, var, filt))
        metadata.append(
            MetadataRow(
                var,
                "OECD SDMX (DF_PERS_FORD)",
                f"R&D personnel (internal), full-time equivalent, {label} performing sector.",
            )
        )

    rdtax = load_oecd_file("oecd_rdtax.csv")
    common_rdtax = {"SIZE": "_Z", "PROFIT_SCENARIO": "_Z", "UNIT_MEASURE": "PT_B1GQ", "PRICE_BASE": "_Z"}
    for measure, var, desc in [
        ("RDTAX", "oecd_rdtax_pct_gdp", "Tax support for R&D as percent of GDP."),
        ("DF", "oecd_direct_funding_pct_gdp", "Direct funding for R&D as percent of GDP."),
        ("RDTAXSUB", "oecd_total_support_pct_gdp", "Total R&D support (tax + direct funding) as percent of GDP."),
        ("GBARD", "oecd_gbard_pct_gdp", "Government budget allocations for R&D as percent of GDP."),
    ]:
        filt = dict(common_rdtax)
        filt["MEASURE"] = measure
        series_frames.append(extract_oecd_series(rdtax, var, filt))
        metadata.append(MetadataRow(var, "OECD SDMX (DF_RDTAX)", desc))

    rdsub = load_oecd_file("oecd_rdsub.csv")
    for size in ["SME", "LARGE"]:
        for profit in ["PROFITABLE", "LOSS_MAKING"]:
            var = f"oecd_b_index_{size.lower()}_{profit.lower()}"
            desc = f"B-index for implied tax subsidy: size={size}, profit scenario={profit}."
            filt = {
                "MEASURE": "RDSUB",
                "UNIT_MEASURE": "IX",
                "SIZE": size,
                "PROFIT_SCENARIO": profit,
            }
            series_frames.append(extract_oecd_series(rdsub, var, filt))
            metadata.append(MetadataRow(var, "OECD SDMX (DF_RDSUB)", desc))

    iptax = load_oecd_file("oecd_iptax.csv")
    series_frames.append(
        extract_oecd_series(
            iptax,
            "oecd_iptax_pct_gdp",
            {
                "MEASURE": "IPTAX",
                "UNIT_MEASURE": "PT_B1GQ",
                "SIZE": "_Z",
                "PROFIT_SCENARIO": "_Z",
                "PRICE_BASE": "_Z",
            },
        )
    )
    metadata.append(
        MetadataRow(
            "oecd_iptax_pct_gdp",
            "OECD SDMX (DF_IPTAX)",
            "Tax support for innovation/pro-IP policy index, as percent of GDP equivalent.",
        )
    )

    msti = load_oecd_file("oecd_msti.csv")
    msti_filtered = msti[
        (msti["UNIT_MEASURE"] == "PT_B1GQ")
        & (msti["PRICE_BASE"] == "_Z")
        & (msti["TRANSFORMATION"] == "_Z")
    ].copy()
    measure_coverage = (
        msti_filtered.groupby("MEASURE", as_index=False)["obs_value"]
        .count()
        .rename(columns={"obs_value": "n_obs"})
        .sort_values("n_obs", ascending=False)
    )
    top_measures = measure_coverage.head(8)["MEASURE"].tolist()
    msti_selected = msti_filtered[msti_filtered["MEASURE"].isin(top_measures)].copy()
    msti_selected["variable"] = "oecd_msti_" + msti_selected["MEASURE"].str.lower() + "_pct_gdp"
    msti_wide = (
        msti_selected.pivot_table(
            index=["iso3c", "year"],
            columns="variable",
            values="obs_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    series_frames.append(msti_wide)
    for code in top_measures:
        metadata.append(
            MetadataRow(
                f"oecd_msti_{code.lower()}_pct_gdp",
                "OECD SDMX (DF_MSTI)",
                f"MSTI measure code {code} in percent-of-GDP units (PT_B1GQ).",
            )
        )

    merged = None
    for frame in series_frames:
        if merged is None:
            merged = frame.copy()
        else:
            merged = merged.merge(frame, on=["iso3c", "year"], how="outer")

    measure_coverage = measure_coverage.reset_index(drop=True)
    return merged, metadata, measure_coverage


def map_ecipe_country_to_iso3(
    countries: Iterable[str],
    wb_meta: pd.DataFrame,
) -> pd.DataFrame:
    manual = {
        "BRUNEI": "BRN",
        "CZECH REPUBLIC": "CZE",
        "HONG KONG": "HKG",
        "KOREA": "KOR",
        "RUSSIA": "RUS",
        "SLOVAKIA": "SVK",
        "TAIWAN": "TWN",
        "TURKEY": "TUR",
        "VIETNAM": "VNM",
    }
    wb_norm_to_iso = dict(zip(wb_meta["wb_name_norm"], wb_meta["iso3c"]))
    rows = []
    for name in sorted(set(countries)):
        if name == "EUROPEAN UNION":
            rows.append({"ecipe_country": name, "iso3c": None, "matched_by": "excluded_non_country"})
            continue
        if name in manual:
            rows.append({"ecipe_country": name, "iso3c": manual[name], "matched_by": "manual"})
            continue
        normalized = normalize_name(name)
        if normalized in wb_norm_to_iso:
            rows.append(
                {"ecipe_country": name, "iso3c": wb_norm_to_iso[normalized], "matched_by": "exact_normalized"}
            )
        else:
            rows.append({"ecipe_country": name, "iso3c": None, "matched_by": "unmatched"})
    return pd.DataFrame(rows)


def load_ecipe_panel(wb_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, List[MetadataRow]]:
    path = DATA_ROOT / "processed" / "regulation" / "ecipe_dte_measures_by_country_year.csv"
    df = pd.read_csv(path)
    mapping = map_ecipe_country_to_iso3(df["Country"].unique(), wb_meta)
    df = df.merge(mapping, left_on="Country", right_on="ecipe_country", how="left")
    df["year"] = to_year(df["year_from_timeframe"])
    df["ecipe_dte_n_measures"] = pd.to_numeric(df["n_measures"], errors="coerce")
    df = df.dropna(subset=["iso3c", "year", "ecipe_dte_n_measures"]).copy()
    out = (
        df.groupby(["iso3c", "year"], as_index=False)["ecipe_dte_n_measures"]
        .sum()
        .sort_values(["iso3c", "year"])
    )
    out["ecipe_dte_n_measures_cum"] = out.groupby("iso3c")["ecipe_dte_n_measures"].cumsum()
    out["ecipe_dte_any_measure"] = (out["ecipe_dte_n_measures"] > 0).astype(int)
    out["ecipe_dte_first_year"] = out.groupby("iso3c")["year"].transform("min")
    out["ecipe_dte_post_any"] = (out["year"] >= out["ecipe_dte_first_year"]).astype(int)

    meta = [
        MetadataRow(
            "ecipe_dte_n_measures",
            "ECIPE DTE",
            "Count of documented digital trade measures starting in that country-year (timeframe parsed).",
        ),
        MetadataRow(
            "ecipe_dte_n_measures_cum",
            "ECIPE DTE",
            "Cumulative count of documented digital trade measures by country-year.",
        ),
        MetadataRow(
            "ecipe_dte_any_measure",
            "ECIPE DTE",
            "Indicator for at least one documented digital trade measure in the country-year.",
        ),
        MetadataRow(
            "ecipe_dte_first_year",
            "ECIPE DTE",
            "First year with at least one documented measure (based on parsed timeframe).",
        ),
        MetadataRow(
            "ecipe_dte_post_any",
            "ECIPE DTE",
            "Indicator for years on/after first documented measure.",
        ),
    ]
    return out, mapping, meta


def load_unctad_static(wb_meta: pd.DataFrame) -> tuple[pd.DataFrame, List[MetadataRow]]:
    path = DATA_ROOT / "processed" / "regulation" / "unctad_cyberlaw_country_wide.csv"
    df = pd.read_csv(path)
    iso2_to_iso3 = dict(zip(wb_meta["iso2c"], wb_meta["iso3c"]))
    df["iso3c"] = df["iso2c"].map(iso2_to_iso3)
    df = df.dropna(subset=["iso3c"]).copy()
    code_cols = [c for c in df.columns if c.endswith("_code")]
    keep = ["iso3c"] + code_cols
    df = df[keep].copy()
    rename = {c: f"unctad_{c}" for c in code_cols}
    df = df.rename(columns=rename).drop_duplicates(subset=["iso3c"])

    meta = []
    for c in rename.values():
        meta.append(
            MetadataRow(
                c,
                "UNCTAD Cyberlaw tracker",
                "Status code snapshot (0=no data, 1=legislation, 2=draft legislation, 3=no legislation).",
            )
        )
    return df, meta


def load_datagouv_static(wb_meta: pd.DataFrame) -> tuple[pd.DataFrame, List[MetadataRow]]:
    path = DATA_ROOT / "processed" / "regulation" / "datagouv_global_privacy_mapping_normalized.csv"
    df = pd.read_csv(path)
    iso2_to_iso3 = dict(zip(wb_meta["iso2c"], wb_meta["iso3c"]))
    df["iso2c"] = df["Code Pays (ISO)"].astype(str).str.upper().str.strip()
    df["iso3c"] = df["iso2c"].map(iso2_to_iso3)
    level_map = {
        "Pas de loi": 0,
        "Loi (non adéquat)": 1,
        "Autorité et loi spécifiques": 2,
        "Pays en adéquation partielle": 3,
        "Pays adéquat": 4,
        "Pays membre de l'UE ou de l'EEE": 5,
    }
    df["datagouv_privacy_level_ordinal"] = df["Niveau de protection"].map(level_map)
    df["datagouv_member_edpb"] = df["Membre de l'EDPB"].map({"Oui": 1, "Non": 0})
    df["datagouv_member_afapdp"] = df["Membre de l'AFAPDP"].map({"Oui": 1, "Non": 0})
    keep = [
        "iso3c",
        "datagouv_privacy_level_ordinal",
        "datagouv_member_edpb",
        "datagouv_member_afapdp",
    ]
    out = df[keep].dropna(subset=["iso3c"]).drop_duplicates(subset=["iso3c"]).copy()
    meta = [
        MetadataRow(
            "datagouv_privacy_level_ordinal",
            "data.gouv.fr global privacy map",
            "Static privacy-protection ordinal score: 0=no law ... 5=EU/EEA member.",
        ),
        MetadataRow(
            "datagouv_member_edpb",
            "data.gouv.fr global privacy map",
            "Static indicator for EDPB membership.",
        ),
        MetadataRow(
            "datagouv_member_afapdp",
            "data.gouv.fr global privacy map",
            "Static indicator for AFAPDP membership.",
        ),
    ]
    return out, meta


def coalesce_country_name(frame: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["country_name", "pwt_country_name", "wb_country_name"] if c in frame.columns]
    result = pd.Series(np.nan, index=frame.index, dtype=object)
    for c in cols:
        result = result.where(result.notna(), frame[c])
    return result


def build_panel() -> None:
    wb_meta = fetch_world_bank_country_metadata()
    wdi, wdi_meta = load_wdi_panel()
    pwt, pwt_meta = load_pwt_panel()
    oecd, oecd_meta, msti_coverage = build_oecd_panel()
    ecipe, ecipe_mapping, ecipe_meta = load_ecipe_panel(wb_meta)
    unctad, unctad_meta = load_unctad_static(wb_meta)
    datagouv, datagouv_meta = load_datagouv_static(wb_meta)

    key_frames = [wdi[["iso3c", "year"]], pwt[["iso3c", "year"]], oecd[["iso3c", "year"]], ecipe[["iso3c", "year"]]]
    keys = pd.concat(key_frames, ignore_index=True).dropna().drop_duplicates()
    keys["year"] = keys["year"].astype("Int64")

    panel = keys.merge(wdi, on=["iso3c", "year"], how="left")
    panel = panel.merge(pwt, on=["iso3c", "year"], how="left")
    panel = panel.merge(oecd, on=["iso3c", "year"], how="left")
    panel = panel.merge(ecipe, on=["iso3c", "year"], how="left")
    panel = panel.merge(unctad, on="iso3c", how="left")
    panel = panel.merge(datagouv, on="iso3c", how="left")
    panel = panel.merge(wb_meta.drop(columns=["wb_name_norm"]), on="iso3c", how="left")

    # Keep country/economy ISO codes and drop WDI aggregate groups.
    valid_iso = set(wb_meta["iso3c"].unique()).union({"TWN"})
    panel = panel[panel["iso3c"].isin(valid_iso)].copy()

    panel["country_name"] = coalesce_country_name(panel)
    panel["log_gdp_per_capita_const_2015"] = np.log(
        pd.to_numeric(panel["NY.GDP.PCAP.KD"], errors="coerce")
    )
    panel["log_patent_resident_plus1"] = np.log1p(pd.to_numeric(panel["IP.PAT.RESD"], errors="coerce"))
    panel["log_patent_total_plus1"] = np.log1p(
        pd.to_numeric(panel["IP.PAT.RESD"], errors="coerce")
        + pd.to_numeric(panel["IP.PAT.NRES"], errors="coerce")
    )

    order_front = ["iso3c", "country_name", "year", "wb_region", "wb_income_level", "wb_lending_type"]
    remaining = [c for c in panel.columns if c not in order_front]
    panel = panel[order_front + remaining]
    panel = panel.sort_values(["iso3c", "year"]).reset_index(drop=True)

    out_dir = DATA_ROOT / "processed" / "panel"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / "social_returns_country_year_panel.csv"
    panel.to_csv(panel_path, index=False)

    ecipe_mapping_path = out_dir / "ecipe_country_iso3_mapping.csv"
    ecipe_mapping.to_csv(ecipe_mapping_path, index=False)

    msti_coverage_path = out_dir / "oecd_msti_measure_coverage.csv"
    msti_coverage.to_csv(msti_coverage_path, index=False)

    metadata_rows = [
        MetadataRow("iso3c", "Harmonized key", "ISO-3 country code."),
        MetadataRow("year", "Harmonized key", "Calendar year."),
        MetadataRow("country_name", "Merged", "Best-available country name across WDI/PWT/WB metadata."),
        MetadataRow("wb_region", "World Bank metadata", "World Bank region."),
        MetadataRow("wb_income_level", "World Bank metadata", "World Bank income group."),
        MetadataRow("wb_lending_type", "World Bank metadata", "World Bank lending category."),
        MetadataRow(
            "log_gdp_per_capita_const_2015",
            "Derived",
            "Natural log of WDI GDP per capita (constant 2015 USD).",
        ),
        MetadataRow(
            "log_patent_resident_plus1",
            "Derived",
            "Natural log of (resident patent applications + 1).",
        ),
        MetadataRow(
            "log_patent_total_plus1",
            "Derived",
            "Natural log of (resident + nonresident patent applications + 1).",
        ),
    ]
    metadata_rows.extend(wdi_meta)
    metadata_rows.extend(pwt_meta)
    metadata_rows.extend(oecd_meta)
    metadata_rows.extend(ecipe_meta)
    metadata_rows.extend(unctad_meta)
    metadata_rows.extend(datagouv_meta)
    metadata_df = pd.DataFrame([m.__dict__ for m in metadata_rows]).drop_duplicates(subset=["variable"])
    metadata_path = out_dir / "social_returns_country_year_panel_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    diagnostics = {
        "rows": int(len(panel)),
        "countries": int(panel["iso3c"].nunique(dropna=True)),
        "year_min": int(panel["year"].min()) if panel["year"].notna().any() else None,
        "year_max": int(panel["year"].max()) if panel["year"].notna().any() else None,
        "columns": int(panel.shape[1]),
        "ecipe_unmatched_countries": sorted(
            ecipe_mapping.loc[ecipe_mapping["iso3c"].isna(), "ecipe_country"].dropna().tolist()
        ),
    }
    diagnostics_path = out_dir / "social_returns_country_year_panel_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Wrote panel: {panel_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote diagnostics: {diagnostics_path}")
    print(f"Wrote ECIPE mapping: {ecipe_mapping_path}")
    print(f"Wrote MSTI coverage: {msti_coverage_path}")
    print(f"Rows={diagnostics['rows']}, countries={diagnostics['countries']}, years={diagnostics['year_min']}-{diagnostics['year_max']}")


if __name__ == "__main__":
    build_panel()
