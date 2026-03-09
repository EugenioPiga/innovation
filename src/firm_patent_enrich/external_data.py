from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

from .utils import to_int_or_none


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_TS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
CENSUS_CBP_URL = "https://api.census.gov/data/{year}/cbp"
BEA_URL = "https://apps.bea.gov/api/data/"

DEFAULT_FRED_SERIES = {
    "FEDFUNDS": "fred_fedfunds",
    "DGS10": "fred_treasury_10y",
    "CPIAUCSL": "fred_cpi_all_items",
    "UNRATE": "fred_unemployment_rate",
}

# Public OEWS/OES proxies (series IDs are transparent and user-overridable).
DEFAULT_BLS_OES_SERIES = {
    "OEUN000000000000011301204": "bls_oes_113012_204",
    "OEUN000000000000011301208": "bls_oes_113012_208",
    "OEUN000000000000011301212": "bls_oes_113012_212",
    "OEUN000000000000000000001": "bls_oes_total_employment",
}

CENSUS_NAICS_BUCKETS = [
    "11",
    "21",
    "22",
    "23",
    "31-33",
    "42",
    "44-45",
    "48-49",
    "51",
    "52",
    "53",
    "54",
    "55",
    "56",
    "61",
    "62",
    "71",
    "72",
    "81",
]


def _safe_get_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 60) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_fred_macro_panel(
    cache_dir: Path,
    start_year: int,
    end_year: int,
    api_key: str | None = None,
    series_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    series_map = series_map or DEFAULT_FRED_SERIES
    panels: list[pd.DataFrame] = []

    for fred_id, col_name in series_map.items():
        cache_file = cache_dir / f"fred_{fred_id}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
        elif api_key:
            params = {
                "series_id": fred_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": f"{start_year}-01-01",
                "observation_end": f"{end_year}-12-31",
            }
            try:
                payload = _safe_get_json(FRED_API_URL, params=params)
                obs = payload.get("observations", [])
                if not obs:
                    continue
                df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"date": "DATE", "value": fred_id})
                df.to_csv(cache_file, index=False)
            except requests.RequestException:
                continue
        else:
            try:
                response = requests.get(FRED_CSV_URL.format(series_id=fred_id), timeout=20)
                response.raise_for_status()
                raw = response.text
                cache_file.write_text(raw, encoding="utf-8")
                df = pd.read_csv(io.StringIO(raw))
            except requests.RequestException:
                continue

        if "DATE" not in df.columns or fred_id not in df.columns:
            continue

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df[col_name] = pd.to_numeric(df[fred_id], errors="coerce")
        df = df.dropna(subset=["DATE", col_name])
        if df.empty:
            continue

        df["year"] = df["DATE"].dt.year
        annual = df.groupby("year", as_index=False)[col_name].mean()
        annual = annual[(annual["year"] >= start_year) & (annual["year"] <= end_year)]
        panels.append(annual)

    if not panels:
        return pd.DataFrame(columns=["year"])

    out = panels[0]
    for p in panels[1:]:
        out = out.merge(p, on="year", how="outer")
    return out.sort_values("year").reset_index(drop=True)


def load_bls_series_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_BLS_OES_SERIES)
    df = pd.read_csv(path)
    required = {"series_id", "metric_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BLS series map missing columns: {sorted(missing)}")
    return {str(r.series_id): str(r.metric_name) for r in df.itertuples(index=False)}


def fetch_bls_oes_panel(
    cache_dir: Path,
    start_year: int,
    end_year: int,
    series_map: dict[str, str],
    api_key: str | None = None,
) -> pd.DataFrame:
    if not series_map:
        return pd.DataFrame(columns=["year"])

    all_rows: list[dict] = []
    series_ids = list(series_map.keys())

    for i in range(0, len(series_ids), 50):
        batch = series_ids[i : i + 50]
        cache_file = cache_dir / f"bls_oes_{i}_{start_year}_{end_year}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            body: dict[str, object] = {
                "seriesid": batch,
                "startyear": str(start_year),
                "endyear": str(end_year),
            }
            if api_key:
                body["registrationkey"] = api_key
            try:
                resp = requests.post(BLS_TS_URL, json=body, timeout=90)
                resp.raise_for_status()
                payload = resp.json()
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            except requests.RequestException:
                continue

        for series in payload.get("Results", {}).get("series", []):
            sid = str(series.get("seriesID", ""))
            metric_name = series_map.get(sid)
            if not metric_name:
                continue
            for item in series.get("data", []):
                if item.get("period") != "A01":
                    continue
                year = to_int_or_none(item.get("year"))
                value = pd.to_numeric(item.get("value"), errors="coerce")
                if year is None or pd.isna(value):
                    continue
                all_rows.append({"year": year, "metric_name": metric_name, "value": float(value)})

    if not all_rows:
        return pd.DataFrame(columns=["year"])

    df = pd.DataFrame(all_rows)
    wide = df.pivot_table(index="year", columns="metric_name", values="value", aggfunc="mean").reset_index()
    wide.columns.name = None
    return wide.sort_values("year").reset_index(drop=True)


def map_sic_to_naics_bucket(sic: str | int | None) -> str | None:
    if sic is None or pd.isna(sic):
        return None
    sic_str = str(sic).strip()
    if not sic_str:
        return None
    try:
        sic2 = int(sic_str[:2])
    except ValueError:
        return None

    if 1 <= sic2 <= 9:
        return "11"
    if 10 <= sic2 <= 14:
        return "21"
    if 15 <= sic2 <= 17:
        return "23"
    if 20 <= sic2 <= 39:
        return "31-33"
    if 40 <= sic2 <= 47:
        return "48-49"
    if sic2 == 48:
        return "51"
    if sic2 == 49:
        return "22"
    if 50 <= sic2 <= 51:
        return "42"
    if 52 <= sic2 <= 59:
        return "44-45"
    if 60 <= sic2 <= 67:
        return "52"
    if 70 <= sic2 <= 89:
        return "54"
    return None


def fetch_census_cbp_industry_panel(cache_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict] = []

    for year in range(start_year, end_year + 1):
        for naics in CENSUS_NAICS_BUCKETS:
            cache_file = cache_dir / f"census_cbp_{year}_{naics.replace('-', '_')}.json"
            if cache_file.exists():
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                params = {
                    "get": "NAICS2017,EMPSZES,EMP,PAYANN,ESTAB",
                    "for": "us:1",
                    "NAICS2017": naics,
                }
                try:
                    payload = _safe_get_json(CENSUS_CBP_URL.format(year=year), params=params)
                except requests.RequestException:
                    continue
                cache_file.write_text(json.dumps(payload), encoding="utf-8")

            if not payload or len(payload) < 2:
                continue
            columns = payload[0]
            data_rows = payload[1:]
            df = pd.DataFrame(data_rows, columns=columns)
            df = df[df["EMPSZES"] == "001"]
            if df.empty:
                continue
            row = df.iloc[0]
            emp = pd.to_numeric(row.get("EMP"), errors="coerce")
            payann = pd.to_numeric(row.get("PAYANN"), errors="coerce")
            estab = pd.to_numeric(row.get("ESTAB"), errors="coerce")
            payroll_per_employee = (payann * 1000 / emp) if pd.notna(payann) and pd.notna(emp) and emp > 0 else pd.NA

            rows.append(
                {
                    "year": year,
                    "naics2_proxy": naics,
                    "census_cbp_emp": emp,
                    "census_cbp_payann_thousand": payann,
                    "census_cbp_estab": estab,
                    "census_cbp_payroll_per_employee": payroll_per_employee,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["year", "naics2_proxy"])

    return pd.DataFrame(rows)


def fetch_bea_gdp_by_industry_panel(
    cache_dir: Path,
    start_year: int,
    end_year: int,
    api_key: str | None,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["year", "naics2_proxy", "bea_gdp_by_industry_value"])

    cache_file = cache_dir / f"bea_gdp_by_industry_{start_year}_{end_year}.json"
    if cache_file.exists():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        params = {
            "UserID": api_key,
            "method": "GetData",
            "datasetname": "GDPbyIndustry",
            "Industry": "ALL",
            "TableID": "1",
            "Frequency": "A",
            "Year": "ALL",
            "ResultFormat": "JSON",
        }
        try:
            payload = _safe_get_json(BEA_URL, params=params)
        except requests.RequestException:
            return pd.DataFrame(columns=["year", "naics2_proxy", "bea_gdp_by_industry_value"])
        cache_file.write_text(json.dumps(payload), encoding="utf-8")

    data = payload.get("BEAAPI", {}).get("Results", {}).get("Data", [])
    if not data:
        return pd.DataFrame(columns=["year", "naics2_proxy", "bea_gdp_by_industry_value"])

    rows: list[dict] = []
    for item in data:
        year = to_int_or_none(item.get("Year"))
        if year is None or year < start_year or year > end_year:
            continue

        industry_code = str(item.get("Industry", "")).strip()
        value = pd.to_numeric(str(item.get("DataValue", "")).replace(",", ""), errors="coerce")
        if pd.isna(value):
            continue

        # Keep broad NAICS-like roots only for stable merge with SIC proxy.
        naics2_proxy = None
        if industry_code in CENSUS_NAICS_BUCKETS:
            naics2_proxy = industry_code
        elif industry_code[:2].isdigit() and industry_code[:2] in {"11", "21", "22", "23", "42", "51", "52", "53", "54", "55", "56", "61", "62", "71", "72", "81"}:
            naics2_proxy = industry_code[:2]
        elif industry_code.startswith("31") or industry_code.startswith("32") or industry_code.startswith("33"):
            naics2_proxy = "31-33"
        elif industry_code.startswith("44") or industry_code.startswith("45"):
            naics2_proxy = "44-45"
        elif industry_code.startswith("48") or industry_code.startswith("49"):
            naics2_proxy = "48-49"

        if naics2_proxy is None:
            continue

        rows.append(
            {
                "year": year,
                "naics2_proxy": naics2_proxy,
                "bea_gdp_by_industry_value": float(value),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["year", "naics2_proxy", "bea_gdp_by_industry_value"])

    return pd.DataFrame(rows).groupby(["year", "naics2_proxy"], as_index=False).mean()
