from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .utils import cache_path, safe_ratio, to_cik_str


COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_no_nodash}/{doc_name}"

# Public proxies aligned with NBER w33171 measurement themes.
TAG_REVENUE = ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"]
TAG_SALES_AND_MARKETING = ["SellingAndMarketingExpense"]
TAG_SGA = ["SellingGeneralAndAdministrativeExpense"]
TAG_RD = ["ResearchAndDevelopmentExpense"]
TAG_CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment"]
TAG_EMPLOYEES = ["NumberOfEmployees"]

TEXT_KEYWORDS = {
    "kw_sales_force": ["sales force", "sales team", "account executive", "go-to-market"],
    "kw_customer_service": ["customer service", "support team", "customer success"],
    "kw_brand": ["brand", "branding", "brand awareness"],
    "kw_advertising": ["advertising", "ad spend", "promotion", "promotional"],
    "kw_customer_data": ["customer data", "user data", "crm", "analytics", "personalization"],
}


def _session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept": "application/json, text/html"})
    return s


def _cached_get_json(session: requests.Session, url: str, cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    cache_file.write_text(resp.text, encoding="utf-8")
    return resp.json()


def _cached_get_text(session: requests.Session, url: str, cache_file: Path) -> str:
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    cache_file.write_text(resp.text, encoding="utf-8", errors="ignore")
    return resp.text


def _extract_metric_by_year(companyfacts: dict, tags: list[str], units_whitelist: set[str] | None = None) -> pd.DataFrame:
    us_gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    rows: list[tuple[int, float]] = []
    for tag in tags:
        node = us_gaap.get(tag)
        if not node:
            continue
        units_map = node.get("units", {})
        for unit, items in units_map.items():
            if units_whitelist is not None and unit not in units_whitelist:
                continue
            for item in items:
                fy = item.get("fy")
                val = item.get("val")
                form = item.get("form", "")
                if fy is None or val is None:
                    continue
                if form not in {"10-K", "10-K/A", "20-F", "40-F"}:
                    continue
                rows.append((int(fy), float(val)))
    if not rows:
        return pd.DataFrame(columns=["year", "value"])
    frame = pd.DataFrame(rows, columns=["year", "value"])
    frame = frame.sort_values(["year"]).drop_duplicates(subset=["year"], keep="last")
    return frame


def _extract_filings_index(submissions: dict) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()
    fields = ["accessionNumber", "filingDate", "reportDate", "form", "primaryDocument"]
    data = {k: recent.get(k, []) for k in fields}
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    frame = frame[frame["form"].isin(["10-K", "10-K/A", "20-F", "40-F"])].copy()
    frame["year"] = pd.to_datetime(frame["reportDate"].fillna(frame["filingDate"]), errors="coerce").dt.year
    frame = frame.dropna(subset=["year", "accessionNumber", "primaryDocument"]) 
    frame["year"] = frame["year"].astype(int)
    return frame


def _clean_html_text(raw_html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _keyword_features(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, terms in TEXT_KEYWORDS.items():
        out[key] = sum(text.count(term) for term in terms)
    out["customer_capital_keyword_total"] = int(sum(out.values()))
    return out


def _build_text_features_for_cik(session: requests.Session, cik: str, submissions: dict, cache_dir: Path) -> pd.DataFrame:
    filings = _extract_filings_index(submissions)
    if filings.empty:
        return pd.DataFrame(columns=["cik", "year", "customer_capital_keyword_total"])

    rows: list[dict] = []
    cik_nozero = str(int(cik))
    for row in filings.itertuples(index=False):
        acc_no_nodash = str(row.accessionNumber).replace("-", "")
        doc_name = str(row.primaryDocument)
        url = ARCHIVES_DOC_URL.format(cik_nozero=cik_nozero, acc_no_nodash=acc_no_nodash, doc_name=doc_name)
        cache_file = cache_dir / f"filing_{cik}_{acc_no_nodash}.html"
        try:
            html_text = _cached_get_text(session, url, cache_file)
        except requests.RequestException:
            continue
        text = _clean_html_text(html_text)
        feats = _keyword_features(text)
        feats.update({"cik": cik, "year": int(row.year)})
        rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=["cik", "year", "customer_capital_keyword_total"])
    return pd.DataFrame(rows).groupby(["cik", "year"], as_index=False).max()


def build_sec_firm_year_panel(
    ciks: list[str],
    cache_dir: Path,
    user_agent: str,
    include_text_features: bool = True,
) -> pd.DataFrame:
    session = _session(user_agent)
    records: list[pd.DataFrame] = []

    for cik in tqdm(ciks, desc="SEC firm enrichment"):
        cik = to_cik_str(cik)
        facts_path = cache_path(cache_dir, f"companyfacts_{cik}")
        subm_path = cache_path(cache_dir, f"submissions_{cik}")

        try:
            companyfacts = _cached_get_json(session, COMPANYFACTS_URL.format(cik=cik), facts_path)
            submissions = _cached_get_json(session, SUBMISSIONS_URL.format(cik=cik), subm_path)
        except requests.RequestException:
            continue

        rev = _extract_metric_by_year(companyfacts, TAG_REVENUE, units_whitelist={"USD"}).rename(columns={"value": "revenue"})
        sm = _extract_metric_by_year(companyfacts, TAG_SALES_AND_MARKETING, units_whitelist={"USD"}).rename(columns={"value": "sales_marketing_expense"})
        sga = _extract_metric_by_year(companyfacts, TAG_SGA, units_whitelist={"USD"}).rename(columns={"value": "sga_expense"})
        rd = _extract_metric_by_year(companyfacts, TAG_RD, units_whitelist={"USD"}).rename(columns={"value": "rd_expense"})
        capex = _extract_metric_by_year(companyfacts, TAG_CAPEX, units_whitelist={"USD"}).rename(columns={"value": "capex"})
        emp = _extract_metric_by_year(companyfacts, TAG_EMPLOYEES, units_whitelist=None).rename(columns={"value": "employees"})

        base = rev.copy()
        for part in [sm, sga, rd, capex, emp]:
            base = base.merge(part, on="year", how="outer")
        if base.empty:
            continue

        base["cik"] = cik
        base["sales_marketing_to_revenue"] = [safe_ratio(a, b) for a, b in zip(base.get("sales_marketing_expense"), base.get("revenue"))]
        base["sga_to_revenue"] = [safe_ratio(a, b) for a, b in zip(base.get("sga_expense"), base.get("revenue"))]
        base["rd_to_revenue"] = [safe_ratio(a, b) for a, b in zip(base.get("rd_expense"), base.get("revenue"))]
        base["capex_to_revenue"] = [safe_ratio(a, b) for a, b in zip(base.get("capex"), base.get("revenue"))]

        if include_text_features:
            text_df = _build_text_features_for_cik(session, cik, submissions, cache_dir)
            base = base.merge(text_df, on=["cik", "year"], how="left")

        profile = {
            "sic": submissions.get("sic"),
            "sic_description": submissions.get("sicDescription"),
            "state_of_incorporation": submissions.get("stateOfIncorporation"),
            "fiscal_year_end": submissions.get("fiscalYearEnd"),
            "entity_type": submissions.get("entityType"),
            "name_sec": submissions.get("name"),
            "ticker_sec": (submissions.get("tickers") or [None])[0],
        }
        for k, v in profile.items():
            base[k] = v

        records.append(base)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)