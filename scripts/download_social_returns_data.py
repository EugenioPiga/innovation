#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LAW_COLUMNS = [
    "Electronic Transactions",
    "Consumer Protection",
    "Privacy and Data Protection",
    "Cybercrime",
    "Indirect Taxation",
]

LAW_STATUS_MAP = {
    0: "no_data",
    1: "legislation",
    2: "draft_legislation",
    3: "no_legislation",
}


WORLD_BANK_INDICATORS = {
    "GB.XPD.RSDV.GD.ZS": "R&D expenditure (% of GDP)",
    "GB.XPD.RSDV.CD": "R&D expenditure (current US$)",
    "SP.POP.SCIE.RD.P6": "Researchers in R&D (per million people)",
    "IP.PAT.RESD": "Patent applications, residents",
    "IP.PAT.NRES": "Patent applications, nonresidents",
    "IP.JRN.ARTC.SC": "Scientific and technical journal articles",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "IT.NET.SECR.P6": "Secure Internet servers (per 1 million people)",
    "IT.CEL.SETS.P2": "Mobile cellular subscriptions (per 100 people)",
    "NY.GDP.PCAP.KD": "GDP per capita (constant 2015 US$)",
    "NY.GDP.MKTP.KD": "GDP (constant 2015 US$)",
    "NY.GDP.PCAP.PP.KD": "GDP per capita, PPP (constant 2021 intl. $)",
    "NE.GDI.TOTL.ZS": "Gross capital formation (% of GDP)",
    "NE.TRD.GNFS.ZS": "Trade (% of GDP)",
    "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
    "NV.IND.TOTL.ZS": "Industry (including construction), value added (% of GDP)",
    "NV.SRV.TOTL.ZS": "Services, value added (% of GDP)",
    "EN.ATM.CO2E.PC": "CO2 emissions (metric tons per capita)",
}


OECD_FLOWS = {
    "oecd_msti": "DSD_MSTI@DF_MSTI",
    "oecd_berd_ma_toe": "DSD_RDS_BERD@DF_BERD_MA_TOE",
    "oecd_berd_ma_sof": "DSD_RDS_BERD@DF_BERD_MA_SOF",
    "oecd_berd_indu": "DSD_RDS_BERD@DF_BERD_INDU",
    "oecd_gerd_toe": "DSD_RDS_GERD@DF_GERD_TOE",
    "oecd_pers_indu": "DSD_RDS_PERS@DF_PERS_INDU",
    "oecd_pers_ford": "DSD_RDS_PERS@DF_PERS_FORD",
    "oecd_gbard_nabs07": "DSD_RDS_GOV@DF_GBARD_NABS07",
    "oecd_anberd_i4": "DSD_ANBERD@DF_ANBERDi4",
    "oecd_rdtax": "DSD_RDTAX@DF_RDTAX",
    "oecd_rdsub": "DSD_RDTAX@DF_RDSUB",
    "oecd_iptax": "DSD_RDTAX@DF_IPTAX",
    "oecd_biblio": "DSD_BIBLIO@DF_BIBLIO",
}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Collector:
    def __init__(self, root: Path, force: bool, skip_biblio: bool) -> None:
        self.repo_root = root
        self.base_dir = self.repo_root / "datasets" / "social_returns_data"
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.tmp_dir = self.base_dir / "tmp"
        self.force = force
        self.skip_biblio = skip_biblio
        self.catalog: List[Dict[str, Any]] = []

        for path in [self.base_dir, self.raw_dir, self.processed_dir, self.tmp_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "social-returns-data-collector/1.0 "
                    "(research automation; contact: local-run)"
                )
            }
        )

    def add_catalog_record(
        self,
        *,
        dataset_id: str,
        source_name: str,
        url: str,
        file_path: Path,
        method: str,
        status: str,
        notes: str = "",
    ) -> None:
        self.catalog.append(
            {
                "dataset_id": dataset_id,
                "source_name": source_name,
                "url": url,
                "method": method,
                "file_path": str(file_path.relative_to(self.repo_root)),
                "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "status": status,
                "retrieved_at_utc": utc_now_iso(),
                "notes": notes,
            }
        )

    def download(
        self,
        *,
        dataset_id: str,
        source_name: str,
        url: str,
        out_path: Path,
        method: str = "GET",
        data: Dict[str, str] | None = None,
        notes: str = "",
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not self.force:
            self.add_catalog_record(
                dataset_id=dataset_id,
                source_name=source_name,
                url=url,
                file_path=out_path,
                method=method,
                status="cached",
                notes=notes,
            )
            return out_path

        response = self.session.request(
            method=method,
            url=url,
            data=data,
            timeout=180,
            stream=True,
        )
        response.raise_for_status()

        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        self.add_catalog_record(
            dataset_id=dataset_id,
            source_name=source_name,
            url=url,
            file_path=out_path,
            method=method,
            status="downloaded",
            notes=notes,
        )
        return out_path

    def register_generated(
        self,
        *,
        dataset_id: str,
        source_name: str,
        file_path: Path,
        notes: str = "",
    ) -> None:
        self.add_catalog_record(
            dataset_id=dataset_id,
            source_name=source_name,
            url="generated",
            file_path=file_path,
            method="LOCAL",
            status="generated",
            notes=notes,
        )

    def download_with_curl(
        self,
        *,
        dataset_id: str,
        source_name: str,
        url: str,
        out_path: Path,
        method: str = "GET",
        data: Dict[str, str] | None = None,
        notes: str = "",
        extra_headers: Dict[str, str] | None = None,
    ) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not self.force:
            self.add_catalog_record(
                dataset_id=dataset_id,
                source_name=source_name,
                url=url,
                file_path=out_path,
                method=f"{method}+curl",
                status="cached",
                notes=notes,
            )
            return out_path

        command = ["curl.exe", "-L", "-sS", "-X", method, url]
        command.insert(1, "-f")
        headers: Dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)
        for k, v in headers.items():
            command.extend(["-H", f"{k}: {v}"])
        if data:
            for k, v in data.items():
                command.extend(["-d", f"{k}={v}"])
        command.extend(["-o", str(out_path)])

        last_error = ""
        for attempt in range(1, 7):
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                break
            last_error = (
                f"curl attempt {attempt} failed for {url} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
            if out_path.exists():
                out_path.unlink()
            time.sleep(min(5 * attempt, 25))
        else:
            raise RuntimeError(last_error)

        self.add_catalog_record(
            dataset_id=dataset_id,
            source_name=source_name,
            url=url,
            file_path=out_path,
            method=f"{method}+curl",
            status="downloaded",
            notes=notes,
        )
        return out_path

    def collect_unctad_cyberlaw(self) -> None:
        print("Collecting UNCTAD cyberlaw data...")
        js_path = self.download(
            dataset_id="unctad_cyberlaw_js",
            source_name="UNCTAD",
            url="https://unctad.org/sites/default/files/data-file/CyberlawData.js?20250620",
            out_path=self.raw_dir / "regulation" / "unctad_cyberlaw_data.js",
            notes="Embedded data file used in UNCTAD cyberlaw tracker page.",
        )

        text = js_path.read_text(encoding="utf-8")
        date_match = re.search(r"creditText:\s*'Source:\s*UNCTAD,\s*([^']+)'", text)
        countries_match = re.search(r"countries:\s*(\{.*?\})\s*};let statistics", text, re.S)

        if not countries_match:
            raise RuntimeError("Could not parse countries object from UNCTAD CyberlawData.js")

        countries_data = json.loads(countries_match.group(1))
        source_date = date_match.group(1) if date_match else ""

        long_rows: List[Dict[str, Any]] = []
        wide_rows: List[Dict[str, Any]] = []

        for iso2c, values in countries_data.items():
            if not isinstance(values, list) or len(values) != len(LAW_COLUMNS):
                continue

            wide_row: Dict[str, Any] = {"iso2c": iso2c}
            for law_name, code in zip(LAW_COLUMNS, values):
                law_slug = slugify(law_name)
                status = LAW_STATUS_MAP.get(int(code), "unknown")
                wide_row[f"{law_slug}_code"] = int(code)
                wide_row[f"{law_slug}_status"] = status
                long_rows.append(
                    {
                        "iso2c": iso2c,
                        "law_domain": law_name,
                        "law_domain_slug": law_slug,
                        "status_code": int(code),
                        "status_label": status,
                        "source_date_text": source_date,
                    }
                )

            wide_row["source_date_text"] = source_date
            wide_rows.append(wide_row)

        long_df = pd.DataFrame(long_rows).sort_values(["iso2c", "law_domain_slug"])
        wide_df = pd.DataFrame(wide_rows).sort_values(["iso2c"])

        out_long = self.processed_dir / "regulation" / "unctad_cyberlaw_country_long.csv"
        out_wide = self.processed_dir / "regulation" / "unctad_cyberlaw_country_wide.csv"
        out_meta = self.processed_dir / "regulation" / "unctad_cyberlaw_metadata.json"
        out_long.parent.mkdir(parents=True, exist_ok=True)

        long_df.to_csv(out_long, index=False)
        wide_df.to_csv(out_wide, index=False)
        out_meta.write_text(
            json.dumps(
                {
                    "source_date_text": source_date,
                    "law_domains": LAW_COLUMNS,
                    "status_map": LAW_STATUS_MAP,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        self.register_generated(
            dataset_id="unctad_cyberlaw_long",
            source_name="UNCTAD",
            file_path=out_long,
            notes="Parsed long format from CyberlawData.js",
        )
        self.register_generated(
            dataset_id="unctad_cyberlaw_wide",
            source_name="UNCTAD",
            file_path=out_wide,
            notes="Parsed wide format from CyberlawData.js",
        )
        self.register_generated(
            dataset_id="unctad_cyberlaw_meta",
            source_name="UNCTAD",
            file_path=out_meta,
            notes="Parsing metadata.",
        )

    def collect_ecipe_dte(self) -> None:
        print("Collecting ECIPE Digital Trade Environment database export...")
        endpoint = "https://ecipe.org/core/wp-admin/admin-ajax.php"
        post_data = {
            "action": "export_measurements",
            "country": "",
            "type": "",
            "chapter": "",
            "subchapter": "",
            "proposals": "",
        }
        out_path = self.raw_dir / "regulation" / "ecipe_dte_measurements.csv"

        def is_valid_ecipe_export(path: Path) -> bool:
            if not path.exists() or path.stat().st_size == 0:
                return False
            first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not first_line:
                return False
            return first_line[0].startswith("Country,Chapter,Subchapter")

        if out_path.exists() and not self.force and not is_valid_ecipe_export(out_path):
            out_path.unlink()

        try:
            csv_path = self.download(
                dataset_id="ecipe_dte_measurements",
                source_name="ECIPE",
                url=endpoint,
                method="POST",
                data=post_data,
                out_path=out_path,
                notes="Full measurements export from ECIPE DTE index backend endpoint.",
            )
        except requests.HTTPError:
            csv_path = self.download_with_curl(
                dataset_id="ecipe_dte_measurements",
                source_name="ECIPE",
                url=endpoint,
                method="POST",
                data=post_data,
                out_path=out_path,
                notes="Fallback via curl after HTTP error from requests.",
            )

        if not is_valid_ecipe_export(csv_path):
            csv_path = self.download_with_curl(
                dataset_id="ecipe_dte_measurements",
                source_name="ECIPE",
                url=endpoint,
                method="POST",
                data=post_data,
                out_path=out_path,
                notes="Forced curl retry because cached/new output did not look like CSV.",
            )

        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        timeframe = df.get("Timeframe", pd.Series("", index=df.index)).astype(str)
        df["year_from_timeframe"] = timeframe.str.extract(r"((?:19|20)\d{2})", expand=False)

        out_country = self.processed_dir / "regulation" / "ecipe_dte_measures_by_country.csv"
        out_chapter = self.processed_dir / "regulation" / "ecipe_dte_measures_by_chapter.csv"
        out_country_year = (
            self.processed_dir / "regulation" / "ecipe_dte_measures_by_country_year.csv"
        )
        out_country.parent.mkdir(parents=True, exist_ok=True)

        (
            df.groupby("Country", dropna=False)
            .size()
            .reset_index(name="n_measures")
            .sort_values("n_measures", ascending=False)
            .to_csv(out_country, index=False)
        )
        (
            df.groupby("Chapter", dropna=False)
            .size()
            .reset_index(name="n_measures")
            .sort_values("n_measures", ascending=False)
            .to_csv(out_chapter, index=False)
        )
        (
            df.dropna(subset=["year_from_timeframe"])
            .groupby(["Country", "year_from_timeframe"], dropna=False)
            .size()
            .reset_index(name="n_measures")
            .sort_values(["Country", "year_from_timeframe"])
            .to_csv(out_country_year, index=False)
        )

        self.register_generated(
            dataset_id="ecipe_dte_country_summary",
            source_name="ECIPE",
            file_path=out_country,
            notes="Count of measures by country.",
        )
        self.register_generated(
            dataset_id="ecipe_dte_chapter_summary",
            source_name="ECIPE",
            file_path=out_chapter,
            notes="Count of measures by chapter.",
        )
        self.register_generated(
            dataset_id="ecipe_dte_country_year_summary",
            source_name="ECIPE",
            file_path=out_country_year,
            notes="Count of measures by country-year using year parsed from Timeframe text.",
        )

    def collect_datagouv_privacy(self) -> None:
        print("Collecting Data.gouv global privacy map dataset...")
        csv_path = self.download(
            dataset_id="datagouv_privacy_csv",
            source_name="data.gouv.fr",
            url="https://www.data.gouv.fr/api/1/datasets/r/4896dbe0-dafa-448d-81b5-ef7fdc26ddb7",
            out_path=self.raw_dir / "regulation" / "datagouv_global_privacy_mapping.csv",
            notes="CSV resource listed in data.gouv.fr dataset 'Global map for personal data protection'.",
        )
        self.download(
            dataset_id="datagouv_privacy_xlsx",
            source_name="data.gouv.fr",
            url="https://www.data.gouv.fr/api/1/datasets/r/00037755-4460-421f-be37-00dedbd66994",
            out_path=self.raw_dir / "regulation" / "datagouv_global_privacy_mapping.xlsx",
            notes="XLSX resource listed in data.gouv.fr dataset 'Global map for personal data protection'.",
        )

        # Normalize delimiter for easier use downstream.
        sample = csv_path.read_text(encoding="utf-8", errors="replace")[:8000]
        sep = ";" if sample.count(";") > sample.count(",") else ","
        normalized_path = (
            self.processed_dir / "regulation" / "datagouv_global_privacy_mapping_normalized.csv"
        )
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(csv_path, sep=sep, dtype=str, low_memory=False).to_csv(
            normalized_path, index=False
        )
        self.register_generated(
            dataset_id="datagouv_privacy_normalized_csv",
            source_name="data.gouv.fr",
            file_path=normalized_path,
            notes="CSV normalized to comma-delimited UTF-8 output.",
        )

    def collect_oecd(self) -> None:
        print("Collecting OECD STI datasets via SDMX API...")
        summary_rows: List[Dict[str, Any]] = []
        for dataset_id, flow in OECD_FLOWS.items():
            if dataset_id == "oecd_biblio" and self.skip_biblio:
                continue
            url = f"https://sdmx.oecd.org/public/rest/data/OECD.STI.STP,{flow},1.0/.?format=csvfile"
            out_path = self.raw_dir / "oecd" / f"{dataset_id}.csv"
            downloaded_path = self.download(
                dataset_id=dataset_id,
                source_name="OECD SDMX API",
                url=url,
                out_path=out_path,
                notes=f"Flow: {flow}",
            )
            with downloaded_path.open("r", encoding="utf-8", errors="replace") as f:
                header = f.readline().strip()
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "flow": flow,
                    "file_path": str(downloaded_path.relative_to(self.repo_root)),
                    "file_size_bytes": downloaded_path.stat().st_size,
                    "column_count": len(header.split(",")) if header else 0,
                }
            )

        workbook_path = self.download(
            dataset_id="oecd_rdtax_workbook",
            source_name="OECD",
            url="https://oe.cd/ds/rdtax",
            out_path=self.raw_dir / "oecd" / "oecd_rdtax_workbook.xlsx",
            notes="OECD short-link download for the R&D tax support dataset workbook.",
        )
        summary_rows.append(
            {
                "dataset_id": "oecd_rdtax_workbook",
                "flow": "workbook",
                "file_path": str(workbook_path.relative_to(self.repo_root)),
                "file_size_bytes": workbook_path.stat().st_size,
                "column_count": "",
            }
        )

        summary_path = self.processed_dir / "oecd" / "oecd_download_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        self.register_generated(
            dataset_id="oecd_download_summary",
            source_name="OECD",
            file_path=summary_path,
            notes="Quick summary of OECD downloads.",
        )

    def collect_world_bank_wdi(self) -> None:
        print("Collecting World Bank WDI indicators...")
        all_frames: List[pd.DataFrame] = []
        indicator_meta_rows: List[Dict[str, str]] = []

        raw_indicator_dir = self.raw_dir / "world_bank"
        raw_indicator_dir.mkdir(parents=True, exist_ok=True)

        for indicator_id, indicator_label in WORLD_BANK_INDICATORS.items():
            print(f"  - {indicator_id}")
            page = 1
            records: List[Dict[str, Any]] = []

            while True:
                url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_id}"
                response = self.session.get(
                    url,
                    params={"format": "json", "per_page": 20000, "page": page},
                    timeout=180,
                )
                response.raise_for_status()
                payload = response.json()

                if not isinstance(payload, list) or len(payload) < 2:
                    break

                meta = payload[0]
                chunk = payload[1] or []
                records.extend(chunk)

                pages = int(meta.get("pages", 1))
                if page >= pages:
                    break
                page += 1

            rows: List[Dict[str, Any]] = []
            for rec in records:
                rows.append(
                    {
                        "indicator_id": indicator_id,
                        "indicator_name": indicator_label,
                        "countryiso3code": rec.get("countryiso3code"),
                        "country_name": (rec.get("country") or {}).get("value"),
                        "year": rec.get("date"),
                        "value": rec.get("value"),
                        "unit": rec.get("unit"),
                        "obs_status": rec.get("obs_status"),
                        "decimal": rec.get("decimal"),
                    }
                )

            frame = pd.DataFrame(rows)
            per_indicator_path = raw_indicator_dir / f"wdi_{indicator_id}.csv"
            frame.to_csv(per_indicator_path, index=False)
            self.register_generated(
                dataset_id=f"world_bank_wdi_{indicator_id}",
                source_name="World Bank API",
                file_path=per_indicator_path,
                notes=f"{indicator_label}",
            )
            all_frames.append(frame)
            indicator_meta_rows.append(
                {"indicator_id": indicator_id, "indicator_name": indicator_label}
            )

        if not all_frames:
            return

        long_df = pd.concat(all_frames, ignore_index=True)
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")

        wide_df = (
            long_df.dropna(subset=["countryiso3code", "year"])
            .pivot_table(
                index=["countryiso3code", "country_name", "year"],
                columns="indicator_id",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        out_long = self.processed_dir / "world_bank" / "wdi_selected_indicators_long.csv"
        out_wide = self.processed_dir / "world_bank" / "wdi_selected_indicators_wide.csv"
        out_meta = self.processed_dir / "world_bank" / "wdi_selected_indicator_metadata.csv"
        out_long.parent.mkdir(parents=True, exist_ok=True)

        long_df.to_csv(out_long, index=False)
        wide_df.to_csv(out_wide, index=False)
        pd.DataFrame(indicator_meta_rows).to_csv(out_meta, index=False)

        self.register_generated(
            dataset_id="world_bank_wdi_long",
            source_name="World Bank API",
            file_path=out_long,
            notes="Stacked panel for selected indicators.",
        )
        self.register_generated(
            dataset_id="world_bank_wdi_wide",
            source_name="World Bank API",
            file_path=out_wide,
            notes="Country-year wide panel for selected indicators.",
        )
        self.register_generated(
            dataset_id="world_bank_wdi_metadata",
            source_name="World Bank API",
            file_path=out_meta,
            notes="Indicator codebook used in this pull.",
        )

    def collect_pwt(self) -> None:
        print("Collecting Penn World Table files...")
        pwt_files = {
            "pwt100_main_xlsx": "https://www.rug.nl/ggdc/docs/pwt100.xlsx",
            "pwt100_na_data_xlsx": "https://www.rug.nl/ggdc/docs/pwt100-na-data.xlsx",
            "pwt100_capital_detail_xlsx": "https://www.rug.nl/ggdc/docs/pwt100-capital-detail.xlsx",
            "pwt100_user_guide_pdf": "https://www.rug.nl/ggdc/docs/pwt100-user-guide-to-data-files.pdf",
            "pwt100_changelog_pdf": "https://www.rug.nl/ggdc/docs/pwt100-changelog.pdf",
        }
        for dataset_id, url in pwt_files.items():
            extension = Path(url).suffix
            self.download(
                dataset_id=dataset_id,
                source_name="Penn World Table (GGDC)",
                url=url,
                out_path=self.raw_dir / "pwt" / f"{dataset_id}{extension}",
            )

    def write_catalog(self) -> None:
        if not self.catalog:
            return
        catalog_df = pd.DataFrame(self.catalog).sort_values(["dataset_id", "file_path"])
        csv_path = self.base_dir / "dataset_catalog.csv"
        json_path = self.base_dir / "dataset_catalog.json"
        catalog_df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(catalog_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        print(f"Catalog written: {csv_path}")
        print(f"Catalog written: {json_path}")

    def run(self) -> None:
        self.collect_unctad_cyberlaw()
        self.collect_ecipe_dte()
        self.collect_datagouv_privacy()
        self.collect_oecd()
        self.collect_world_bank_wdi()
        self.collect_pwt()
        self.write_catalog()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize datasets for social returns to data regulation research."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--skip-biblio",
        action="store_true",
        help="Skip very large OECD bibliometrics pull (~200MB).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    collector = Collector(repo_root, force=args.force, skip_biblio=args.skip_biblio)
    collector.run()


if __name__ == "__main__":
    main()
