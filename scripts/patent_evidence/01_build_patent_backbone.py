#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPSTAT_DIR = REPO_ROOT / "datasets" / "compustat_patents_data"
PATENTSVIEW_RAW_DIR = REPO_ROOT / "datasets" / "patentsview" / "raw"
OUT_DIR = REPO_ROOT / "datasets" / "patent_evidence"
INTERMEDIATE_DIR = OUT_DIR / "intermediate"
OUTPUT_DIR = OUT_DIR / "output"


def _strip_quotes(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.strip('"')


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def _file_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _load_patent_universe(chunksize: int) -> pd.DataFrame:
    usecols = ["patent_id", "appYear", "grantYear"]
    files = sorted(COMPSTAT_DIR.glob("staticTranche*.csv"))
    if not files:
        full_static = COMPSTAT_DIR / "static.csv"
        if not full_static.exists():
            raise FileNotFoundError("No static patent files found in compustat_patents_data.")
        files = [full_static]

    frames: list[pd.DataFrame] = []
    for file_path in files:
        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            dtype={"patent_id": "string", "appYear": "Int64", "grantYear": "Int64"},
            chunksize=chunksize,
        ):
            chunk["patent_id"] = _strip_quotes(chunk["patent_id"]).str.upper()
            chunk = chunk.dropna(subset=["patent_id"])
            frames.append(chunk)

    universe = pd.concat(frames, ignore_index=True)
    universe = (
        universe.groupby("patent_id", as_index=False)
        .agg(
            app_year=("appYear", "min"),
            grant_year=("grantYear", "max"),
        )
        .sort_values("patent_id")
        .reset_index(drop=True)
    )
    return universe


def _stream_filtered_table(
    *,
    zip_path: Path,
    usecols: list[str],
    patent_set: set[str],
    output_csv: Path,
    chunksize: int,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    total = 0
    first = True
    for chunk in pd.read_csv(
        zip_path,
        sep="\t",
        compression="zip",
        usecols=usecols,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = _clean_columns(chunk)
        for c in chunk.columns:
            chunk[c] = _strip_quotes(chunk[c])
        chunk["patent_id"] = chunk["patent_id"].str.upper()
        chunk = chunk[chunk["patent_id"].isin(patent_set)]
        if chunk.empty:
            continue
        chunk.to_csv(output_csv, mode="a", index=False, header=first)
        first = False
        total += len(chunk)
    return total


def _build_location_map() -> dict[str, str]:
    loc_zip = PATENTSVIEW_RAW_DIR / "g_location_disambiguated.tsv.zip"
    if not loc_zip.exists():
        raise FileNotFoundError(f"Missing file: {loc_zip}")

    location = pd.read_csv(loc_zip, sep="\t", compression="zip", dtype="string", low_memory=False)
    location = _clean_columns(location)
    for c in location.columns:
        location[c] = _strip_quotes(location[c])
    location["location_id"] = location["location_id"].str.lower()
    location["disambig_country"] = location["disambig_country"].str.upper()
    location = location.dropna(subset=["location_id"])
    return dict(zip(location["location_id"], location["disambig_country"]))


def _stream_country_raw(
    *,
    zip_path: Path,
    usecols: list[str],
    patent_set: set[str],
    location_map: dict[str, str],
    sequence_col: str,
    output_csv: Path,
    chunksize: int,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    total = 0
    first = True
    for chunk in pd.read_csv(
        zip_path,
        sep="\t",
        compression="zip",
        usecols=usecols,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = _clean_columns(chunk)
        for c in chunk.columns:
            chunk[c] = _strip_quotes(chunk[c])
        chunk["patent_id"] = chunk["patent_id"].str.upper()
        chunk = chunk[chunk["patent_id"].isin(patent_set)]
        if chunk.empty:
            continue
        chunk["location_id"] = chunk["location_id"].str.lower()
        chunk["country"] = chunk["location_id"].map(location_map)
        chunk[sequence_col] = pd.to_numeric(chunk[sequence_col], errors="coerce").fillna(999).astype("Int64")
        out = chunk[["patent_id", sequence_col, "country"]].copy()
        out.to_csv(output_csv, mode="a", index=False, header=first)
        first = False
        total += len(out)
    return total


def _build_cpc_aggregates(
    *,
    zip_path: Path,
    patent_set: set[str],
    output_csv: Path,
    chunksize: int,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    measurement_prefixes = ("G01", "A61B5", "G06M", "G01N")
    testing_prefixes = ("G01M", "G06F11", "G05B23", "H04L43")
    ml_prefixes = ("G06N", "G06V", "G16H")
    data_proc_prefixes = ("G06F17", "G06F19", "G06Q", "H04L")
    bio_assay_prefixes = ("C12Q", "C12M", "A61B10", "G01N33")
    statistics_prefixes = ("G06F17/18", "G06N7", "G06F16")

    total = 0
    first = True
    for chunk in pd.read_csv(
        zip_path,
        sep="\t",
        compression="zip",
        usecols=["patent_id", "cpc_section", "cpc_class", "cpc_subclass", "cpc_group", "cpc_type"],
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = _clean_columns(chunk)
        for c in chunk.columns:
            chunk[c] = _strip_quotes(chunk[c]).str.upper()
        chunk = chunk.rename(columns={"cpc_group": "cpc_group_full"})
        chunk["patent_id"] = chunk["patent_id"].str.upper()
        chunk = chunk[chunk["patent_id"].isin(patent_set)]
        if chunk.empty:
            continue

        chunk["has_measurement_cpc"] = chunk["cpc_subclass"].str.startswith(measurement_prefixes, na=False).astype(int)
        chunk["has_testing_cpc"] = chunk["cpc_subclass"].str.startswith(testing_prefixes, na=False).astype(int)
        chunk["has_ml_cpc"] = chunk["cpc_subclass"].str.startswith(ml_prefixes, na=False).astype(int)
        chunk["has_data_processing_cpc"] = chunk["cpc_subclass"].str.startswith(data_proc_prefixes, na=False).astype(int)
        chunk["has_bio_assay_cpc"] = chunk["cpc_subclass"].str.startswith(bio_assay_prefixes, na=False).astype(int)
        chunk["has_statistics_cpc"] = (
            chunk["cpc_group_full"].str.startswith(statistics_prefixes, na=False)
            | chunk["cpc_subclass"].str.startswith(("G06N7",), na=False)
        ).astype(int)

        for sec in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            chunk[f"cpc_section_{sec}_count"] = (chunk["cpc_section"] == sec).astype(int)

        out = chunk[
            [
                "patent_id",
                "cpc_subclass",
                "cpc_section",
                "has_measurement_cpc",
                "has_testing_cpc",
                "has_ml_cpc",
                "has_data_processing_cpc",
                "has_bio_assay_cpc",
                "has_statistics_cpc",
            ]
        ]
        out.to_csv(output_csv, mode="a", index=False, header=first)
        first = False
        total += len(out)

    return total


def build_backbone(chunksize: int = 500_000, reuse_intermediate: bool = True) -> dict[str, int]:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(INTERMEDIATE_DIR / "patent_evidence.duckdb"))

    universe_path = INTERMEDIATE_DIR / "patent_universe.csv"
    patent_core_path = INTERMEDIATE_DIR / "patent_core_filtered.csv"
    patent_abs_path = INTERMEDIATE_DIR / "patent_abstract_filtered.csv"
    assignee_raw_path = INTERMEDIATE_DIR / "assignee_country_raw.csv"
    inventor_raw_path = INTERMEDIATE_DIR / "inventor_country_raw.csv"
    cpc_agg_chunk_path = INTERMEDIATE_DIR / "cpc_agg_chunk.csv"
    required_intermediate = [
        universe_path,
        patent_core_path,
        patent_abs_path,
        assignee_raw_path,
        inventor_raw_path,
        cpc_agg_chunk_path,
    ]
    have_all_intermediate = all(_file_has_content(p) for p in required_intermediate)

    n_core: int | None = None
    n_abs: int | None = None
    n_assignee: int | None = None
    n_inventor: int | None = None
    n_cpc_rows: int | None = None
    universe_rows: int | None = None

    if reuse_intermediate and have_all_intermediate:
        print("Reusing existing intermediate filtered files.")
    else:
        print("Loading Compustat patent universe...")
        universe = _load_patent_universe(chunksize=chunksize)
        universe_rows = int(len(universe))
        universe.to_csv(universe_path, index=False)
        patent_set = set(universe["patent_id"].astype(str).tolist())
        print(f"Universe patents: {universe_rows:,}")

        # 1) Core patent metadata (title, date, claims)
        print("Filtering PatentsView patent core table...")
        n_core = _stream_filtered_table(
            zip_path=PATENTSVIEW_RAW_DIR / "g_patent.tsv.zip",
            usecols=[
                "patent_id",
                "patent_type",
                "patent_date",
                "patent_title",
                "num_claims",
            ],
            patent_set=patent_set,
            output_csv=patent_core_path,
            chunksize=chunksize,
        )

        # 2) Abstracts
        print("Filtering PatentsView abstract table...")
        n_abs = _stream_filtered_table(
            zip_path=PATENTSVIEW_RAW_DIR / "g_patent_abstract.tsv.zip",
            usecols=["patent_id", "patent_abstract"],
            patent_set=patent_set,
            output_csv=patent_abs_path,
            chunksize=chunksize,
        )

        # 3) Country assignment raw rows from assignee / inventor
        print("Building location map and filtering assignee/inventor tables...")
        location_map = _build_location_map()

        n_assignee = _stream_country_raw(
            zip_path=PATENTSVIEW_RAW_DIR / "g_assignee_disambiguated.tsv.zip",
            usecols=["patent_id", "assignee_sequence", "location_id"],
            patent_set=patent_set,
            location_map=location_map,
            sequence_col="assignee_sequence",
            output_csv=assignee_raw_path,
            chunksize=chunksize,
        )

        n_inventor = _stream_country_raw(
            zip_path=PATENTSVIEW_RAW_DIR / "g_inventor_disambiguated.tsv.zip",
            usecols=["patent_id", "inventor_sequence", "location_id"],
            patent_set=patent_set,
            location_map=location_map,
            sequence_col="inventor_sequence",
            output_csv=inventor_raw_path,
            chunksize=chunksize,
        )

        # 4) CPC aggregates
        print("Filtering CPC table...")
        n_cpc_rows = _build_cpc_aggregates(
            zip_path=PATENTSVIEW_RAW_DIR / "g_cpc_current.tsv.zip",
            patent_set=patent_set,
            output_csv=cpc_agg_chunk_path,
            chunksize=chunksize,
        )

    # 5) Load all intermediate tables into DuckDB
    con.execute("DROP TABLE IF EXISTS universe")
    con.execute("DROP TABLE IF EXISTS patent_core")
    con.execute("DROP TABLE IF EXISTS patent_abs")
    con.execute("DROP TABLE IF EXISTS assignee_raw")
    con.execute("DROP TABLE IF EXISTS inventor_raw")
    con.execute("DROP TABLE IF EXISTS cpc_chunk")

    con.execute(
        f"""
        CREATE TABLE universe AS
        SELECT
            patent_id,
            TRY_CAST(app_year AS INTEGER) AS app_year,
            TRY_CAST(grant_year AS INTEGER) AS grant_year
        FROM read_csv_auto('{universe_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)
        """
    )
    con.execute(
        f"CREATE TABLE patent_core AS SELECT * FROM read_csv_auto('{patent_core_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    )
    con.execute(
        f"CREATE TABLE patent_abs AS SELECT * FROM read_csv_auto('{patent_abs_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    )
    con.execute(
        f"CREATE TABLE assignee_raw AS SELECT * FROM read_csv_auto('{assignee_raw_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    )
    con.execute(
        f"CREATE TABLE inventor_raw AS SELECT * FROM read_csv_auto('{inventor_raw_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    )
    con.execute(
        f"CREATE TABLE cpc_chunk AS SELECT * FROM read_csv_auto('{cpc_agg_chunk_path.as_posix()}', HEADER=TRUE, ALL_VARCHAR=TRUE)"
    )

    if universe_rows is None:
        universe_rows = int(con.execute("SELECT COUNT(*) FROM universe").fetchone()[0])
    if n_core is None:
        n_core = int(con.execute("SELECT COUNT(*) FROM patent_core").fetchone()[0])
    if n_abs is None:
        n_abs = int(con.execute("SELECT COUNT(*) FROM patent_abs").fetchone()[0])
    if n_assignee is None:
        n_assignee = int(con.execute("SELECT COUNT(*) FROM assignee_raw").fetchone()[0])
    if n_inventor is None:
        n_inventor = int(con.execute("SELECT COUNT(*) FROM inventor_raw").fetchone()[0])
    if n_cpc_rows is None:
        n_cpc_rows = int(con.execute("SELECT COUNT(*) FROM cpc_chunk").fetchone()[0])

    # 6) Country aggregation tables
    con.execute(
        """
        CREATE OR REPLACE TABLE assignee_country_primary AS
        WITH ranked AS (
            SELECT
                patent_id,
                country,
                assignee_sequence,
                ROW_NUMBER() OVER (PARTITION BY patent_id ORDER BY assignee_sequence ASC NULLS LAST) AS rn
            FROM assignee_raw
            WHERE country IS NOT NULL AND country <> ''
        )
        SELECT patent_id, country AS assignee_country_primary
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE assignee_country_mode AS
        WITH counts AS (
            SELECT patent_id, country, COUNT(*) AS n
            FROM assignee_raw
            WHERE country IS NOT NULL AND country <> ''
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                patent_id,
                country,
                n,
                ROW_NUMBER() OVER (PARTITION BY patent_id ORDER BY n DESC, country ASC) AS rn
            FROM counts
        )
        SELECT patent_id, country AS assignee_country_mode
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE inventor_country_primary AS
        WITH ranked AS (
            SELECT
                patent_id,
                country,
                inventor_sequence,
                ROW_NUMBER() OVER (PARTITION BY patent_id ORDER BY inventor_sequence ASC NULLS LAST) AS rn
            FROM inventor_raw
            WHERE country IS NOT NULL AND country <> ''
        )
        SELECT patent_id, country AS inventor_country_primary
        FROM ranked
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE inventor_country_mode AS
        WITH counts AS (
            SELECT patent_id, country, COUNT(*) AS n
            FROM inventor_raw
            WHERE country IS NOT NULL AND country <> ''
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                patent_id,
                country,
                n,
                ROW_NUMBER() OVER (PARTITION BY patent_id ORDER BY n DESC, country ASC) AS rn
            FROM counts
        )
        SELECT patent_id, country AS inventor_country_mode
        FROM ranked
        WHERE rn = 1
        """
    )

    # 7) CPC aggregation across chunk tables
    con.execute(
        """
        CREATE OR REPLACE TABLE cpc_agg AS
        SELECT
            patent_id,
            COUNT(*) AS cpc_total_count,
            COUNT(DISTINCT cpc_subclass) AS cpc_unique_subclass_count,
            MAX(TRY_CAST(has_measurement_cpc AS INTEGER)) AS has_measurement_cpc,
            MAX(TRY_CAST(has_testing_cpc AS INTEGER)) AS has_testing_cpc,
            MAX(TRY_CAST(has_ml_cpc AS INTEGER)) AS has_ml_cpc,
            MAX(TRY_CAST(has_data_processing_cpc AS INTEGER)) AS has_data_processing_cpc,
            MAX(TRY_CAST(has_bio_assay_cpc AS INTEGER)) AS has_bio_assay_cpc,
            MAX(TRY_CAST(has_statistics_cpc AS INTEGER)) AS has_statistics_cpc,
            SUM(CASE WHEN cpc_section = 'A' THEN 1 ELSE 0 END) AS cpc_section_A_count,
            SUM(CASE WHEN cpc_section = 'B' THEN 1 ELSE 0 END) AS cpc_section_B_count,
            SUM(CASE WHEN cpc_section = 'C' THEN 1 ELSE 0 END) AS cpc_section_C_count,
            SUM(CASE WHEN cpc_section = 'D' THEN 1 ELSE 0 END) AS cpc_section_D_count,
            SUM(CASE WHEN cpc_section = 'E' THEN 1 ELSE 0 END) AS cpc_section_E_count,
            SUM(CASE WHEN cpc_section = 'F' THEN 1 ELSE 0 END) AS cpc_section_F_count,
            SUM(CASE WHEN cpc_section = 'G' THEN 1 ELSE 0 END) AS cpc_section_G_count,
            SUM(CASE WHEN cpc_section = 'H' THEN 1 ELSE 0 END) AS cpc_section_H_count
        FROM cpc_chunk
        GROUP BY patent_id
        """
    )

    # 8) Final backbone
    con.execute(
        """
        CREATE OR REPLACE TABLE patent_backbone AS
        SELECT
            u.patent_id,
            CAST(u.app_year AS INTEGER) AS app_year,
            CAST(u.grant_year AS INTEGER) AS grant_year,
            CAST(pc.patent_date AS DATE) AS patent_date,
            CAST(EXTRACT(YEAR FROM CAST(pc.patent_date AS DATE)) AS INTEGER) AS patent_date_year,
            pc.patent_type,
            TRY_CAST(pc.num_claims AS INTEGER) AS num_claims,
            pc.patent_title,
            pa.patent_abstract,
            am.assignee_country_mode,
            ap.assignee_country_primary,
            im.inventor_country_mode,
            ip.inventor_country_primary,
            cpc.cpc_total_count,
            cpc.cpc_unique_subclass_count,
            cpc.has_measurement_cpc,
            cpc.has_testing_cpc,
            cpc.has_ml_cpc,
            cpc.has_data_processing_cpc,
            cpc.has_bio_assay_cpc,
            cpc.has_statistics_cpc,
            cpc.cpc_section_A_count,
            cpc.cpc_section_B_count,
            cpc.cpc_section_C_count,
            cpc.cpc_section_D_count,
            cpc.cpc_section_E_count,
            cpc.cpc_section_F_count,
            cpc.cpc_section_G_count,
            cpc.cpc_section_H_count
        FROM universe u
        LEFT JOIN patent_core pc USING (patent_id)
        LEFT JOIN patent_abs pa USING (patent_id)
        LEFT JOIN assignee_country_mode am USING (patent_id)
        LEFT JOIN assignee_country_primary ap USING (patent_id)
        LEFT JOIN inventor_country_mode im USING (patent_id)
        LEFT JOIN inventor_country_primary ip USING (patent_id)
        LEFT JOIN cpc_agg cpc USING (patent_id)
        """
    )

    backbone_parquet = OUTPUT_DIR / "patent_backbone.parquet"
    backbone_csv_gz = OUTPUT_DIR / "patent_backbone.csv.gz"
    con.execute(f"COPY patent_backbone TO '{backbone_parquet.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(
        f"COPY patent_backbone TO '{backbone_csv_gz.as_posix()}' (HEADER, DELIMITER ',', COMPRESSION GZIP)"
    )

    coverage = con.execute(
        """
        SELECT
            COUNT(*) AS n_patents,
            SUM(CASE WHEN patent_title IS NOT NULL THEN 1 ELSE 0 END) AS n_with_title,
            SUM(CASE WHEN patent_abstract IS NOT NULL THEN 1 ELSE 0 END) AS n_with_abstract,
            SUM(CASE WHEN assignee_country_mode IS NOT NULL THEN 1 ELSE 0 END) AS n_with_assignee_country,
            SUM(CASE WHEN inventor_country_mode IS NOT NULL THEN 1 ELSE 0 END) AS n_with_inventor_country,
            SUM(CASE WHEN cpc_total_count IS NOT NULL THEN 1 ELSE 0 END) AS n_with_cpc
        FROM patent_backbone
        """
    ).fetchdf().iloc[0].to_dict()

    coverage_path = OUTPUT_DIR / "patent_backbone_coverage.json"
    coverage_path.write_text(json.dumps({k: int(v) if pd.notna(v) else None for k, v in coverage.items()}, indent=2))

    stats = {
        "patent_universe_rows": int(universe_rows),
        "patent_core_filtered_rows": int(n_core),
        "patent_abstract_filtered_rows": int(n_abs),
        "assignee_country_raw_rows": int(n_assignee),
        "inventor_country_raw_rows": int(n_inventor),
        "cpc_chunk_rows": int(n_cpc_rows),
        "backbone_rows": int(coverage["n_patents"]),
    }
    (OUTPUT_DIR / "patent_backbone_build_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patent-level backbone with text + metadata + country assignment.")
    parser.add_argument("--chunksize", type=int, default=500_000, help="Chunk size for TSV processing.")
    parser.add_argument(
        "--no-reuse-intermediate",
        action="store_true",
        help="Force re-extraction even if intermediate filtered files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_backbone(chunksize=args.chunksize, reuse_intermediate=not args.no_reuse_intermediate)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
