# Social Returns to Data: Dataset Bundle

This folder contains a reproducible data pull for studying whether data/privacy regulation changes private and social returns to R&D by affecting data access costs.

## Run

From the repository root:

```bash
python scripts/download_social_returns_data.py
```

Optional:

```bash
python scripts/download_social_returns_data.py --force
python scripts/download_social_returns_data.py --skip-biblio
```

## Build merged country-year panel

```bash
python scripts/build_social_returns_panel.py
```

This writes:

- `processed/panel/social_returns_country_year_panel.csv`
- `processed/panel/social_returns_country_year_panel_metadata.csv`
- `processed/panel/social_returns_country_year_panel_diagnostics.json`
- `processed/panel/ecipe_country_iso3_mapping.csv`
- `processed/panel/oecd_msti_measure_coverage.csv`

## Output structure

- `raw/`: direct source downloads
- `processed/`: normalized tables and summary files
- `dataset_catalog.csv`: machine-readable file inventory with URLs, timestamps, and file sizes
- `dataset_catalog.json`: same catalog as JSON

## Included source families

- UNCTAD cyberlaw tracker data file (country-level legal status by domain)
- ECIPE Digital Trade Environment measures export
- Data.gouv global map for personal data protection
- OECD STI SDMX datasets (MSTI, BERD, GERD, R&D personnel, tax support, and bibliometrics)
- World Bank WDI indicator panel (R&D, patents, publications, digitalization, and macro controls)
- Penn World Table files (productivity and related components)

## Note

Some sources are large (especially OECD bibliometrics). Use `--skip-biblio` if you want a faster pull first.
