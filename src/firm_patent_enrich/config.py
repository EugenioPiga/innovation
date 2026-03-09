from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    start_year: int = 1997
    end_year: int = 2022
    max_firms: int | None = None
    sec_user_agent: str = "research@example.com"
    include_10k_text_features: bool = True
    include_fred_macro: bool = True
    include_bls_oes: bool = True
    include_census_context: bool = True
    include_bea_context: bool = False
    fred_api_key: str | None = None
    bls_api_key: str | None = None
    bea_api_key: str | None = None
    link_fuzzy_threshold: float = 0.86
    link_score_gap: float = 0.03
    link_top_n: int = 5

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
