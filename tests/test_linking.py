import pandas as pd

from firm_patent_enrich.linking import LinkConfig, auto_link_gvkey_to_cik


def test_auto_link_exact_and_conflict_flags():
    firms = pd.DataFrame(
        {
            "gvkey": ["001000"],
            "firm_name": ["Acme Corp"],
            "patent_count": [10],
        }
    )
    sec = pd.DataFrame(
        {
            "cik": ["0000000001", "0000000002"],
            "ticker": ["ACM", "ACME"],
            "sec_name": ["Acme Corp", "Acme Corporation"],
            "sec_name_norm": ["acme", "acme"],
            "sec_name_compact": ["acme", "acme"],
            "exchange": ["NYSE", "NASDAQ"],
        }
    )

    links, candidates, conflicts = auto_link_gvkey_to_cik(
        primary_names=firms,
        sec_tickers=sec,
        ticker_hints=None,
        config=LinkConfig(fuzzy_threshold=0.5, score_gap_for_conflict=0.5, top_n_candidates=5),
    )

    assert len(links) == 1
    assert links.iloc[0]["link_method"] in {"exact_normalized_name", "exact_compact_name"}
    assert not candidates.empty
    assert not conflicts.empty
