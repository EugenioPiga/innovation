import pandas as pd

from firm_patent_enrich.patents import build_patent_firm_year_panel


def test_build_patent_firm_year_panel_basic():
    df = pd.DataFrame(
        {
            "patent_id": [1, 2, 2, 3],
            "appYear": [2000, 2000, 2000, 2001],
            "gvkeyUO": ["001000", "001000", "001000", "001000"],
            "gvkeyFR": ["001000", "001000", "001000", "001000"],
            "clean_name": ["ACME", "ACME", "ACME", "ACME"],
            "privateSubsidiary": [0, 1, 1, 0],
            "grantYear": [2001, 2002, 2002, 2003],
        }
    )

    out = build_patent_firm_year_panel(df)
    row_2000 = out[out["year"] == 2000].iloc[0]

    assert int(row_2000["patents_applied"]) == 2
    assert int(row_2000["private_subsidiary_patents"]) == 2
    assert float(row_2000["private_subsidiary_share"]) == 1.0