from firm_patent_enrich.utils import normalize_name, to_cik_str


def test_normalize_name_and_cik():
    assert normalize_name("Acme, Inc.") == "acme"
    assert to_cik_str("320193") == "0000320193"