from firm_patent_enrich.external_data import map_sic_to_naics_bucket


def test_map_sic_to_naics_bucket_basic():
    assert map_sic_to_naics_bucket("2834") == "31-33"
    assert map_sic_to_naics_bucket("6021") == "52"
    assert map_sic_to_naics_bucket("5311") == "44-45"
    assert map_sic_to_naics_bucket(None) is None
