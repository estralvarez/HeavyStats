import pandas as pd
import heavystats as hs


def test_validate_data_on_sample():
    df = hs.load_data()
    report = hs.validate_data(df)
    assert isinstance(report, hs.ValidationReport)
    assert isinstance(report.checks, list)
    assert len(report.checks) > 0


def test_validation_report_outputs():
    df = hs.load_data()
    report = hs.validate_data(df)

    df_rep = report.to_dataframe()
    assert isinstance(df_rep, pd.DataFrame)
    assert "Estado" in df_rep.columns
    assert "Criterio" in df_rep.columns

    text_rep = report.to_text()
    assert isinstance(text_rep, str)
    assert len(text_rep) > 0

    html_rep = report.to_html()
    assert isinstance(html_rep, str)
    assert "<table" in html_rep
