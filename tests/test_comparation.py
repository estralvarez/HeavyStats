import pandas as pd
import heavystats as hs
from heavystats.comparation import compare_groups, ComparationReport


def test_compare_groups_on_sample():
    df = hs.load_data()
    if "Muestra_Codificada" in df.columns:
        report = compare_groups(df, sample_col="Muestra_Codificada")
        assert isinstance(report, ComparationReport)
        dfs = report.to_dataframes()
        assert isinstance(dfs, dict)
        assert len(dfs) > 0
        html = report.to_html()
        assert isinstance(html, str)
        assert "<div" in html
