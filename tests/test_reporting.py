import numpy as np
import pandas as pd

from momentum.reporting import generate_tearsheet


def test_generate_tearsheet_writes_html_file(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, size=120), index=dates)
    benchmark = pd.Series(rng.normal(0.0003, 0.008, size=120), index=dates)

    output_path = tmp_path / "tearsheet.html"
    generate_tearsheet(returns, benchmark, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
