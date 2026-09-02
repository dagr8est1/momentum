import pandas as pd

from momentum import cli


def test_main_run_command_wires_config_backtest_and_reporting(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "benchmark: SPY\nstart_date: '2022-01-01'\nend_date: '2022-06-01'\n"
        "universe:\n  source: static\n  tickers: [MSFT]\n"
    )
    output_path = tmp_path / "out.html"

    fake_returns = pd.Series([0.01, -0.02], index=pd.date_range("2022-01-03", periods=2))
    fake_benchmark = pd.Series([0.005, -0.01], index=pd.date_range("2022-01-03", periods=2))
    calls = {}

    def _fake_run_backtest(config):
        calls["config"] = config
        return fake_returns, fake_benchmark

    def _fake_generate_tearsheet(returns, benchmark, output_path):
        calls["returns"] = returns
        calls["benchmark"] = benchmark
        calls["output_path"] = output_path

    monkeypatch.setattr(cli, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(cli, "generate_tearsheet", _fake_generate_tearsheet)

    exit_code = cli.main(["run", str(config_path), "--output", str(output_path)])

    assert exit_code == 0
    assert calls["config"].benchmark == "SPY"
    assert calls["output_path"] == str(output_path)
    assert str(output_path) in capsys.readouterr().out
