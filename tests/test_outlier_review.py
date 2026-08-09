import numpy as np
import pandas as pd

from app.datasets import load_dataset, put_dataframe
from app.nodes.outlier_review import _build_options, _recommend, outlier_review_node


def _skewed_series_with_outliers() -> pd.Series:
    rng = np.random.default_rng(0)
    values = list(rng.normal(10, 1, 100)) + [500, 520, 480]  # a few extreme outliers
    return pd.Series(values)


def test_build_options_uses_real_abex_treatments():
    values = _skewed_series_with_outliers()
    from abex.data.outliers import detect_outliers

    mask = detect_outliers(values, method="iqr")
    options = _build_options(values, mask)

    methods = {o["method"] for o in options}
    assert {"winsorize", "trim", "cap", "log_transform", "none"} <= methods

    trim_option = next(o for o in options if o["method"] == "trim")
    assert trim_option["n_affected"] == int(mask.sum())
    assert trim_option["n_affected"] > 0


def test_recommend_picks_trim_for_low_outlier_share():
    profile = {"kind": "continuous", "skewness": 0.2, "outlier_share": 0.01, "zero_share": 0.0}
    assert _recommend(profile) == "trim"


def test_recommend_picks_log_transform_for_high_skew():
    profile = {"kind": "continuous", "skewness": 3.0, "outlier_share": 0.05, "zero_share": 0.0}
    assert _recommend(profile) == "log_transform"


def test_outlier_review_node_pauses_and_resumes(monkeypatch):
    calls = {}

    def fake_ask_human(payload):
        calls["payload"] = payload
        return {"method": "winsorize", "params": {"lower_q": 0.01, "upper_q": 0.99}}

    monkeypatch.setattr("app.nodes.outlier_review.ask_human", fake_ask_human)

    values = _skewed_series_with_outliers()
    groups = ["control", "treatment"] * (len(values) // 2) + ["control"] * (len(values) % 2)
    df = pd.DataFrame({"metric": values, "group": groups})
    from abex.data.outliers import detect_outliers
    from abex.data.profiling import profile_metric
    from dataclasses import asdict

    mask = detect_outliers(df["metric"], method="iqr")
    profile = asdict(profile_metric(df, "metric", "group"))

    state = {
        "company_id": "_test",
        "dataset_id": put_dataframe(df, company_id="_test"),
        "metric_col": "metric",
        "metric_profile": profile,
        "outlier_mask": mask.tolist(),
    }

    result = outlier_review_node(state)

    assert calls["payload"]["kind"] == "outlier_review"
    assert result["outlier_decision"]["method"] == "winsorize"
    assert result["treated_metric_col"] == "metric__treated"
    assert "metric__treated" in load_dataset(result["treated_dataset_id"]).columns
