"""Tests for model serialization, loading, and prediction."""

import tempfile
from pathlib import Path

from demand_ml.data import make_synthetic_demand
from demand_ml.features import FEATURE_COLUMNS, add_calendar_features
from demand_ml.model import train_model
from demand_ml.persistence import load_model, predict, save_model


def test_save_and_load_model() -> None:
    df = make_synthetic_demand(n_rows=100, seed=2)
    model, _ = train_model(df, test_size=0.2, random_state=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "model.joblib"
        save_model(model, fpath)
        assert fpath.exists()

        loaded = load_model(fpath)
        assert hasattr(loaded, "predict")


def test_predict_with_model_object() -> None:
    df = make_synthetic_demand(n_rows=100, seed=3)
    model, _ = train_model(df, test_size=0.2, random_state=3)
    featured = add_calendar_features(df.iloc[:10])

    preds = predict(model, featured)
    assert len(preds) == 10
    assert (preds >= 0).all()


def test_predict_with_filepath() -> None:
    df = make_synthetic_demand(n_rows=100, seed=4)
    model, _ = train_model(df, test_size=0.2, random_state=4)
    featured = add_calendar_features(df.iloc[:10])

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "model.joblib"
        save_model(model, fpath)

        preds = predict(fpath, featured)
        assert len(preds) == 10
        assert (preds >= 0).all()


def test_predict_with_feature_subset() -> None:
    df = make_synthetic_demand(n_rows=100, seed=5)
    model, _ = train_model(df, test_size=0.2, random_state=5)
    featured = add_calendar_features(df.iloc[:10])

    preds_subset = predict(model, featured, use_feature_subset=True)
    assert len(preds_subset) == 10

    preds_as_is = predict(model, featured[FEATURE_COLUMNS], use_feature_subset=False)
    assert len(preds_as_is) == 10


def test_predict_preserves_index() -> None:
    df = make_synthetic_demand(n_rows=50, seed=6)
    model, _ = train_model(df, test_size=0.2, random_state=6)
    featured = add_calendar_features(df.iloc[5:15])

    preds = predict(model, featured)
    assert preds.index.equals(featured.index)


def test_save_model_creates_directories() -> None:
    df = make_synthetic_demand(n_rows=50, seed=7)
    model, _ = train_model(df, test_size=0.2, random_state=7)

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "nested" / "dirs" / "model.joblib"
        save_model(model, fpath)
        assert fpath.exists()
