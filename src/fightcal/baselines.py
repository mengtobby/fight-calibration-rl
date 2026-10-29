"""Non-language-model forecasting baselines fitted on training bouts only."""

from collections.abc import Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fightcal.data.odds import market_prob_a
from fightcal.data.schema import Bout
from fightcal.features import feature_matrix
from fightcal.predictions import PredictionRecord


def _record(bout: Bout, arm: str, p: float | None) -> PredictionRecord:
    return PredictionRecord(bout.bout_id, arm, "base", -1, p, bout.a_won, market_prob_a(bout))


def market_baseline(bouts: Iterable[Bout]) -> list[PredictionRecord]:
    return [_record(b, "market", market_prob_a(b)) for b in bouts]


def constant_baseline(
    bouts: Iterable[Bout], p: float = 0.5, arm: str = "constant_0.5"
) -> list[PredictionRecord]:
    if isinstance(p, bool) or not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a finite probability in [0, 1]")
    return [_record(b, arm, p) for b in bouts]


def base_rate_baseline(train: Iterable[Bout], test: Iterable[Bout]) -> list[PredictionRecord]:
    training = list(train)
    if not training:
        raise ValueError("train must contain at least one bout")
    return constant_baseline(test, sum(b.a_won for b in training) / len(training), "base_rate")


def logistic_baseline(
    train: Iterable[Bout], test: Iterable[Bout], C: float = 1.0
) -> list[PredictionRecord]:
    if isinstance(C, bool) or not np.isfinite(C) or C <= 0:
        raise ValueError("C must be finite and positive")
    training, testing = list(train), list(test)
    if len({b.a_won for b in training}) != 2:
        raise ValueError("train must contain both outcome classes")
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=C, random_state=0, max_iter=1000),
    )
    model.fit(feature_matrix(training), [b.a_won for b in training])
    if not testing:
        return []
    return [
        _record(b, "logreg", float(p))
        for b, p in zip(testing, model.predict_proba(feature_matrix(testing))[:, 1], strict=True)
    ]
