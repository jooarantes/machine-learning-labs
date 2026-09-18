import json
import numpy as np
from sklearn.metrics import make_scorer


def load_payoff_matrix(path: str) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["payoff_matrix"])


def credit_gain_score(
    y_true,
    y_score,
    threshold=0.5,
    payoff_matrix=None
):
    y_true = np.asarray(y_true).astype(int)

    if y_score.ndim == 2:
        y_score = y_score[:, 1]

    y_pred = (y_score >= threshold).astype(int)
    gains = payoff_matrix[y_true, y_pred]

    return gains.mean()


def make_credit_gain_scorer(
    payoff_matrix,
    threshold=0.5
):
    def scorer_fn(y_true, y_score):
        return credit_gain_score(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
            payoff_matrix=payoff_matrix
        )

    return make_scorer(
        scorer_fn,
        greater_is_better=True,
        response_method="predict_proba"
    )
