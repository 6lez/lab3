"""Линейные модели с пакетным градиентным спуском."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _prepare(data):
    scaler = StandardScaler()
    train = scaler.fit_transform(np.asarray(data.train_features, dtype=float))
    test = scaler.transform(np.asarray(data.test_features, dtype=float))
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    if len(target) != len(train):
        raise ValueError("На каждую обучающую строку нужна одна целевая метка")
    return train, target, test, scaler


def _original_parameters(weights, bias, scaler):
    # ((x - mean) / scale) @ w + b = x @ w_raw + b_raw
    weights = weights / scaler.scale_
    return weights, float(bias - scaler.mean_ @ weights)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Минимизирует MSE на train; коэффициенты возвращаются в масштабе входа."""
    train, target, test, scaler = _prepare(data)
    weights = np.zeros(train.shape[1])
    bias = 0.0
    history = []
    residual = -target  # начальный прогноз равен нулю
    for _ in range(data.epochs):
        weights -= data.learning_rate * (2 / len(train)) * (train.T @ residual)
        bias -= data.learning_rate * 2 * residual.mean()
        residual = train @ weights + bias - target
        history.append(float(np.mean(residual**2)))

    predictions = test @ weights + bias
    weights, bias = _original_parameters(weights, bias, scaler)
    return RegressionResult(
        weights=weights, bias=bias, loss_history=history, predictions=predictions
    )


def _sigmoid(logits):
    # sigmoid(z) = exp(-log(1 + exp(-z))); не переполняется при больших |z|.
    return np.exp(-np.logaddexp(0.0, -logits))


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Минимизирует BCE на train и предсказывает вероятность класса 1 на test."""
    train, target, test, scaler = _prepare(data)
    if not np.isin(target, (0, 1)).all():
        raise ValueError("Для бинарной кросс-энтропии нужны метки 0 или 1")

    weights = np.zeros(train.shape[1])
    bias = 0.0
    logits = np.zeros(len(train))
    history = []
    signs = 1 - 2 * target
    for _ in range(data.epochs):
        residual = _sigmoid(logits) - target
        weights -= data.learning_rate * (train.T @ residual) / len(train)
        bias -= data.learning_rate * residual.mean()
        logits = train @ weights + bias
        # BCE = softplus(z) для y=0 и softplus(-z) для y=1.
        history.append(float(np.logaddexp(0.0, signs * logits).mean()))

    probabilities = _sigmoid(test @ weights + bias)
    weights, bias = _original_parameters(weights, bias, scaler)
    return ClassificationResult(
        weights=weights, bias=bias, loss_history=history,
        probabilities=probabilities, predictions=(probabilities >= 0.5).astype(int),
    )


def _metrics(target, probabilities, predictions) -> ClassificationMetrics:
    target = np.asarray(target).reshape(-1)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def evaluate_baseline(data: ClassificationInput, estimator) -> ClassificationMetrics:
    """Обучает эталонный классификатор на том же train, не меняя разбиение."""
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(data.train_features, np.asarray(data.train_target).reshape(-1))
    probabilities = model.predict_proba(data.test_features)[:, 1]
    return _metrics(data.test_target, probabilities, model.predict(data.test_features))


def fit_and_compare_classifiers(
    data: ClassificationInput,
) -> tuple[ClassificationResult, BaselineComparisonResult]:
    """Сохраняет и историю обучения, и метрики без повторного обучения модели."""
    own = train_logistic_classifier(data)
    k = min(5, len(data.train_target))
    comparison = BaselineComparisonResult(
        logistic_regression=_metrics(data.test_target, own.probabilities, own.predictions),
        decision_tree=evaluate_baseline(data, DecisionTreeClassifier(random_state=42)),
        knn=evaluate_baseline(data, KNeighborsClassifier(n_neighbors=k)),
    )
    return own, comparison


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Возвращает метрики своей логистической регрессии, дерева и kNN на одном test."""
    _, comparison = fit_and_compare_classifiers(data)
    return comparison
