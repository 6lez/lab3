"""Воспроизведение экспериментов из REPORT.md на CSV рядом со скриптом."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    evaluate_baseline,
    fit_and_compare_classifiers,
    train_linear_regression,
)

ROOT = Path(__file__).resolve().parent


def _split(frame, target_column, *, stratify=False):
    target = frame[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        frame.drop(columns=target_column), target,
        test_size=0.2, random_state=42, stratify=target if stratify else None,
    )
    return {
        "train_features": x_train,
        "train_target": y_train,
        "test_features": x_test,
        "test_target": y_test,
    }


def _print_history(name, history):
    for epoch in (1, 10, 100, len(history)):
        print(f"{name}, эпоха {epoch}: {history[epoch - 1]:.8f}")


def main():
    students = pd.read_csv(ROOT / "Student_Performance.csv")
    activity = "Extracurricular Activities"
    students[activity] = students[activity].map({"No": 0, "Yes": 1})
    regression = RegressionInput(**_split(students, "Performance Index"))
    linear = train_linear_regression(regression)
    test_mse = mean_squared_error(regression.test_target, linear.predictions)
    _print_history("Train MSE", linear.loss_history)
    mean_prediction = regression.train_target.mean()
    constant_mse = ((regression.test_target - mean_prediction) ** 2).mean()
    print(f"Test MSE для среднего по train: {constant_mse:.8f}")

    credit = pd.read_csv(ROOT / "german.csv", sep=";")
    classification = ClassificationInput(**_split(credit, "Creditability", stratify=True))
    own, comparison = fit_and_compare_classifiers(classification)
    _print_history("Train BCE", own.loss_history)
    print("Матрица ошибок (строка — факт, столбец — прогноз; классы 0, 1):")
    print(confusion_matrix(classification.test_target, own.predictions, labels=[0, 1]))

    # Независимая проверка GD: та же нерегуляризованная BCE, другой оптимизатор.
    reference = evaluate_baseline(
        classification,
        LogisticRegression(penalty=None, solver="lbfgs", tol=1e-9, max_iter=1000),
    )
    results = {
        "linear_regression": {"test_mse": float(test_mse)},
        "classification": {
            "own_logistic": asdict(comparison.logistic_regression),
            "logistic_regression": asdict(reference),
            "decision_tree": asdict(comparison.decision_tree),
            "knn": asdict(comparison.knn),
        },
    }
    output = json.dumps(results, indent=2, allow_nan=False)
    (ROOT / "experiment_results.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
