import unittest

import numpy as np

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)


class LinearModelsTest(unittest.TestCase):
    def test_regression_returns_parameters_in_input_units(self):
        train = np.array([[3, 9], [4, 9], [7, 9], [10, 9]])
        target = 1.5 * train[:, 0] - 2
        test = np.array([[-12, 9], [100, 9]])
        data = RegressionInput(
            train, target[:, None], test, None, learning_rate=0.1, epochs=150
        )
        result = train_linear_regression(data)

        np.testing.assert_allclose(result.weights, [1.5, 0.0])
        self.assertAlmostEqual(result.bias, -2.0)
        np.testing.assert_allclose(result.predictions, test @ result.weights + result.bias)
        self.assertEqual(len(result.loss_history), data.epochs)
        # После первого шага при таком линейном y прогноз равен 2 * lr * y.
        self.assertAlmostEqual(result.loss_history[0], np.mean((0.8 * target) ** 2))
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        mse = np.mean((train @ result.weights + result.bias - target) ** 2)
        self.assertAlmostEqual(result.loss_history[-1], mse)

    def test_logistic_batch_update_and_extreme_test_logits(self):
        # На train среднее = 3, std = 2. За один шаг w_std = 0.2, b_std = 0.
        data = ClassificationInput(
            train_features=[[1], [1], [5], [5]], train_target=[0, 0, 1, 1],
            test_features=[[-1e8], [3], [1e8]], test_target=None,
            learning_rate=0.4, epochs=1,
        )
        with np.errstate(over="raise", invalid="raise"):
            result = train_logistic_classifier(data)

        np.testing.assert_allclose(result.weights, [0.1])
        self.assertAlmostEqual(result.bias, -0.3)
        np.testing.assert_allclose(result.loss_history, [np.logaddexp(0, -0.2)])
        np.testing.assert_allclose(result.probabilities, [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(result.predictions, [0, 1, 1])

    def test_comparison_uses_probabilities_for_auc(self):
        data = ClassificationInput(
            train_features=[[-3], [-2], [-1], [1], [2], [3]], train_target=[0, 0, 0, 1, 1, 1],
            test_features=[[0.1], [0.2], [0.3], [0.4]], test_target=[0, 0, 1, 1], epochs=10,
        )
        comparison = compare_classical_models(data)
        # Все прогнозы — класс 1, но логистическая модель строго ранжирует объекты.
        self.assertEqual(comparison.logistic_regression.roc_auc, 1.0)
        self.assertEqual(comparison.decision_tree.roc_auc, 0.5)
        self.assertEqual(comparison.knn.roc_auc, 0.5)
        for scores in (comparison.logistic_regression, comparison.decision_tree, comparison.knn):
            self.assertEqual(scores.accuracy, 0.5)
            self.assertEqual(scores.precision, 0.5)
            self.assertEqual(scores.recall, 1.0)
            self.assertAlmostEqual(scores.f1, 2 / 3)


if __name__ == "__main__":
    unittest.main()
