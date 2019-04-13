import unittest

import sklearn.metrics as skm
import manual_ml.helpers.metrics as mlm

from typing import Callable


class RunTests:
    def run_a(self, baseline_f: Callable, test_f: Callable) -> None:
        """

        :param baseline_f: Presumed correct function.
        :param test_f: Function to test.
        """
        for s in self.sets:
            if len(s[0]) == len(s[1]):
                # First try running baseline. If it errors, assert same error raised by test function.
                try:
                    bs_res = baseline_f(s[0], s[1])

                    # Baseline runs? Run test function and compare.
                    self.assertEqual(bs_res, test_f(s[0], s[1]))
                except Exception as e:
                    self.assertRaises(type(e), lambda: test_f(s[0], s[1]))


class TestBinary(RunTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sets = [([0, 0, 0, 0], [0, 0, 0, 0]),
                    ([0, 0, 0, 1], [0, 0, 0, 1]),
                    ([0, 1, 0, 1], [0, 0, 0, 1]),
                    ([1, 1, 1, 1], [1, 1, 1, 1]),
                    ([1, 0, 0, 0, 0], [1, 0, 0, 0, 0]),
                    ([1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),
                    ([1, 0, 1, 0, 1], [1, 0, 0, 0, 1]),
                    ([1, 1, 1, 1, 1], [1, 1, 1, 1])]

    def test_accuracy(self) -> None:
        self.run_a(baseline_f=skm.accuracy_score,
                   test_f=mlm.accuracy)

    def test_log_loss(self) -> None:
        self.run_a(baseline_f=skm.log_loss,
                   test_f=mlm.log_loss)


class TestProba(RunTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sets = [([0.5, 0.5, 0.45, 0.35], [0.65, 0.5, 0.45, 0.65]),
                    ([0.1, 0.1, 0.1, 1], [0.1, 0.1, 0.1, 1]),
                    ([0.25, 0.25, 0, 1], [0.25, 0, 0, 1]),
                    ([1, 1, 1, 1], [1, 1, 1, 1]),
                    ([0.5, 0.4, 0.3, 0.2, 0.1], [0.6, 0.5, 0.6, 0.3, 0.2]),
                    ([1, 1, 1, 1, 1], [1, 1, 1, 1])]

    def test_mse(self) -> None:
        self.run_a(baseline_f=skm.mean_squared_error,
                   test_f=mlm.mse)


if __name__ == "__main__":
    unittest.main()
