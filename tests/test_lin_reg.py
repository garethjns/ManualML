import unittest

import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from manual_ml.supervised.lin_reg import LinReg


class Models:
    def _fit_predict_plot(self, mod):
        mod.fit(self.x_train, self.y_train)
        mod.predict(self.x_test)

        mod.plot_history(log=True)

        y_pred_train = mod.predict(self.x_train)
        y_pred_test = mod.predict(self.x_test)
        plt.scatter(self.y_train, y_pred_train)
        plt.scatter(self.y_test, y_pred_test)
        plt.xlabel('y')
        plt.xlabel('y_pred')
        plt.show()

        print(mod.results['coefs'], mod.results['b'])

    def test_no_reg(self):

        mod = LinReg(learning_rate=0.01,
                     max_its=10000,
                     lr_decay=0.0001,
                     norm=True,
                     a=0.00000000000000011,
                     reg=None,
                     conv_steps=10)

        self._fit_predict_plot(mod)


    def test_l1_reg(self):

        mod = LinReg(learning_rate=0.01,
                     max_its=10000,
                     lr_decay=0.0001,
                     norm=True,
                     a=1,
                     reg='l2',
                     conv_steps=10)

        self._fit_predict_plot(mod)

    def assert_coefs(self, coefs, intercept):
        if self.expected_coefs is not None:
            [self.assertAlmostEqual(c, c_, 2) for c, c_ in zip(self.expected_coefs, coefs)]
        if self.expected_bias is not None:
            self.assertAlmostEqual(self.expected_intercept, intercept, 2)


class TestBoston(Models, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = load_boston(return_X_y=True)
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(cls.x, cls.y,
                                                                            test_size=0.25)

        cls.expected_coefs = None
        cls.expected_intercept = None


class TestMini(Models, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [3, 3], [3, 2]])
        cls.y = 2 * cls.x[:, 0] + 3 * cls.x[:, 1] + 1
        cls.x_train, cls.x_test, cls.y_train, cls.y_test = train_test_split(cls.x, cls.y,
                                                                            test_size=0.25)

        cls.expected_coefs = [2, 3]
        cls.expected_intercept = 1
