"""
Some tests for the tree classifier. TODO: Not finished, just structure.
"""

import unittest

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split

from manual_ml.helpers.metrics import accuracy
from manual_ml.supervised.tree import Tree


class Models:
    def test_tree_1(self):

        self.mod = Tree(max_depth=2,
                        min_data=2,
                        dynamic_bias=False)

        self.mod.fit(self.x_train, self.y_train)
        self.mod._print()

        y_train_pred = self.mod.predict(self.x_train)
        y_test_pred = self.mod.predict(self.x_test)

        train_acc = accuracy(self.y_train, y_train_pred)
        test_acc =accuracy(self.y_test, y_test_pred)
        print(f"Train accuracy: {train_acc}")
        print(f"Test accuracy: {test_acc}")

        self.assertTrue(train_acc > 0.5)
        self.assertTrue(test_acc > 0.5)


class TestSmallData(Models, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.DataFrame({'x1': [2.77, 1.73, 3.68, 3.96, 2.99, 7.50, 9.00, 7.44,
                                    10.12, 6.64],
                             'x2': [1.78, 1.17, 2.81, 2.62, 2.21, 3.16, 3.34, 0.48,
                                    3.23, 3.32],
                             'y': [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]})
        cls.y = data.y
        cls.x = data[['x1', 'x2']]
        cls.x_train = cls.x
        cls.x_test = cls.x
        cls.y_train = cls.y
        cls.y_test = cls.y
        cls.data = data

    def print_and_plot(self, y_pred):
        """Not actually used...."""
        self.mod._print()

        plt.scatter(self.data.x1[self.data.y == 0], self.data.x2[self.data.y == 0])
        plt.scatter(self.data.x1[self.data.y == 1], self.data.x2[self.data.y == 1])
        plt.show()
        plt.scatter(self.data.x1[y_pred == 0], self.data.x2[y_pred == 0])
        plt.scatter(self.data.x1[y_pred == 1], self.data.x2[y_pred == 1])
        plt.show()


class TestBreastCancer(Models, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.y = sklearn.datasets.load_breast_cancer(return_X_y=True)

        x_train, x_test, y_train, y_test = train_test_split(cls.x, cls.y,
                                                            test_size=0.25,
                                                            random_state=512)

        cls.x_train = pd.DataFrame(x_train)
        cls.x_test = pd.DataFrame(x_test)
        cls.y_train = pd.Series(y_train)
        cls.y_tes = pd.Series(y_test)

        cls.mod = Tree(min_data=10,
                       max_depth=6,
                       dynamic_bias=False,
                       bias=0.5)
