from typing import List, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manual_ml.base import BaseModel
from manual_ml.helpers.metrics import accuracy


class Tree(BaseModel):
    """
    Binary classification tree
    """
    def __init__(self,
                 min_data: int=2,
                 max_depth: int =3,
                 dynamic_bias: bool=True,
                 bias: float=0.5):
        self.params = {'max_depth': max_depth,
                       'min_data': min_data,
                       'dynamic_bias': dynamic_bias,
                       'bias': bias}
        self.feature_names = []

        self.tree: Dict[str, Any] = None

    def _print(self,
               nodes: List[str]=None,
               root: bool=True):

        if nodes is None:
            nodes = []

        if root:
            nodes = self.tree

        Tree.print_node(nodes)

        if nodes['class'] == -1:
            self._print(nodes=nodes['l_node'],
                        root=False)
            self._print(nodes=nodes['r_node'],
                        root=False)

    @staticmethod
    def print_node(node):
        bs =' ' * node['depth'] * 2
        bs2 = '-' * 2
        print(bs+'|'+bs2+'*' * 50)
        print(bs+'|'+bs+node['node_str'])
        print(bs+'|'+bs+'Depth:', node['depth'])
        print(bs+'|'+bs+'n samples:', node['n'])

        if not node['terminal']:
            print(bs+'|'+bs+'Name: '+node['name'])
            print(bs+'|'+bs+'Split Value:', node['split_val'])
            print(bs+'|'+bs+'Gini coeff:', node['gini'])
            print(bs+'|'+bs+'Class prop requirement:', node['bias'],
                  '('+node['biasMode']+')')
            print(bs+'|'+bs+'Prop L:', node['prop_l'])
            print(bs+'|'+bs+'Prop R:', node['prop_r'])

        else:
            print(bs+'|'+bs+'Leaf')
            print(bs+'|'+bs+node['note'])

    def fit(self, x, y, debug=False):
        """
        Convert data to matrix if dataframe
        Recursively create nodes using tree.buildNodes()
        """

        # Set feature names
        self.set_names(x)

        # Convert to mats if not
        x = self.strip_df(x)
        y = self.strip_df(y)

        self.tree = Tree.build_nodes(x, y,
                                     max_depth=self.params['max_depth'],
                                     min_data=self.params['min_data'],
                                     dynamic_bias=self.params['dynamic_bias'],
                                     debug=debug, names=self.feature_names)

        return self

    @staticmethod
    def build_nodes(x, y, names: List[str],
                    max_depth: int=2,
                    min_data: int=2,
                    dynamic_bias: bool=False,
                    depth: int=0,
                    nom_class: int=-777,
                    bias: float=0.5,
                    d_str: str= 'Root',
                    debug: bool=False):
        """
        Recursively all branches of nodes. Each branch continues adding nodes until a terminal condition is met.

        :param x: Features.
        :param y: Labels.
        :param names: Feature column names.
        :param max_depth: Max branch depth. Default=2.
        :param min_data: Min number of observations left to build another node. Default=2.
        :param dynamic_bias:
        :param depth: Current depth.
        :param nom_class:
        :param bias:
        :param d_str: String name for node.
        :param debug:
        :return:
        """

        if dynamic_bias:
            bias = Tree.prop(y)
            ds = 'dynamic'
        else:
            if bias == '':
                ds = 'highest'
            else:
                ds = 'static'

        # Add terminal checks here
        # If a terminal node, return a node (dict) containing just the class label
        # This label is set by highest represented label in subset
        if depth > max_depth:
            # Too deep: Terminal
            cla = Tree.high_class(y, bias)
            node = {'class': cla,
                    'depth': depth,
                    'note': 'Max depth reached, class is: '+ str(cla),
                    'terminal': True,
                    'n': len(x),
                    'node_str': d_str}

        elif x.shape[0] < min_data:
            if x.shape[0] == 0:
                # Too few data points: Terminal
                cla = nom_class
            else:
                cla = Tree.high_class(y, bias)

            node = {'class': cla,
                    'depth': depth,
                    'note': f'Too few data points, class is: {cla}',
                    'terminal': True,
                    'n': len(x),
                    'node_str': d_str}
            # In this case, y will be empty
            # So use nominal class that will be the opposite of the other side node

        elif x.shape[1] < 1:
            # Too few features: Terminal
            cla = Tree.high_class(y, bias)
            node = {'class': cla,
                    'depth': depth,
                    'note': f'No features remaining, class is: {cla}',
                    'terminal': True,
                    'n': len(x),
                    'node_str': d_str}

        elif len(np.unique(y)) == 1:
            # Only one class: Terminal
            cla = Tree.high_class(y, bias)
            node = {'class': cla,
                    'depth': depth,
                    'note': f'One class at depth, class is: {cla}',
                    'terminal': True,
                    'n': len(x),
                    'node_str': d_str}
        else:
            # Not terminal. Build next node.

            # First find best split to run
            col_idx, best_x, gini = Tree.get_best_split_all(x, y)

            # Split into left and right subsets
            l_idx = (x[:, col_idx] < best_x).squeeze()
            r_idx = (x[:, col_idx] >= best_x).squeeze()

            nom_class_l = -999
            nom_class_r = -999
            if np.sum(l_idx) == 0:
                nom_class_l = np.int8(not Tree.high_class(y[r_idx], bias))
            if np.sum(r_idx) == 0:
                nom_class_r = np.int8(not Tree.high_class(y[l_idx], bias))

            # Build next node, leaving out used feature and data not in this split
            l_node = Tree.build_nodes(x[l_idx][:, ~col_idx], y[l_idx],
                                      max_depth=max_depth,
                                      min_data=min_data,
                                      depth=depth + 1,
                                      nom_class=nom_class_l,
                                      dynamic_bias=dynamic_bias,
                                      bias=bias,
                                      d_str=d_str + '->L',
                                      names=[n for ni, n in enumerate(names) if ni != np.argmax(col_idx)])

            r_node = Tree.build_nodes(x[r_idx][:, ~col_idx], y[r_idx],
                                      max_depth=max_depth,
                                      min_data=min_data,
                                      depth=depth + 1,
                                      nom_class=nom_class_r,
                                      dynamic_bias=dynamic_bias,
                                      bias=bias,
                                      d_str=d_str + '->R',
                                      names=[n for ni, n in enumerate(names) if ni != np.argmax(col_idx)])

            # Return a full node containing meta/debug data
            # As this isn't a leaf/terminal node, set class to -1
            node = {'name': names[np.argmax(col_idx)],
                    'n': len(x),
                    'n_l': np.sum(l_idx),
                    'n_r': np.sum(r_idx),
                    'l_idx': l_idx,
                    'r_idx': r_idx,
                    'split_val': best_x.squeeze(),
                    'gini': gini.squeeze(),
                    'depth': depth,
                    'l_node': l_node,
                    'r_node': r_node,
                    'class': -1,
                    'prop_l': Tree.prop(y[l_idx]),
                    'prop_r': Tree.prop(y[r_idx]),
                    'biasMode': ds,
                    'bias': bias,
                    'node_str' : d_str,
                    'terminal': False}

        if debug:
            Tree.print_node(node)

        return node

    def predict(self, x) -> np.ndarray:
        """Predict from tree."""

        y_pred = x.apply(Tree._predict,
                         args=(self.tree,),
                         axis=1)

        return y_pred

    @staticmethod
    def _predict(x, mod) -> np.ndarray:

        # If this is a leaf node, return class
        if mod['class'] > -1:
            return mod['class']

        # If this isn't a leaf node, check X against split value
        # and follow tree
        if x.loc[mod['name']] < mod['split_val']:
            # If less than split val, go left
            y_pred = Tree._predict(x, mod['l_node'])
        else:
            # If greater than split val, go right
            y_pred = Tree._predict(x, mod['r_node'])

        return y_pred

    @staticmethod
    def gi(groups: Union[List[int], np.array],
           classes: Union[List[int], np.array]) -> float:
        """Calculate Gini."""
        groups = np.array(groups)
        classes = np.array(classes)

        # For each group
        sum_p = 0.0
        for g in np.unique(groups):
            # print('G:',g)
            g_idx = groups == g
            # Calculate and sum class proportions
            p = 0.0
            # For each class
            for c in np.unique(classes):
                # print('C:',c)
                c_idx = classes[g_idx] == c
                # Get proportions and square
                # And sum across classes
                p += (np.sum(c_idx) / np.sum(g_idx)) ** 2
                # print('P:',P)

            # Weight by size of group
            # And sum across groups
            sum_p += (1 - p) * sum(g_idx) / len(g_idx)

        return sum_p

    @staticmethod
    def split(x, y, split_val) -> float:
        groups = np.int8(x < split_val)

        return Tree.gi(groups, y)

    @staticmethod
    def get_best_split_all(x, y) -> Tuple[int, float, float]:
        """
        This function calculates all splits on all columns
        Returns the column index with best split and the values to use
        """
        m = x.shape[1]
        col_best_gin = np.ones(shape=m)
        col_best_val = np.ones(shape=m)
        for c in range(m):
            best = 1
            best_x = 0
            for i in np.unique(x[:, c]):
                gini = Tree.split(x[:, c], y, i)
                if gini < best:
                    best = gini
                    best_x = i
                col_best_gin[c] = best
                col_best_val[c] = best_x

        # Select best feature to split on
        col_idx = np.argmin(col_best_gin)
        # Convert to bool index
        col_idx = np.array(range(x.shape[1])) == col_idx

        return col_idx, col_best_val[col_idx], col_best_gin[col_idx]

    @staticmethod
    def prop(y: np.array) -> Union[int, float]:
        if np.sum(y) > 0:
            return y.sum() / y.shape[0]
        else:
            return 0

    @staticmethod
    def high_class(y,
                   bias: str='') -> int:

        if bias == '':
            # Just return highest class
            return np.argmax(y.value_counts())
        else:
            # Return logical of class prop>bias
            if len(y) > 0:
                return np.int8(Tree.prop(y) > bias)
            else:
                return 0


if __name__ == '__main__':
    data = pd.DataFrame({'x1': [2.77, 1.73, 3.68, 3.96, 2.99, 7.50, 9.00, 7.44,
                                10.12, 6.64],
                         'x2': [1.78, 1.17, 2.81, 2.62, 2.21, 3.16, 3.34, 0.48,
                                3.23, 3.32],
                         'y': [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]})
    y = data.y
    x = data[['x1', 'x2']]

    mod = Tree(max_depth=2,
               min_data=2,
               dynamic_bias=False)

    mod.fit(x, y)

    mod._print()

    y_pred = mod.predict(x)
    accuracy(y, y_pred)

    plt.scatter(data.x1[data.y == 0], data.x2[data.y == 0])
    plt.scatter(data.x1[data.y == 1], data.x2[data.y == 1])
    plt.show()
    plt.scatter(data.x1[y_pred == 0], data.x2[y_pred == 0])
    plt.scatter(data.x1[y_pred == 1], data.x2[y_pred == 1])
    plt.show()
