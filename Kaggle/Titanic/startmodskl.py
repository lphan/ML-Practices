#!/usr/bin/env python3
#
# Copyright (c) 2018
#
# This software is licensed to you under the GNU General Public License,
# version 2 (GPLv2). There is NO WARRANTY for this software, express or
# implied, including the implied warranties of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. You should have received a copy of GPLv2
# along with this software; if not, see
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.

__author__ = 'Long Phan'


# from sklearn.pipeline import make_pipeline
from startmod import *
from startvis import *
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix


class StartModSKL(StartMod):
    """
      Description: StartModSKL - Start Models from scikit-learn
        linear_regression,

      Start:
          jupyter notebook
          -> from startmod import *
          -> info_mod
    """

    def __init__(self):
        pass

    @classmethod
    def linear_regression(cls, data, dependent_label):
        """
        Ordinary least squares Linear Regression
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        :param data:
        :return: the predicted result based on test
        """
        # convert data into numpy-values
        # x = data.iloc[:, :-1].values
        # y = data.iloc[:, 1].values

        y = data[dependent_label].values
        x = data.drop(dependent_label, axis=1).values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # return x_train, x_test, y_train, y_test
        # fit model with training-data
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Predicting the Test and return the predicted result
        return reg.predict(x_test), y_test

    @staticmethod
    def info_help():
        info = {
            "info_help_StartMod": StartMod.__name__,
            "StartMod.feature_engineering(data)": StartMod.feature_engineering.__doc__,
            "StartMod.feature_engineering_merge_cols(data)": StartMod.feature_engineering_merge_cols.__doc__,
            }
        # info.update(StartML.info_help())

        return info


info_modskl = StartMod.info_help()
