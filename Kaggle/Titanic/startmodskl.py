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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

import statsmodels.formula.api as sm
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
    def split_data(cls, data, dependent_label, split=True):
        # convert data into numpy-values (in case: the last column is dependent label)
        # x = data.iloc[:, :-1].values
        # y = data.iloc[:, 1].values
        # save the dependent value into y
        y = data[dependent_label].values

        # drop dependent value from data and save the independent values into x
        x = data.drop(dependent_label, axis=1).values

        if not split:
            return x, y

        # split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Feature scaling
        sc_x = StandardScaler(copy=True, with_mean=True, with_std=True)
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)

        return x_train, x_test, y_train, y_test

    @classmethod
    def regression_linear(cls, data, dependent_label, poly=False):
        """
        Ordinary least squares Linear Regression y = ax + b (one independent variable, one dependent_label)
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        :param data: clean encoded data without NaN-value
        :param dependent_label:
        :param poly: initiate polynomial linear regression
        :return: LinearRegression-object (or PolynomialFeatures-object if poly=True), predicted_result, test_result
        """

        x_train, x_test, y_train, y_test = StartModSKL.split_data(data, dependent_label)

        # Feature scaling
        sc_x = StandardScaler(copy=True, with_mean=True, with_std=True)
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)

        if poly:
            # fit polynomial regression degree level = 3, with training-data
            reg_poly = PolynomialFeatures(degree=3)
            x_poly = reg_poly.fit_transform(x_train)
            reg_poly.fit(x_poly, y_train)

            # apply linear regression to polynomial data
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(x_poly, y_train)

            # predict value on testing data by applying polynomial regression object
            y_predict = lin_reg_2.predict(reg_poly.fit_transform(x_test))

            return reg_poly, y_predict, y_test

        else:

            # fit model with training-data
            reg_lin = LinearRegression()
            reg_lin.fit(x_train, y_train)

            # Visualizing
            StartVis.vis_obj_predict(x_train, y_train, reg_lin)

            # Predicting the Test and return the predicted result
            y_predict = reg_lin.predict(x_test)

            # Visualizing
            StartVis.vis_obj_predict(x_test, y_test, reg_lin)

            return reg_lin, y_predict, y_test

    @classmethod
    def regression_multi_linear(cls, data, dependent_label):
        """
        Multiple Linear Regression y = b0 + b1.x1 + b2.x2 + ... + bn.xn (Method: Backward Elimination)
        :param data:
        :param dependent_label:
        :return: RegressionResultsWrapper object, predicted_result, test_result
        """

        # save the dependent value into y
        # y = data[dependent_label].values

        # drop dependent value from data and save the independent values into x
        # data = data.drop(dependent_label, axis=1)
        # x = data.values
        x, y = StartModSKL.split_data(data, dependent_label, split=False)

        # add column 0 as constant b0 implicitly
        x = np.append(arr=np.ones(shape=(x.shape[0], 1)).astype(int), values=x, axis=1)

        # Start Method Backward Elimination
        # Step 1: select a significance level to stay
        sl = 0.05
        x_opt = x[:, [i for i in range(len(data.columns))]]
        # print(range(len(data.columns)), x_opt.shape, range(len(data.columns)))

        # Step 2: fit the full model with all possible predictors
        # create new object for Ordinary Least Square OLS
        reg_ols = sm.OLS(endog=y, exog=x_opt).fit()

        # Step 3: select and remove the redundant columns with pvalue > significant level 0.05
        max_pvalue, col_idx = StartML.find_idx_max_value(reg_ols.pvalues)

        while max_pvalue > sl:
            x_opt = np.delete(x_opt, col_idx, axis=1)

            # recompute regressor with new value without the redundant column
            reg_ols = sm.OLS(endog=y, exog=x_opt).fit()
            max_pvalue, idx = StartML.find_idx_max_value(reg_ols.pvalues)
            # print(max_pvalue, reg_ols.pvalues, idx)

        print("x_value is optimal with pvalue ", max_pvalue)
        x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=0)

        # Execute Linear Regression on optimal value
        # fit model with training-data
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Visualizing
        StartVis.vis_obj_predict(x_train, y_train, reg)

        # Predicting the Test and return the predicted result
        y_predict = reg.predict(x_test)

        # Visualizing
        StartVis.vis_obj_predict(x_test, y_test, reg)

        return reg_ols, y_predict, y_test

    @classmethod
    def regression_decision_tree(cls, data, dependent_label):

        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values
        #
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        x_train, x_test, y_train, y_test = StartModSKL.split_data(data, dependent_label)

        reg_dt = DecisionTreeRegressor(random_state=0)
        reg_dt.fit(x_train, y_train)

        # Visualizing
        StartVis.vis_obj_predict(x_train, y_train, reg_dt)

        # Predicting a new result
        y_predict = reg_dt.predict(x_test)

        # Visualizing
        StartVis.vis_obj_predict(x_test, y_test, reg_dt)

        return reg_dt, y_predict, y_test

    @classmethod
    def regression_logistic(cls, data, dependent_label):
        """
        apply method logistic regression to data
        :param data: DataFrame Pandas
        :return:
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        x_train, x_test, y_train, y_test = StartModSKL.split_data(data, dependent_label)

        reg_log = LogisticRegression(random_state=0)
        reg_log.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = reg_log.predict(x_test)

        return reg_log, y_predict, y_test

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
