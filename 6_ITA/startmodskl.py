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
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix


class StartModSKL(StartMod):
    """
        Description: StartModSKL - Start Models Scikit-Learn
        regression, classification

        Source:
        http://scikit-learn.org/stable/modules/classes.html

        Start:
          jupyter notebook
          -> from startmodskl import *
          -> info_modskl
    """

    def __init__(self):
        pass

    @classmethod
    def regression_linear(cls, data, dependent_label, poly=False):
        """
        Ordinary least squares Linear Regression y = ax + b (one independent variable, one dependent_label)
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        :param data: Pandas-DataFrame
        :param dependent_label:
        :param poly: initiate polynomial linear regression
        :return: LinearRegression-object (or PolynomialFeatures-object if poly=True), true_test_result, predicted_result
        """

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

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

            return reg_poly, y_test, y_predict

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

            return reg_lin, y_test, y_predict

    @classmethod
    def regression_multi_linear(cls, data, dependent_label):
        """
        Multiple Linear Regression y = b0 + b1.x1 + b2.x2 + ... + bn.xn (Method: Backward Elimination)
        :param data: Pandas-DataFrame
        :param dependent_label:
        :return: RegressionResultsWrapper object, true_test_result, predicted_result
        """

        # save the dependent value into y
        # y = data[dependent_label].values

        # drop dependent value from data and save the independent values into x
        # data = data.drop(dependent_label, axis=1)
        # x = data.values
        x, y = StartMod.split_data(data, dependent_label, split=False)

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

        return reg_ols, y_test, y_predict

    @classmethod
    def regression_decision_tree(cls, data, dependent_label):
        """
        Decision Tree regression method
        :param data: Pandas-DataFrame
        :param dependent_label:
        :return: DecisionTreeRegressor, true_test_result, predicted_result
        """

        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values
        #
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

        reg_dt = DecisionTreeRegressor(random_state=0)
        reg_dt.fit(x_train, y_train)

        # Visualizing
        StartVis.vis_obj_predict(x_train, y_train, reg_dt)

        # Predicting a new result
        y_predict = reg_dt.predict(x_test)

        # Visualizing
        StartVis.vis_obj_predict(x_test, y_test, reg_dt)

        return reg_dt, y_test, y_predict

    @classmethod
    def regression_logistic(cls, data, dependent_label):
        """
        apply method logistic regression to data
        :param data: Pandas-DataFrame
        :return: LogisticRegression object, true_test_result, predicted_result
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

        reg_log = LogisticRegression(random_state=0)
        reg_log.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = reg_log.predict(x_test)

        return reg_log, y_test, y_predict

    @classmethod
    def classification_knn(cls, data, dependent_label, k=5):
        """
        Apply k-Nearest Neighbours method to classify data
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        :param data: Pandas-DataFrame
        :param dependent_label:
        :return: KNeighborsClassifier object, true_test_result, predicted_result
        """

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

        clf_knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', p=2)
        clf_knn.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = clf_knn.predict(x_test)

        return clf_knn, y_test, y_predict

    @classmethod
    def classification_svm(cls, data, dependent_label):
        """
        Apply Support Vector Machine method to classify data
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            http://mlkernels.readthedocs.io/en/latest/kernels.html
        :param data: Pandas-DataFrame
        :param dependent_label:
        :param k: kernel (default is 'rbf')
        :return: SVC Object, true_test_result, predicted_result
        """
        def kernel_compute(kn):
            kc_clf_svc = SVC(kernel=kn, random_state=0)
            kc_clf_svc.fit(x_train, y_train)

            # Predicting the Test set results
            kc_y_predict = kc_clf_svc.predict(x_test)
            cm = confusion_matrix(y_test, kc_y_predict)
            kc_correct = cm[0][0] + cm[1][1]
            print(kc_clf_svc, kc_correct)
            return kc_correct, kc_clf_svc, kc_y_predict

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

        # Find and choose the best Kernel-SVM to fit with the Training set
        # (Kernel options: rbf (default), linear, poly, sigmoid
        kernel_options = ['linear', 'poly', 'sigmoid']
        default_max_correct, default_clf_svc, default_y_predict = kernel_compute('rbf')

        for kernel in kernel_options:
            max_correct, clf_svc, y_predict = kernel_compute(kernel)
            if max_correct > default_max_correct:
                default_clf_svc = clf_svc
                default_y_predict = y_predict

        return default_clf_svc, y_test, default_y_predict

    @classmethod
    def classification_nb(cls, data, dependent_label):
        """
        Apply Gaussian Naive Bayes method to classify data
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        :param data: Pandas-DataFrame
        :return: GaussianNB, true_test_result, predicted_result
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label)

        clf_gnb = GaussianNB()
        clf_gnb.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = clf_gnb.predict(x_test)

        return clf_gnb, y_test, y_predict

    @classmethod
    def clustering_k_mean_noc(cls, data, plot=False):
        """
        Source:
            http://www.awesomestats.in/python-cluster-validation/
        find the number of clusters using the elbow method.
        :param data:
        :return: plot of data to identify number of clusters (still manually)
        """
        cluster_errors = []

        # given the range of number of clusters from 1 to 10
        cluster_range = range(1, 11)

        for i in cluster_range:
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(data.values)
            cluster_errors.append(kmeans.inertia_)

        # plot to see result
        if plot:
            plt.figure(figsize=(16, 14))
            plt.plot(range(1, 11), cluster_errors)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('')
            plt.show()

        clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})
        # tbd: compute average variance in cluster_errors to choose the best number of clusters
        # and return it

        return clusters_df

    @classmethod
    def clustering_k_mean(cls, data, noc):
        """
        Requirement: run clustering_k_mean_noc first to find 'noc' (number of possible clusters)
        Apply method clustering k_means++ to cluster data with the given 'noc'
        :param data:
        :param noc:
        :return:
        """
        k_means = KMeans(n_clusters=noc, init='k-means++', random_state=0)
        y_clusters = k_means.fit_predict(data.values)

        return k_means, y_clusters

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModSKL": StartMod.__name__,
            "StartModSKL.regression_linear": StartModSKL.regression_linear.__doc__,
            }
        # info.update(StartML.info_help())

        return info


info_modskl = StartMod.info_help()
