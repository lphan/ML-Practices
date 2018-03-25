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

from Starts.startmod import *
from Starts.startvis import *

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans


class StartModSKL(StartMod):
    """
        Description: StartModSKL - Start Models Scikit-Learn
        regression, classification

        References:
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
        Apply method Linear regression y = ax + b (one independent variable, one dependent_label)

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param poly: initiate polynomial linear regression
        :return: LinearRegression-object (or PolynomialFeatures-object if poly=True), true_test_result, predicted_result
        """

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

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

            # estimate the model by cross_validation method and training_data
            # StartMod.validation(reg_poly, x_train, y_train)

            return reg_poly, y_test, y_predict
        else:
            # fit model with training-data
            reg_lin = LinearRegression()
            reg_lin.fit(x_train, y_train)

            # estimate the model by cross_validation method and training_data
            # StartMod.validation(reg_lin, x_train, y_train)

            # Visualizing
            StartVis.vis_obj_predict(x_train, y_train, reg_lin)

            # Predicting the Test and return the predicted result
            y_predict = reg_lin.predict(x_test)

            # Visualizing
            StartVis.vis_obj_predict(x_test, y_test, reg_lin)

            return reg_lin, y_test, y_predict

    @classmethod
    def regression_multi_linear(cls, data, dependent_label, pr=True):
        """
        Apply method Multiple Linear regression y = b0 + b1.x1 + b2.x2 + ... + bn.xn
        choose the optimal feature_columns using algorithm Backward Elimination

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param pr: default True to display info
        :return: RegressionResultsWrapper object, true_test_result, predicted_result
        """

        # drop dependent value from data and save the independent values into x
        # data = data.drop(dependent_label, axis=1)
        # x = data.values

        # split data into feature (independent) values and dependent values in type Numpy.array
        x, y = StartMod.split_data(data, dependent_label, split=False, type_pd=False)

        # add new column constant b0 with value 0
        x = np.append(arr=np.ones(shape=(x.shape[0], 1)).astype(int), values=x, axis=1)

        # apply backward_eliminate to find pvalue and result of (adj)_r_squared
        reg_ols, max_pvalue, x_opt = StartMod.backward_eliminate(data, x, y)

        if pr:
            print("x_value is optimal with p_value: ", max_pvalue)
            print("\nR_Squared: ", reg_ols.rsquared)
            print("\nAdjusted_R_Squared: ", reg_ols.rsquared_adj)
            print("\nSummary: ", reg_ols.summary())

        x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=0)

        # Execute Linear Regression on optimal value, fit model with training-data
        reg = LinearRegression()
        reg.fit(x_train, y_train)

        # Estimate the model by cross_validation method and training_data (not appropriate method for regression_models)
        # StartMod.validation(reg, x_train, y_train)

        # Visualizing
        StartVis.vis_obj_predict(x_train, y_train, reg)

        # Predicting the Test and return the predicted result
        y_predict = reg.predict(x_test)

        # Visualizing
        StartVis.vis_obj_predict(x_test, y_test, reg)

        return reg, y_test, y_predict

    @classmethod
    def regression_decision_tree(cls, data, dependent_label, random_state=0):
        """
        Apply method Decision Tree regression

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param random_state: default is 0
        :return: DecisionTreeRegressor object, true_test_result, predicted_result
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_true, y_train, y_true = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_dt = DecisionTreeRegressor(random_state=random_state)
        reg_dt.fit(x_train, y_train)

        # Estimate the model by cross_validation method and training_data
        # StartMod.validation(reg_dt, x_train, y_train)

        # Visualizing
        StartVis.vis_obj_predict(x_train, y_train, reg_dt)

        # Predicting a new result
        y_predict = reg_dt.predict(x_true)

        # Visualizing
        StartVis.vis_obj_predict(x_true, y_true, reg_dt)

        return reg_dt, y_true, y_predict

    @classmethod
    def regression_random_forest(cls, data, dependent_label, n_est=10, ens=False):
        """
        Apply method Random forest regression

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        :param data:
        :param dependent_label:
        :param n_estimators: the number of trees in the forest.
        :param ens: ensemble learning by decision tree and other regression model
        :return:
        """
        x_train, x_true, y_train, y_true = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_rf = RandomForestRegressor(n_estimators=n_est, random_state=0)
        reg_rf.fit(x_train, y_train)

        # if ens (ensemble learning), re_implement random forest by applying other regression_model as one decision tree
        # then get the mean result from every decision tree

        # Predicting a new result
        y_predict = reg_rf.predict(x_true)

        return reg_rf, y_true, y_predict

    @classmethod
    def regression_logistic(cls, data, dependent_label):
        """
        Apply method Logistic regression

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :return: LogisticRegression object, true_test_result, predicted_result
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_log = LogisticRegression(random_state=0)
        reg_log.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = reg_log.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(reg_log, x_train, y_train)

        return reg_log, y_test, y_predict

    @classmethod
    def classification_knn(cls, data, dependent_label, k_nb=5):
        """
        Apply k-Nearest Neighbours method to classify data

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param  k_nb: number of neighbors
        :return: KNeighborsClassifier object, true_test_result, predicted_result
        """

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        clf_knn = KNeighborsClassifier(n_neighbors=k_nb, metric='euclidean', p=2)
        clf_knn.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = clf_knn.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_knn, x_train, y_train)

        return clf_knn, y_test, y_predict

    @classmethod
    def classification_svm(cls, data, dependent_label, kernel='rbf'):
        """
        Apply Support Vector Machine method to classify data

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            http://mlkernels.readthedocs.io/en/latest/kernels.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param kernel: default is 'rbf', kernel_options = ['linear', 'poly', 'sigmoid']
        :return: SVC Object, true_test_result, predicted_result
        """
        def kernel_compute(kn):
            kc_clf_svc = SVC(kernel=kn, random_state=0)
            kc_clf_svc.fit(x_train, y_train)

            # Predicting the Test set results
            kc_y_predict = kc_clf_svc.predict(x_true)
            # cm = confusion_matrix(y_test, kc_y_predict)
            # kc_correct = cm[0][0] + cm[1][1]
            # print(kc_clf_svc, kc_correct)
            return kc_clf_svc, kc_y_predict

        x_train, x_true, y_train, y_true = StartMod.split_data(data, dependent_label, type_pd=False)

        # Find and choose the best Kernel-SVM to fit with the Training set
        # kernel_options = ['linear', 'poly', 'sigmoid']
        classifier_svc, y_predict = kernel_compute(kernel)

        # for kernel in kernel_options:
        #     max_correct, clf_svc, y_predict = kernel_compute(kernel)
        #     if max_correct > default_max_correct:
        #         default_clf_svc = clf_svc
        #         default_y_predict = y_predict

        # estimate the model by cross_validation method and training_data
        StartMod.validation(classifier_svc, x_train, y_train)

        return classifier_svc, y_true, y_predict

    @classmethod
    def classification_nb(cls, data, dependent_label):
        """
        Apply Gaussian Naive Bayes method to classify data

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :return: GaussianNB, true_test_result, predicted_result
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        clf_gnb = GaussianNB()
        clf_gnb.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_gnb.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_gnb, x_train, y_train)

        return clf_gnb, y_test, y_predict

    @classmethod
    def classification_random_forest(cls, data, dependent_label):
        """
        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        :param data:
        :param dependent_label:
        :return:
        """
        pass

    @classmethod
    def clustering_k_mean_noc(cls, data, plot=False):
        """
        Find the number of clusters using the elbow method.

        References:
            http://www.awesomestats.in/python-cluster-validation/

        :param data: pandas.core.frame.DataFrame
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
        # tbd: compute average variance in cluster_errors to choose the best number of clusters and return it

        return clusters_df

    @classmethod
    def clustering_k_mean(cls, data, noc):
        """
        Requirement: run clustering_k_mean_noc first to find 'noc' (number of possible clusters)
        Apply method clustering k_means++ to cluster data with the given 'noc'

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        :param data: pandas.core.frame.DataFrame
        :param noc: number of clusters
        :return: KMeans object, predicted y_values
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
