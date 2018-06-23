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
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import plot_importance

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
    def regression_linear(cls, data, dependent_label, poly=False, vis=True):
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
            # setup default degree
            deg = 3

            # fit polynomial regression degree level = 3, with training-data
            reg_poly = PolynomialFeatures(degree=deg)
            x_poly = reg_poly.fit_transform(x_train)
            reg_poly.fit(x_poly, y_train)

            # apply linear regression to polynomial data
            lin_reg_2 = LinearRegression()
            lin_reg_2.fit(x_poly, y_train)

            print("Calculating coefficients: ", lin_reg_2.coef_)
            # print("Evaluation using r-square: ", lin_reg_2.score(x_test, y_test))

            # predict value on testing data by applying polynomial regression object
            y_predict = lin_reg_2.predict(reg_poly.fit_transform(x_test))

            # estimate the model by cross_validation method and training_data
            # StartMod.validation(reg_poly, x_train, y_train)

            if vis:
                # Visual the result
                StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict,
                                         title='Training vs predicted data')

            return reg_poly, y_test, y_predict
        else:
            # fit model with training-data
            lin_reg = LinearRegression()
            lin_reg.fit(x_train, y_train)

            print("Calculating coefficients: ", lin_reg.coef_)
            print("Evaluation using r-square: ", lin_reg.score(x_test, y_test))

            # estimate the model by cross_validation method and training_data
            # StartMod.validation(reg_lin, x_train, y_train)

            # Predicting the Test and return the predicted result
            y_predict = lin_reg.predict(x_test)

            if vis:
                # Visualizing
                StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict,
                                         title='Training vs predicted data')

            return lin_reg, y_test, y_predict

    @classmethod
    def regression_multi_linear(cls, data, dependent_label, pr=True, vis=True):
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

        print("\nCalculating coefficients: ", reg.coef_)
        print("\nEvaluation using r-square: ", reg.score(x_test, y_test))

        # Estimate the model by cross_validation method and training_data (not appropriate method for regression_models)
        # StartMod.validation(reg, x_train, y_train)

        # Predicting the Test and return the predicted result
        y_predict = reg.predict(x_test)

        # # checking the magnitude of features
        # features = data.columns.drop(dependent_label)
        # print(len(features), len(reg.coef_), data.columns, features)

        if vis:
            # plot all coefficients to see which features have the most impact on model
            # reg_coefficient = pd.DataFrame(data=reg.coef_, index=features)
            # plt.figure(1)
            # reg_coefficient.plot(kind='bar')
            # plt.title("Feature_Important")
            # plt.xticks(rotation=45)

            # Visual the result
            StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict, title='Training vs predicted data')

        return reg, y_test, y_predict

    @classmethod
    def regression_decision_tree(cls, data, dependent_label, random_state=0, vis=False):
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

        # checking the magnitude of features
        features = data.columns.drop(dependent_label)
        features_important = reg_dt.feature_importances_
        print("Features: ", features, "\nFeatures_Important: ", features_important)

        # Predict
        y_predict = reg_dt.predict(x_true)

        if vis:
            # plot all coefficients to see which features have the most impact on model
            data_features = pd.DataFrame(data=features_important, index=features)
            plt.figure(1)
            data_features.plot(kind='bar')
            plt.title("Feature_Important")
            plt.xticks(rotation=45)

            # Visual the result
            StartVis.vis_obj_predict(list(range(len(x_true))), y_true, y_predict, title='Training vs predicted data')

        return reg_dt, y_true, y_predict

    @classmethod
    def regression_random_forest(cls, data, dependent_label, n_est=10, ens=False):
        """
        Apply method Random forest regression

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param n_est: the number of trees in the forest.
        :param ens: ensemble learning by decision tree and other regression model
        :return:
        """
        x_train, x_true, y_train, y_true = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_rf = RandomForestRegressor(n_estimators=n_est, random_state=0)
        reg_rf.fit(x_train, y_train)

        # If ens (ensemble learning), re_implement random forest by applying other regression_model as one decision tree
        # then get the mean result from every decision tree

        # Predicting a new result
        y_predict = reg_rf.predict(x_true)

        return reg_rf, y_true, y_predict

    @classmethod
    def regression_svr(cls, data, dependent_label):
        """

        :param data:
        :return:
        """
        pass

    @classmethod
    def regression_logistic(cls, data, dependent_label, random_state=None, solver='liblinear'):
        """
        Apply method Regularized logistic regression:
        Solver:
            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘
                liblinear’ is limited to one-versus-rest schemes.

            ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param random_state: default None
        :param solver: default 'liblinear', others: {‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’}
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

        # estimate regularization
        StartMod.regularization(data, dependent_label)

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
    def classification_bagged_dt(cls, data, dependent_label, num_trees, seed = 7):
        """

        References:
            https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param num_trees:
        :param seed:
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        cart = DecisionTreeClassifier()
        clf_bdt = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
        clf_bdt.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_bdt.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_bdt, x_train, y_train)

        return clf_bdt, y_test, y_predict

    @classmethod
    def classification_rf(cls, data, dependent_label, num_trees, max_features):
        """
        Apply Ensemble Random Forest for classification

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            http://scikit-learn.org/stable/modules/ensemble.html#random-forests
            https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/


        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param num_trees: number of decision trees
        :param max_features: number of maximal features to select randomly
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        # In order to reduce the size of the model, you can change these parameters:
        # min_samples_split, min_samples_leaf, max_leaf_nodes and max_depth
        # setup n_jobs=-1 to setup all cores available on the machine are used
        clf_rf = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, n_jobs=-1)
        clf_rf.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_rf.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_rf, x_train, y_train)

        return clf_rf, y_test, y_predict

    @classmethod
    def classification_adab(cls, data, dependent_label, num_trees=30, seed=7):
        """
        Apply Ensemble AdaBoost for classification

        References:
            http://scikit-learn.org/stable/modules/ensemble.html#adaboost

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param num_trees:
        :param seed:
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        clf_adab = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
        clf_adab.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_adab.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_adab, x_train, y_train)

        return clf_adab, y_test, y_predict

    @classmethod
    def classification_sgb(cls, data, dependent_label, num_trees=30, seed=7, regression=False):
        """
        Apply Ensemble Stochastic Gradient Boosting for classification

        References:
            http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting
            http://scikit-learn.org/stable/modules/ensemble.html#loss-functions

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param num_trees:
        :param seed:
        :param regression:
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        if regression:
            clf_sgb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,
                                            loss='ls')
        else:
            clf_sgb = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=1.0, max_depth=1, random_state=seed,
                                             loss='deviance')
        clf_sgb.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_sgb.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_sgb, x_train, y_train)

        return clf_sgb, y_test, y_predict

    @classmethod
    def classification_xgb(cls, data, dependent_label):
        """
        Apply Extreme Gradient Boosting (XGBoost) method to classify data

        References:
            http://xgboost.readthedocs.io/en/latest/model.html
            https://machinelearningmastery.com/xgboost-python-mini-course/
            http://scikit-learn.org/stable/modules/ensemble.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :return:
        """
        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)
        # print(type(x_train), type(y_train))

        clf_xgb = XGBClassifier()
        clf_xgb.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = clf_xgb.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_xgb, x_train, y_train)

        plot_importance(clf_xgb)

        return clf_xgb, y_test, y_predict
        # return x_train, x_test, y_train, y_test

    @classmethod
    def classification_voting(cls, models, data, dependent_label):
        """
        voting ensemble model for classification by combining the predictions from multiple machine learning algorithms.
        parameter:
            voting='hard': mode label
            voting='soft': weighted average value

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
            http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

        :param models:
        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :return:
        """
        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        ensemble = VotingClassifier(models)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(ensemble, x_train, y_train)

        return ensemble

    @classmethod
    def clustering_k_mean_noc(cls, data, plot=False):
        """
        Find the number of clusters using the elbow method.

        References:
            http://www.awesomestats.in/python-cluster-validation/

        :param data: pandas.core.frame.DataFrame
        :param plot: show plot (default is False)
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

    @classmethod
    def ensemble(cls, data):
        """
        References:
            http://scikit-learn.org/stable/modules/ensemble.html
            https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
        :param data:
        :return:
        """
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModSKL": StartMod.__name__,
            "StartModSKL.regression_linear": StartModSKL.regression_linear.__doc__,
            }
        # info.update(StartML.info_help())

        return info


info_modskl = StartMod.info_help()
