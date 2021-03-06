#!/usr/bin/env python3
#
# Copyright (c) 2019
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

from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from scipy.stats import linregress
from xgboost import XGBClassifier
from xgboost import plot_importance


class StartModSKL(StartMod):
    """
        Description: StartModSKL - Start Models Scikit-Learn (Regression, Classification)

        References:
            http://scikit-learn.org/stable/modules/classes.html
            https://www.python.org/dev/peps/pep-3135/

        Start:
            jupyter notebook
            -> from startmodskl import *
            -> info_modskl
    """

    __random_state = 100

    def __init__(self):
        super().__init__()  # StartMod.__init__(self)  or super(StartModSKL, self).__init__()

    @classmethod
    def regress_linear_simple(cls, data, dependent_label, x_pred):
                
        # convert to type Numpy
        y = data[dependent_label].values
        # drop dependent value from data and save the independent values into x
        x = data.drop(dependent_label, axis=1).values

        # fit linear regression model and identify the coefficients b1, b0 of linear function
        a, b, r_value, p_value, std_err = linregress(x, y)

        # make predictions
        yhat = b + a*x_pred

        # define new input, expected value and prediction
        # x_in = x[0]
        # y_out = y[0]
        # yhat_out = yhat[0]

        # estimate stdev of yhat
        sum_errs = arraysum((y - yhat)**2)
        stdev = sqrt(1/(len(y)-2) * sum_errs)

        # calculate prediction interval
        interval = 1.96 * stdev
        print('Prediction Interval: %.3f' % interval)
        lower, upper = yhat - interval, yhat + interval
        print('95%% likelihood that the true value is between %.3f and %.3f' % (lower, upper))
        print('Predicted value: %.3f' % yhat)

        # plot dataset and prediction with interval
        pyplot.scatter(x, y)
        pyplot.plot(x_pred, yhat, color='red')
        pyplot.errorbar(x_pred, yhat, yerr=interval, color='black', fmt='o')
        pyplot.show()

    @classmethod
    def regress_linear(cls, data, dependent_label, poly=False, ridge=False, vis=True, save=True, regularization=True):
        """
        Description: apply method Linear regression e.g y = ax + b (one independent variable, one dependent_label)

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param poly: initiate polynomial linear regression
        :param ridge: Linear least squares with l2 regularization.
        :param vis: default True to visualize
        :param save: default True to save the trained model
        :param regularization:
        :return: LinearRegression-object (or PolynomialFeatures-object if poly=True), true_test_result, predicted_result
        """

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        # Feature scaling
        sc_x = StandardScaler(copy=True, with_mean=True, with_std=True)
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        filename = StartModSKL.regress_linear.__name__

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

            if save:
                joblib.dump(lin_reg_2, filename+'_poly_model.sav')

            if regularization:
                StartMod.regularization(data, dependent_label)

            if vis:
                # Visual the result
                StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict,
                                         title='Training vs predicted data')

            return reg_poly, y_test, y_predict

        elif ridge:
            # apply Ridge regression
            model = Ridge()
            model.fit(x_train, y_train)

            y_predict = model.predict(model.fit_transform(x_test)) 

            if save:
                joblib.dump(model, filename+'_model.sav')

            if regularization:
                StartMod.regularization(data, dependent_label)

            if vis:
                # Visual the result
                StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict,
                                         title='Training vs predicted data')

            print("Calculating coefficients: ", model.coef_)
            print("Evaluation using r-square: ", model.score(x_test, y_test))

            return model, y_test, y_predict

        else:
            # fit model with training-data
            lin_reg = LinearRegression()
            lin_reg.fit(x_train, y_train)

            if save:
                joblib.dump(lin_reg, filename+'_model.sav')

            print("Calculating coefficients: ", lin_reg.coef_)
            print("Evaluation using r-square: ", lin_reg.score(x_test, y_test))

            # estimate the model by cross_validation method and training_data
            # StartMod.validation(reg_lin, x_train, y_train)

            # Predicting the Test and return the predicted result
            y_predict = lin_reg.predict(x_test)

            if regularization:
                StartMod.regularization(data, dependent_label)

            if vis:
                # Visualizing
                StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict,
                                         title='Training vs predicted data')

            return lin_reg, y_test, y_predict

    @classmethod
    def regress_knn(cls, data, dependent_label):
        """
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, random_state=cls.__random_state,
                                                               type_pd=False)

        reg_knn = KNeighborsRegressor()
        reg_knn.fit(x_train, y_train)

        # checking the magnitude of features
        features = data.columns.drop(dependent_label)
        features_important = reg_knn.feature_importances_
        print("Features: ", features, "\nFeatures_Important: ", features_important)

        # Predicting the Test and return the predicted result
        y_predict = reg_knn.predict(x_test)

        return reg_knn, y_test, y_predict

    @classmethod
    def regression_multi_linear(cls, data, dependent_label, pr=True, vis=True, save=True):
        """
        Description: apply method Multiple Linear regression y = b0 + b1.x1 + b2.x2 + ... + bn.xn
            choose the optimal feature_columns using algorithm Backward Elimination

        Reference:


        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param pr: default True to display info
        :param vis: default True to visualize
        :param save: default True to save the trained model
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

        x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size=0.2, random_state=cls.__random_state)

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

        if save:
            filename = StartModSKL.regression_multi_linear.__name__
            joblib.dump(reg, filename+'_model.sav')

        if vis:
            # plot all coefficients to see which features have the most impact on model
            # reg_coefficient = pd.DataFrame(data=reg.coef_, index=x_opt)
            # plt.figure(1)
            # reg_coefficient.plot(kind='bar')
            # plt.title("Feature_Important")
            # plt.xticks(rotation=45)

            # Visual the result
            StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict, title='Training vs predicted data')

        return reg, y_test, y_predict

    @classmethod
    def regress_decision_tree(cls, data, dependent_label, vis=False, save=True):
        """
        Description: apply method Decision Tree regression

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param vis: default False (set True to visualize)
        :param save: default True to save the trained model
        :return: DecisionTreeRegressor object, true_test_result, predicted_result
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, random_state=cls.__random_state,
                                                               type_pd=False)

        # Default parameters value: criterion='mean squared error' with max_depth=3 
        reg_dt = DecisionTreeRegressor(criterion='mse', max_depth=3, random_state=cls.__random_state)
        reg_dt.fit(x_train, y_train)

        # Estimate the model by cross_validation method and training_data
        # StartMod.validation(reg_dt, x_train, y_train)

        # checking the magnitude of features
        features = data.columns.drop(dependent_label)
        features_important = reg_dt.feature_importances_
        print("Features: ", features, "\nFeatures_Important: ", features_important)

        # Predict
        y_predict = reg_dt.predict(x_test)

        if save:
            filename = StartModSKL.regress_decision_tree.__name__
            joblib.dump(reg_dt, filename + '_model.sav')

        if vis:
            # plot all coefficients to see which features have the most impact on model
            data_features = pd.DataFrame(data=features_important, index=features)
            plt.figure(1)
            data_features.plot(kind='bar')
            plt.title("Feature_Important")
            plt.xticks(rotation=45)

            # Visual the result
            StartVis.vis_obj_predict(list(range(len(x_test))), y_test, y_predict, title='Training vs predicted data')

        return reg_dt, y_test, y_predict

    @classmethod
    def regress_random_forest(cls, data, dependent_label, n_trees=10, save=True):
        """
        Description: combine decision trees to form an ensemble random forest for regression in order to decrease the model’s variance. 

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param n_trees: the number of decision trees in the forest (default: 10)
        :param save: default True to save the trained model
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_rf = RandomForestRegressor(n_estimators=n_trees, random_state=cls.__random_state)
        reg_rf.fit(x_train, y_train)      

        # Predicting a new result
        y_predict = reg_rf.predict(x_test)

        if save:
            filename = StartModSKL.regress_random_forest.__name__
            joblib.dump(reg_rf, filename + '_model.sav')

        return reg_rf, y_test, y_predict

    @classmethod
    def regress_svr(cls, data, dependent_label, kn='rbf'):
        """
        Description: apply Support Vector Regression (SVR) to predict the information as real number
        References: 
            https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
            https://www.saedsayad.com/support_vector_machine_reg.htm
            https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: 
        :param kernel: default is Radial Basis Function ('rbf'), kernel_options = ['linear', 'poly', 'sigmoid']

        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_svr = SVR(gamma='auto', kernel=kn)
        reg_svr.fit(x_train, y_train)

        print("Number features: %d" % reg_svr.n_features_)
        print("Selected features: %s" % reg_svr.support_)
        print("Feature Ranking: %s" % reg_svr.ranking_)

        # Predicting the Test set results
        y_predict = reg_svr.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(reg_svr, x_train, y_train)

        return reg_svr, y_test, y_predict

    @classmethod
    def regress_logistic(cls, data, dependent_label, solver='liblinear', k_features=3, max_iter=100, multi_class='warn', save=True, regularization=True):
        """
        Description: apply method Regularized logistic regression
        apply multi_class = ('multinomial', 'ovr')
            with 'Multinomial Logistic Regression' classifier or 'ovr' one vs rest classifier
            to solve the classification problem with more than 2 labels.

        Solver:
            For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

            For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘
                liblinear’ is limited to one-versus-rest schemes.

            ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.

        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param solver: default 'liblinear', others: {‘newton-cg’, ‘lbfgs’, ‘sag’, ‘saga’}
        :param k_features: number of chosen features (default is)
        :param save: default True to save the trained model
        :param regularization: default True to turn on the regularization methods L1, L2
        :return: LogisticRegression_model, true_test_result, predicted_result
        """
        # # save the dependent value into y
        # y = data[dependent_label].values
        #
        # # drop dependent value from data and save the independent values into x
        # x = data.drop(dependent_label, axis=1).values

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        reg_log = LogisticRegression(random_state=cls.__random_state, solver=solver)
        
        # Feature extraction using Recursive Feature Elimination (RFE)
        rfe = RFE(reg_log, k_features)
        rfe_fit = rfe.fit(x_train, y_train)

        print("Number features: %d" % rfe_fit.n_features_)
        print("Selected features: %s" % rfe_fit.support_)
        print("Feature Ranking: %s" % rfe_fit.ranking_)

        # Predicting the Test set results
        y_predict = rfe_fit.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(rfe_fit, x_train, y_train)

        # estimate regularization
        if regularization:
            StartMod.regularization(data, dependent_label)

        if save:
            filename = StartModSKL.regress_random_forest.__name__
            joblib.dump(rfe_fit, filename + '_model.sav')

        return rfe_fit, y_test, y_predict

    @classmethod
    def classify_knn(cls, data, dependent_label, k_nb=5, save=True):
        """
        Description: 
            apply k-Nearest Neighbours method to classify data
            and scikit-optimize for tuning the hyperparameters 

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param k_nb: number of neighbors
        :param save: export the trained model (default True)
        :return: knn_model, true_test_result, predicted_result
        """

        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        clf_knn = KNeighborsClassifier(n_neighbors=k_nb, metric='euclidean', p=2)
        clf_knn.fit(x_train, y_train)

        # Predicting the Test set results
        y_predict = clf_knn.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_knn, x_train, y_train)

        if save:
            filename = StartModSKL.classify_knn.__name__
            joblib.dump(clf_knn, filename + '_model.sav')

        return clf_knn, y_test, y_predict

    @classmethod
    def classify_svm(cls, data, dependent_label, kernel='rbf', save=True):
        """
        Description: apply Support Vector Machine method to classify data
            Advice:
                n = number of features, m = number of training data
                n > m: use logistic regression or SVM without a kernel (linear kernel)
                n is small, m is intermediate: use SVM with Gaussian kernel ('rbf' radial basis function)
                n is small, m is large: create/ add more features, then logistic regression or SVM without a kernel

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            http://mlkernels.readthedocs.io/en/latest/kernels.html
            https://www.youtube.com/watch?v=FCUBwP-JTsA&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=75

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param kernel: default is Radial Basis Function ('rbf'), other kernel_options = ['linear', 'poly', 'sigmoid']
        :param save: default True to save the trained model
        :return: SVC_model, true_test_result, predicted_result
        """
        def kernel_compute(kn):
            kc_clf_svc = SVC(kernel=kn, random_state=cls.__random_state, gamma='auto')
            kc_clf_svc.fit(x_train, y_train)

            # Predicting the Test set results
            kc_y_predict = kc_clf_svc.predict(x_test, return_std=True)
            # cm = confusion_matrix(y_test, kc_y_predict)
            # kc_correct = cm[0][0] + cm[1][1]
            # print(kc_clf_svc, kc_correct)
            return kc_clf_svc, kc_y_predict

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        # TODO: Find and choose the best Kernel-SVM to fit with the Training set
        # kernel_options = ['linear', 'poly', 'sigmoid']
        classifier_svc, y_predict = kernel_compute(kernel)

        # for kernel in kernel_options:
        #     max_correct, clf_svc, y_predict = kernel_compute(kernel)
        #     if max_correct > default_max_correct:
        #         default_clf_svc = clf_svc
        #         default_y_predict = y_predict

        # estimate the model by cross_validation method and training_data
        # StartMod.validation(classifier_svc, x_train, y_train)

        if save:
            filename = StartModSKL.classify_knn.__name__
            joblib.dump(classifier_svc, filename + '_model.sav')

        return classifier_svc, y_test, y_predict

    @classmethod
    def classify_NaiveBayes(cls, data, dependent_label, clfnb='Gaussian', save=True):
        """
        Description: 
            apply Naive Bayes classification methods (GaussianNB, BernoulliNB, MultinomialNB) of Gaussian distributed input variables.
            BernoulliNB and MultinomialNB classifier are suitable for discrete data.
            
        References:
            http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical column
        :param clfnb: type of Naive Bayes (Gaussian, Bernoulli, Multinominal)
        :param save: default True to save the trained model
        :return: naiveBayes_model, true_test_result, predicted_result
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        if clfnb == 'Multinomial':
            clf = MultinomialNB()

        elif clfnb == 'Bernoulli':
            clf = BernoulliNB()

        else: 
            # default
            clf = GaussianNB()

        clf.fit(x_train, y_train)

        # make a probabilistic prediction
        y_predict_prob = clf.predict_proba(x_test)

        # predict the test set results
        y_predict = clf.predict(x_test)

        # estimate the model by cross_validation method and training_data
        print("Predicted Probabilities: ", y_predict_prob)
        print("Classes: ", clf.classes_, "Class Count: ", clf.class_count_,
              "Class Prior: ", clf.class_prior_)
        StartMod.validation(clf, x_train, y_train)

        if save:
            filename = StartModSKL.classification_gnb.__name__
            joblib.dump(clf, filename + '_model.sav')

        return clf, y_test, y_predict

    @classmethod
    def classify_bagged_dt(cls, data, dependent_label, n_trees, save=True):
        """
        Description: apply Ensemble meta-estimator BaggingClassifier Decision Tree to classify data

        # References:
            https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: (categorical) target label
        :param n_trees: number of trees
        :param save: default True to save the trained model
        :return: decision tree_model, y_test, y_predict
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        # Default parameters: criterion='entropy', max_depth of decision tree=4, random_state=1
        cart = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
        clf_bdt = BaggingClassifier(base_estimator=cart, n_estimators=n_trees, random_state=cls.__random_state)
        clf_bdt.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_bdt.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_bdt, x_train, y_train)

        if save:
            filename = StartModSKL.classification_bagged_dty__
            joblib.dump(clf_bdt, filename + '_model.sav')

        return clf_bdt, y_test, y_predict

    @classmethod
    def classify_extraTrees(cls, data, n_estimators):
        """
        Description: 
            classification using extra Trees
        """
        array = data.values
        X = array[:, 0:len(array)]
        Y = array[:, len(array)] 

        # feature extraction
        model = ExtraTreesClassifier(n_estimators=n_estimators)
        model.fit(X, Y)
        print(model.feature_importances_)
        return model

    @classmethod
    def classify_rf(cls, data, dependent_label, n_trees, max_features, save=True):
        """
        Description: apply Ensemble Random Forest for classification

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            http://scikit-learn.org/stable/modules/ensemble.html#random-forests
            https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param n_trees: number of decision trees
        :param max_features: number of maximal features to select randomly
        :param save: default True to save the trained model
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        # In order to reduce the size of the model, you can change these parameters:
        # min_samples_split, min_samples_leaf, max_leaf_nodes and max_depth
        # setup n_jobs=-1 to setup all cores available on the machine are used
        clf_rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_features, n_jobs=-1)
        clf_rf.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_rf.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_rf, x_train, y_train, vis=False)

        if save:
            filename = StartModSKL.ye__
            joblib.dump(clf_rf, filename + '_model.sav')

        return clf_rf, y_test, y_predict

    @classmethod
    def classify_adab(cls, data, dependent_label, n_trees=30, save=True):
        """
        Description: apply Ensemble AdaBoost for classification

        # References:
            http://scikit-learn.org/stable/modules/ensemble.html#adaboost

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param n_trees:
        :param save: default True to save the trained model
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        clf_adab = AdaBoostClassifier(n_estimators=n_trees, random_state=cls.__random_state)
        clf_adab.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_adab.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_adab, x_train, y_train, vis=False)

        if save:
            filename = StartModSKL.y__
            joblib.dump(clf_adab, filename + '_model.sav')

        return clf_adab, y_test, y_predict

    @classmethod
    def classify_sgb(cls, data, dependent_label, n_trees=30, regression=False, save=True):
        """
        Description: apply Ensemble Stochastic Gradient Boosting for classification

        # References:
            http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting
            http://scikit-learn.org/stable/modules/ensemble.html#loss-functions

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param n_trees: number of trees
        :param regression:
        :param save: default True to save the trained model
        :return:
        """
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        if regression:
            clf_sgb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                                                random_state=cls.__random_state, loss='ls')
        else:
            clf_sgb = GradientBoostingClassifier(n_estimators=n_trees, learning_rate=1.0, max_depth=1,
                                                 random_state=cls.__random_state, loss='deviance')
        clf_sgb.fit(x_train, y_train)

        # predict the test set results
        y_predict = clf_sgb.predict(x_test)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(clf_sgb, x_train, y_train, vis=False)

        if save:
            filename = StartModSKL.ye__
            joblib.dump(clf_sgb, filename + '_model.sav')

        return clf_sgb, y_test, y_predict

    @classmethod
    def classify_xgb(cls, data, dependent_label, save=True):
        """
        Description: apply Extreme Gradient Boosting (XGBoost) method to classify data

        # References:
            http://xgboost.readthedocs.io/en/latest/model.html
            https://machinelearningmastery.com/xgboost-python-mini-course/
            http://scikit-learn.org/stable/modules/ensemble.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param save: default True to save the trained model
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

        if save:
            filename = StartModSKL.y__
            joblib.dump(clf_xgb, filename + '_model.sav')

        return clf_xgb, y_test, y_predict

    @classmethod
    def classify_voting(cls, models, data, dependent_label):
        """
        Description: voting ensemble model for classification by combining the predictions from multiple machine learning algorithms.
            parameter:
                voting='hard': mode label
                voting='soft': weighted average value

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
            http://scikit-learn.org/stable/modules/ensemble.html#voting-classifier

        :param models:
        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param save: default True to save the trained model
        :return:
        """
        # split data into feature (independent) values and dependent values in type Numpy.array
        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_label, type_pd=False)

        ensemble = VotingClassifier(models)

        # estimate the model by cross_validation method and training_data
        StartMod.validation(ensemble, x_train, y_train)

        return ensemble

    @classmethod
    def cluster_k_mean_noc(cls, data, plot=False, save=True):
        """
        Description: find the number of clusters using the elbow method.

        # References:
            http://www.awesomestats.in/python-cluster-validation/

        :param data: pandas.core.frame.DataFrame
        :param plot: show plot (default is False)
        :param save: default True to save the trained model
        :return: plot of data to identify number of clusters (still manually)
        """
        cluster_errors = []

        # given the range of number of clusters from 1 to 10
        cluster_range = range(1, 11)

        for i in cluster_range:
            k_means = KMeans(n_clusters=i, init='k-means++', random_state=cls.__random_state)
            k_means.fit(data.values)
            cluster_errors.append(k_means.inertia_)

        if save:
            filename = StartModSKL.cluster_k_mean_noc.__name__
            joblib.dump(k_means, filename + '_model.sav')

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
    def cluster_k_mean(cls, data, n_clusters, save=True):
        """
        Description:
            Requirement: run cluster_k_mean_noc first to find 'noc' (number of possible clusters)
            Apply method clustering k_means++ to cluster data with the given 'noc'

        # References:
            http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

        :param data: pandas.core.frame.DataFrame
        :param n_clusters: number of clusters
        :param save: default True to save the trained model
        :return: KMeans object, predicted y_values
        """
        k_means = KMeans(n_clusters=n_clusters, init='k-means++', random_state=cls.__random_state)
        y_clusters = k_means.fit_predict(data.values)

        if save:
            filename = StartModSKL.cluster_k_mean.__name__
            joblib.dump(k_means, filename + '_model.sav')

        return k_means, y_clusters
 
    @staticmethod
    def info_help():
        info = {
            "info_help_StartModSKL": StartModSKL.__name__,
            "StartModSKL.regress_linear": StartModSKL.regress_linear.__doc__,
            }
        # info.update(StartML.info_help())

        return info


info_modskl = StartModSKL.info_help()
