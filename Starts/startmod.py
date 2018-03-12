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


from Starts.startml import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import make_pipeline

import statsmodels.formula.api as sm


class StartMod(StartML):
    """
      Description: StartMod - Start Models
      Apply different Models in Machine Learning Regression and Classification
        k-NN, Decision Tree, SVM, Neural Network, etc.

      Start:
          jupyter notebook
          -> from startmod import *
          -> info_mod
    """

    @classmethod
    def encode_label_column(cls, data, label_columns, one_hot=False):
        """
        Encode object-columns
        This encoding is needed for feeding categorical data to many scikit-learn estimators,
        notably linear models and SVMs with the standard kernels.

        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        :param data: pandas.core.frame.DataFrame
        :param label_columns: list of all labels (object-type)
        :param one_hot: Boolean-value, set value True for categorical column
        :return: data and x_values (the encoded data in type numpy.array)
        """

        for label_column in label_columns:
            try:
                # Encode label only applies to Column in type Object
                if data[label_column].dtype == np.float64 or data[label_column].dtype == np.int64:
                    print("Type of label_column " + label_column + " is " + str(data[label_column].dtypes))
            except ValueError:
                return []

        x_values = data.values
        try:
            for label_column in label_columns:
                label_idx = data.columns.get_loc(label_column)
                # label_encoder to turn object-column into number-column
                label_encoder = LabelEncoder()
                x_values[:, label_idx] = label_encoder.fit_transform(x_values[:, label_idx])
        except TypeError:
            print("Data might have mixed type str and float, please make sure there is no (float) NaN-Value")

        # one_hot=True is used for categorical column
        if one_hot:
            try:
                labels_idx = []
                for label_column in label_columns:
                    # get column index
                    label_idx = data.columns.get_loc(label_column)
                    # data.values[:, label_idx] = label_encoder.fit_transform(data.values[:, label_idx])
                    labels_idx.append(label_idx)

                # create dummy columns to encode the above label_column
                one_hot_encoder = OneHotEncoder(categorical_features=labels_idx)

                # data.values[:, label_idx]
                # = one_hot_encoder.fit_transform(data.values[:,label_idx].reshape(1,-1)).toarray()
                x_values = one_hot_encoder.fit_transform(x_values).toarray()

                # remove dummy trap
                # x_values = x_values[:, 1:]

                # TODO: replace data.columns into new_data.columns (parameter to transfer column name in fit_transform?)
                # new_data = pd.DataFrame(data=new_values, columns=...)
                data = pd.DataFrame(data=x_values)

                # convert to numeric-type DataFrame
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col])
            except ValueError:
                print("Data might be containing NaN_values")

            return data

        else:
            # one_hot=False is used for object_feature columns
            data = pd.DataFrame(data=x_values, columns=data.columns, index=None)
            # convert to numeric-type DataFrame
            for col in data.columns:
                data[col] = pd.to_numeric(data[col])
            return data

    @classmethod
    def split_columns(cls, data):
        """
        Split data by feature_columns into 2 different datasets
        :param data:
        :return:
        """
        pass

    @classmethod
    def split_data(cls, data, dependent_label, test_size=0.2, random_state=0, type_pd=True, split=True):
        """
        Split data by rows into training_data and test_data used for (regression, classification) methods

        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: categorical label
        :param test_size: (default is 0.2)
        :param random_state: (default is 0)
        :param type_pd: (default is Pandas Dataframe)
        :param split: (default is True)
        :return: x_train, x_test, y_train, y_test (default type Pandas DataFrame)
        """

        # convert data into numpy-values (in case: the last column is dependent label)
        # x = data.iloc[:, :-1].values
        # y = data.iloc[:, 1].values
        # save the dependent value into y

        if type_pd:
            # convert to type Pandas DataFrame
            # y = data.pop(dependent_label)  # should not use pop, instead of using data[dependent_label]
            y = data[dependent_label]
            x = data
        else:
            # convert to type Numpy
            y = data[dependent_label].values
            # drop dependent value from data and save the independent values into x
            x = data.drop(dependent_label, axis=1).values

        if not split:
            return x, y

        try:
            # split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                                random_state=random_state, shuffle=True)
        except ValueError:
            print("Data set is not valid yet, need to be preprocessed first")
            print("No splitting happen")
            return data

        return x_train, x_test, y_train, y_test

    @classmethod
    def backward_eliminate(cls, data, x_data, y_data):
        """
        Support the evaluation on (regression) models by finding maximal pvalue (< pre-defined SL)
        and applying method Backward Elimination for feature selection

        Source:
            http://www.stephacking.com/multivariate-linear-regression-python-step-6-backward-elimination/

        :param data: pandas.core.frame.DataFrame
        :param x_data: feature_values in type numpy.ndarray
        :param y_data: categorical_values in numpy.ndarray
        :return: regressor_ols object, max_pvalue, x_opt
        """

        # Start Method Backward Elimination for feature selection
        # Step 1: select a significance level to stay
        sl = 0.05
        # initiate data for x with full columns
        x_opt = x_data[:, [i for i in range(len(data.columns))]]

        # Step 2: fit the full model with all possible predictors, create new object in (Ordinary Least Squares) OLS
        reg_ols = sm.OLS(endog=y_data, exog=x_opt).fit()

        # Step 3: select and remove the redundant columns with pvalue > significant level 0.05
        max_pvalue, col_idx = StartML.find_idx_max_value(reg_ols.pvalues)

        while max_pvalue > sl:
            # remove value at column col_idx and refresh data of x_opt
            x_opt = np.delete(x_opt, col_idx, axis=1)

            # recompute regressor with new x_opt
            reg_ols = sm.OLS(endog=y_data, exog=x_opt).fit()
            max_pvalue, idx = StartML.find_idx_max_value(reg_ols.pvalues)

        return reg_ols, max_pvalue, x_opt

    @classmethod
    def feature_columns(cls, data, label=None):
        """
        find and return object and non-object columns
        :param data: pandas.core.frame.DataFrame
        :param label: default is None
        :return: list of non_obj_feature, list of obj_feature
        """
        if label is None:
            non_obj_feature = [lab for lab in data.columns if data[lab].dtype != 'object']
            obj_feature = [lab for lab in data.columns if data[lab].dtype == 'object']
            return non_obj_feature, obj_feature
        else:
            # x, y = train_data, train_data.pop('Time') will remove column 'Time' completely out of train_data
            train_x, _ = StartMod.split_data(data, split=False)
            non_obj_feature = [lab for lab in train_x.columns if train_x[lab].dtype != 'object']
            obj_feature = [lab for lab in train_x.columns if train_x[lab].dtype == 'object']

            return non_obj_feature, obj_feature

    @classmethod
    def feature_scaling(cls, data, type_pd=True, std=True):
        """
        Standardization involves rescaling the features such that they have the properties
        of a standard normal distribution with a mean of zero and a standard deviation of one

        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html

        :param data:
        :param type_pd: default True to convert data in Pandas Data-Frame
        :return: data in scaled format
        """
        if type_pd:
            # convert data in Pandas DataFrame, apply Min_Max method
            data[data.columns] = data[data.columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            return data
        else:
            if std:
                scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            else:
                scaler = MinMaxScaler()
            # Compute the mean and std to be used for later scaling
            scaler.fit(data)
            return scaler.transform(data)

    @classmethod
    def feature_selection(cls, data, rm_columns, dependent_label=None, rm=False, pr=True):
        """
        simply feature selection/ dimensionality reduction
        apply Backward Elimination, Forward Selection, Bidirectional Elimination, Score comparision

        Source:
            http://scikit-learn.org/stable/modules/feature_selection.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_label: label of categorical column
        :param rm_columns: list of feature_columns which will be removed
        :param rm: default False (if True: columns from rm_columns will be removed)
        :param pr: turn on/ off print function (default: True to turn on)
        :return:
        """
        # calculate R_Squared & adj_R_Squared with full feature_columns
        X, y = StartMod.split_data(data, dependent_label, type_pd=False, split=False)

        try:
            regressor_ols = sm.OLS(endog=y, exog=X).fit()
        except TypeError:
            print("Data might have NaN_value, please clean them before building model")
            return data

        # backup data for not removing case
        orig_data = data
        if pr:
            print(data.columns)
            print("\nRSquared: ", regressor_ols.rsquared)
            print("\nAdj_RSquared: ", regressor_ols.rsquared_adj)
            print("\n", regressor_ols.summary())

        # calculate r_squared, adj_r_squared on the removing columns
        for rmc in rm_columns:
            print("\nRemove column: ", rmc)
            data = data.drop(rmc, axis=1)
            print(data.columns)
            X, y = StartMod.split_data(data, dependent_label, type_pd=False, split=False)

            regressor_ols = sm.OLS(endog=y, exog=X).fit()
            if print:
                print("RSquared: ", regressor_ols.rsquared)
                print("Adj_RSquared: ", regressor_ols.rsquared_adj)
                print("\n", regressor_ols.summary())

        # if yes, return data without column
        if rm:
            return data
        else:
            # return original data without any damages
            return orig_data

    @classmethod
    def feature_hashing(cls, data):
        """
        Benefit on low-memory and speed up the performance

        Source:
            http://scikit-learn.org/stable/modules/feature_extraction.html
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        # tbd
        pass

    @classmethod
    def feature_extraction(cls, data, dependent_variable):
        """
        Using Principal component analysis (PCA) to extract the most important independent variables (features)

        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        :param data: pandas.core.frame.DataFrame
        :param dependent_variable:
        :return: array of explained_variance_ratio, x_train, x_test, y_train, y_test
        """
        # tbd: choose the most best variance to get max possible total percentages

        x_train, x_test, y_train, y_test = StartMod.split_data(data, dependent_variable)
        pca = PCA(n_components=None)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        return pca.explained_variance_ratio_, x_train, x_test, y_train, y_test

    @classmethod
    def feature_engineering(cls, data, old_feature, new_feature, new_attributes, rm=False):
        """
        Renew data with new_feature using the new attributes_new_feature

        Source:
            https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

        :param data: pandas.core.frame.DataFrame
        :param old_feature:
        :param new_feature:
        :param new_attributes:
        :return: data (with new feature)
        """

        def find_attributes_feature(search_object, attributes):
            for attr in attributes:
                if attr in search_object:
                    # print(attr, search_object)
                    return attr
            return np.nan

        data[new_feature] = data[old_feature].map(lambda x: find_attributes_feature(x, new_attributes))
        if rm:
            data = data.drop(old_feature, axis=1)
        return data

    @classmethod
    def feature_engineering_merge_cols(cls, data, features, new_feature, datetime=False):
        """
        Merge many features into one new_feature (column)

        :param data: pandas.core.frame.DataFrame
        :param features: list of the merging features
        :param new_feature: name of merged new_feature (plus_combined)
        :param datetime: default is False (if True, then it will merge list of ['date', 'time'] features into
        new_feature (e.g. 'Date_Time')
        :return: data (with new column feature)
        """
        if datetime:
            # merge columns features ['Date','Time'] into new column new_feature ['Date_Time']
            data[new_feature] = data[features].apply(lambda x: pd.datetime.combine(*list(x)), axis=1)
        else:
            # initialize new feature column
            data[new_feature] = 0

            for feature in features:
                data[new_feature] = data[new_feature]+data[feature]

        return data.drop(features, axis=1)

    @classmethod
    def metrics_report(cls, y_true, y_pred, target_names=None):
        """
        measure the quality of the models (comparing results before and after running prediction)

        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 Score = 2 * Precision * Recall / (Precision + Recall)

        Source:
            http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
            http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

        :param y_true: the truth values
        :param y_pred: the predicted values
        :param target_names: label (categorical) name
        :return:
        """

        if target_names is not None:
            print("Classification Report: \n", classification_report(y_true, y_pred, target_names=target_names))
            print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred, labels=np.unique(y_true)))
        else:
            print("Classification Report: \n", classification_report(y_true, y_pred))
            print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred, labels=np.unique(y_true)))
        print("\nAccuracy: \n", accuracy_score(y_true, y_pred))

    @classmethod
    def validation(cls, classifier, x_train, y_train, cv=None):
        """
        Apply K-Fold Cross_Validation to estimate the model (classification)

        Source:
            http://scikit-learn.org/stable/modules/cross_validation.html

        :param classifier:
        :param x_train: feature_values
        :param y_train: categorical_values
        :param cv (Cross_Validation) default is 10-fold if None
        :return:
        """
        accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=cv)
        print("\nAccuracies: ", accuracies)
        print("\nMean of accuracies: ", accuracies.mean())
        print("\nStandard Deviation: ", accuracies.std())

        accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
        accuracies.mean()
        accuracies.std()

    @staticmethod
    def info_help():
        info = {
            "info_help_StartMod": StartMod.__name__,
            "StartMod.split_data": StartMod.split_data.__doc__,
            "StartMod.feature_engineering(data)": StartMod.feature_engineering.__doc__,
            "StartMod.feature_engineering_merge_cols(data)": StartMod.feature_engineering_merge_cols.__doc__,
            }

        return info


info_mod = StartMod.info_help()