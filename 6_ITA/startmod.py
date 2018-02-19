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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# from sklearn.pipeline import make_pipeline
from startml import *


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

        :param data:
        :param label_columns
        :param one_hot: Boolean-value, True to choose method OneHotEncoder
        :return: data and x_values (the encoded data in type numpy.array)
        """

        for label_column in label_columns:
            try:
                # Encode label only applies to Column in type Object
                if data[label_column].dtype == np.float64 or data[label_column].dtype == np.int64:
                    print("Warning: type of label_column " + label_column + " is " + str(data[label_column].dtypes))
            except ValueError:
                return []

        x_values = data.values
        for label_column in label_columns:
            label_idx = data.columns.get_loc(label_column)
            # label_encoder to turn object-column into number-column
            label_encoder = LabelEncoder()
            x_values[:, label_idx] = label_encoder.fit_transform(x_values[:, label_idx])

        if one_hot:
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

            # TODO: replace data.columns into new_data.columns (parameter to transfer column name in fit_transform ?)
            # new_data = pd.DataFrame(data=new_values, columns=...)

            return pd.DataFrame(data=x_values)
        else:
            return pd.DataFrame(data=x_values, columns=data.columns, index=None)

    @classmethod
    def split_data(cls, data, dependent_label, test_size=0.2, random_state=0, type_pd=True, split=True):
        """
        split data for regression methods

        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        :param data: Pandas-DataFrame
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
            # print(type(x), type(y))
        else:
            # convert to type Numpy
            y = data[dependent_label].values
            # drop dependent value from data and save the independent values into x
            x = data.drop(dependent_label, axis=1).values
            # print(type(x), type(y))

        if not split:
            return x, y

        try:
            # split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        except ValueError:
            print("Data set is not valid yet, need to be preprocessed first")
            print("No splitting happen")
            return data

        return x_train, x_test, y_train, y_test

    @classmethod
    def feature_columns(cls, data, label=None):
        """
        find and return object and non-object columns
        :param data:
        :param label=None in default
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
        :return: data in scaled format
        """
        if type_pd:
            # convert data in Pandas DataFrame, apply Min_Max method
            # print(data.columns)
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
    def feature_selection(cls, data):
        """
        feature selection/dimensionality reduction
        Source:
            http://scikit-learn.org/stable/modules/feature_selection.html
        :param data:
        :return:
        """
        # tbd
        pass

    @classmethod
    def feature_hashing(cls, data):
        """
        Benefit on low-memory and speed up the performance
        Source:
            http://scikit-learn.org/stable/modules/feature_extraction.html
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
        :param data:
        :return:
        """
        # tbd
        pass

    @classmethod
    def feature_extraction(cls, data, dependent_variable):
        """
        Using Principal component analysis (PCA) to extract the most important independent variables
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        :param data:
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
    def feature_engineering(cls, data, old_feature, new_feature, attributes_new_feature):
        """
        Source:
            https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

        renew data with new_feature which has the attributes_new_feature
        :param data:
        :param old_feature:
        :param new_feature:
        :param attributes_new_feature:
        :return: data (with new feature)
        """

        def find_attributes_feature(search_object, attributes):
            for attr in attributes:
                if attr in search_object:
                    # print(attr, search_object)
                    return attr
            return np.nan

        data[new_feature] = data[old_feature].map(lambda x: find_attributes_feature(x, attributes_new_feature))
        return data

    @classmethod
    def feature_engineering_merge_cols(cls, data, features, new_feature):
        """
        Merge many features into one new_feature (column)
        :param data:
        :param features: list of the merging features
        :param new_feature: name of merged new_feature
        :return: data (with new column feature)
        """
        # initialize new feature column
        data[new_feature] = 0

        for feature in features:
            data[new_feature] = data[new_feature]+data[feature]

        # data = data.drop(features, axis=1)
        return data.drop(features, axis=1)

    @classmethod
    def metrics_report(cls, y_true, y_pred, target_names=None):
        """
        measure the quality of the models (comparing results before and after running prediction)
        Source:
            https://www.kaggle.com/dansbecker/handling-missing-values
            http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
            http://scikit-learn.org/stable/modules/model_evaluation.html#model-evaluation
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        :param y_true: the truth values
        :param y_pred: the predicted values
        :param target_names: label (categorical) name
        :return:
        """
        # tbd
        if target_names is not None:
            print("Classification Report: \n", classification_report(y_true, y_pred, target_names=target_names))
            print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred, labels=target_names))
        else:
            print("Classification Report: \n", classification_report(y_true, y_pred))
            print("Confusion Matrix: \n", confusion_matrix(y_true, y_pred))

    @staticmethod
    def info_help():
        info = {
            "info_help_StartMod": StartMod.__name__,
            "StartMod.split_data": StartMod.split_data.__doc__,
            "StartMod.feature_engineering(data)": StartMod.feature_engineering.__doc__,
            "StartMod.feature_engineering_merge_cols(data)": StartMod.feature_engineering_merge_cols.__doc__,
            }
        # info.update(StartML.info_help())

        return info


info_mod = StartMod.info_help()
