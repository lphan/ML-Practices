#!/usr/bin/env python3
#
# Copyright (c) 2014-2015
#
# This software is licensed to you under the GNU General Public License,
# version 2 (GPLv2). There is NO WARRANTY for this software, express or
# implied, including the implied warranties of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. You should have received a copy of GPLv2
# along with this software; if not, see
# http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt.

__author__ = 'Long Phan'


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
      -> info_help
    """

    @classmethod
    def feature_hashing(cls, data):
        """
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
        :param data:
        :return:
        """
        # tbd
        pass

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
        :return: data with new feature
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
        merge many features into one new_feature (column)
        :param data:
        :param features: list of the merging features
        :param new_feature: name of merged new_feature
        :return: data with new column (feature)
        """
        # initialize new feature column
        data[new_feature] = 0

        for feature in features:
            data[new_feature] = data[new_feature]+data[feature]

        # data = data.drop(features, axis=1)
        # and also remove all old features
        return data.drop(features, axis=1)

    @classmethod
    def encode_label_column(cls, data, label_column):
        """
        Source:
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

        Encode object-columns
        This encoding is needed for feeding categorical data to many scikit-learn estimators,
        notably linear models and SVMs with the standard kernels.
        :param data:
        :param label_column
        :return: the encoded data in type numpy.array
        """
        try:
            if data[label_column].dtype == np.float64 or data[label_column].dtype == np.int64:
                print("label_column is not valid, only accept type-object ")
        except ValueError:
            return []

        x_values = data.values

        # get column index
        label_idx = data.columns.get_loc(label_column)
        # print(label_idx)

        # transform into numpy values
        # print(x_values)

        # label_encoder to turn object-column into number-column
        label_encoder = LabelEncoder()
        # print(x_values[:, label_idx])

        x_values[:, label_idx] = label_encoder.fit_transform(x_values[:, label_idx])
        # print(X[:, label_idx])

        # tbd:
        # to make the code as X = onehotencoder.fit_transform(X).toarray() to create many dummy-columns
        # one_hot_encoder to turn number-column into sparse matrix without reshape(1, -1)
        one_hot_encoder = OneHotEncoder(categorical_features=[label_idx])
        x_values[:, label_idx] = one_hot_encoder.fit_transform(x_values[:, label_idx].reshape(1, -1)).toarray()

        return x_values

    @classmethod
    def score_dataset(cls):
        """
        Source:
            https://www.kaggle.com/dansbecker/handling-missing-values
        measure the quality of the models (comparing results before and after running prediction)

        :return:
        """
        # tbd
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartMod": StartMod.__name__,
            "StartMod.feature_engineering(data)": StartMod.feature_engineering.__doc__,
            "StartMod.feature_engineering_merge_cols(data)": StartMod.feature_engineering_merge_cols.__doc__,
            }
        info.update(StartML.info_help())

        return info


info_help = StartMod.info_help()
