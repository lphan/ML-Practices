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

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from startmod import *
from startmodskl import StartModSKL


class StartModTF(StartMod):
    """
        Description: StartModTF - Start Models Tensorflow
        Pre-made Estimators (regression, classification) and Custom Estimators

        Sources:
            https://www.tensorflow.org/api_docs/python/
            https://github.com/tensorflow/models/tree/master/samples/core/get_started
            https://github.com/tensorflow/models/tree/master/samples/cookbook/regression

        Start:
          jupyter notebook
          -> from startmodtf import *
          -> info_modtf
    """

    def __init__(self, data, label=None, hidden_units=[10, 10], n_classes=2, optimizer="Adagrad", activation_fn="relu",
                 learning_rate=0.001, steps=1000, batch_size=10, num_epochs=10):
        """
        setup parameters for neuron network
        """
        self.label = label
        self.hidden_units = hidden_units  # default setup 10 neuron in 2 layers
        self.n_classes = n_classes  # default setup 2 classes
        self.optimizer = optimizer  # default Adagrad optimizer
        self.activation_fn = activation_fn  # default 'relu' function
        self.learning_rate = learning_rate  # default 0.001
        self.steps = steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if self.label is not None:
            self.data = data
            self.x_train, self.x_test, self.y_train, self.y_test = StartMod.split_data(data, self.label, typenp=False)

    @classmethod
    def train_input_func(cls, features, labels, batch_size, epochs):
        """
        An input function for training
        Source:
            https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
        """

        # if features and labels are numpy-type, then use numpy_input_fn
        return tf.estimator.inputs.pandas_input_fn(x=features, y=labels, batch_size=batch_size,
                                                   num_epochs=epochs, shuffle=True)
        # Alternatives:
        # tensors = (dict(features), labels)
        #
        # dataset = tf.data.Dataset.from_tensor_slices(tensors)
        #
        # # Shuffle, repeat, and batch the examples.
        # return dataset.shuffle(1000).repeat().batch(batch_size)

    @classmethod
    def eval_input_fn(cls, features, label, batch_size, epochs):
        """
        An input function for evaluation
        Source:
            https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
        """
        return tf.estimator.inputs.pandas_input_fn(x=features, y=label, batch_size=batch_size,
                                                   num_epochs=epochs, shuffle=False)

    @classmethod
    def pred_input_fn(cls, test_features, batch_size, epochs):
        """
        An input function for prediction
        :return:
        """
        return tf.estimator.inputs.pandas_input_fn(x=test_features, batch_size=batch_size, num_epochs=epochs,
                                                   shuffle=False)

    @classmethod
    def keras_sequential(cls, data, dependent_label):
        """
        Setup Keras and run the Sequential method to predict value
        Source:
            https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py
        :param data: Pandas-DataFrame
        :param dependent_label: sequential-object model, predicted value, actual (true) value
        :return: Keras-Sequential object, the actual (true) value, the predicted value
        """
        # Initialising the ANN
        model = Sequential()

        x_train, x_test, y_train, y_test = StartModSKL.split_data(data, dependent_label)

        # number of nodes in the hidden-layer
        # tbd: use parameter tuning to find the exact numbers
        input_weight_combis = 6
        output_weight_combis = 1

        # Create n=2 layers neural network (idea: setup parameters 'how many layers' from config.ini)
        # Adding the input layer and the first hidden layer, activation function as rectifier function
        model.add(Dense(activation="relu", input_dim=x_train.shape[1], units=input_weight_combis,
                        kernel_initializer="uniform"))

        # Adding the second hidden layer, activation function as rectifier function
        model.add(Dense(activation="relu", units=input_weight_combis, kernel_initializer="uniform"))

        # Adding the output layer (in case of there's only one dependent_label), activation function as sigmoid function
        model.add(Dense(activation="sigmoid", units=output_weight_combis, kernel_initializer="uniform"))

        # Compiling the ANN with optimizer='adam'
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # tbd compute and find the appropriate batch_size and epochs
        # b_s = int(len(train_data)/10)  # default batch_size=32
        n_e = x_train.shape[1]   # default nb_epoch=10

        # fit the keras_model to the training_data and see the real time training of model on data
        # with result of loss and accuracy.
        # The smaller batch_size and higher epochs, the better the result. However, slow_computing!
        model.fit(x_train, y_train, batch_size=1, epochs=n_e)

        # predictions and evaluating the model
        y_predict = model.predict(x_test)

        return model, y_test, y_predict

    @classmethod
    def setup_feature_columns(cls, data):
        """
        Source:
            https://www.tensorflow.org/get_started/feature_columns
        :param data: Pandas-DataFrame
        :return:
        """
        # setup feature columns
        non_obj_feature, obj_feature = StartMod.feature_columns(data)

        # tbd
        # setup categorical features (categorical_column_with_vocabulary_list) or (categorical_column_with_hash_bucket)

        # setup continuous features
        feature_columns = [tf.feature_column.numeric_column(key=fea) for fea in non_obj_feature]
        # tbd: process object_feature

        return feature_columns

    @classmethod
    def regressor_custom(cls, data):
        """
        tbd
        Source:
            https://github.com/tensorflow/models/tree/master/samples/cookbook/regression
        :param data: Pandas-DataFrame
        :return:
        """
        # batch_size

        # Variables

        # Placeholders

        # Graph

        # Loss Function

        # Optimizer

        # Initialize Variables

        # Setup Session

        pass

    @classmethod
    def regressor_estimator(cls, data):
        """
        apply Estimator API
        :param data: Pandas-DataFrame
        :return:
        """

        # init feature_columns

        # setup estimator inputs

        # train the estimator

        # evaluation

        # predictions
        pass

    @classmethod
    def classifier_custom(cls, data):
        """
        tbd
        :param data: Pandas-DataFrame
        :return:
        """
        # retrieve value of feature_columns

        # create model_obj

        # evaluation

        # predictions

        pass

    def classifier_estimator(self, model_lin=True):
        """
        apply pre-made Estimator to classify data
        :param data: Pandas-DataFrame
        :param model_lin: default is LinearClassifier
        :return:
        """

        # create model_obj
        fea_cols = StartModTF.setup_feature_columns(self.data)

        if model_lin:
            classifier = tf.estimator.LinearClassifier(feature_columns= fea_cols, n_classes=self.n_classes,
                                                       optimizer='Ftrl')
        else:
            classifier = tf.estimator.DNNClassifier(feature_columns=fea_cols, hidden_units=self.hidden_units,
                                                    n_classes=self.n_classes, optimizer=self.optimizer,
                                                    activation_fn=self.activation_fn)
        # Train the Model
        train_input_func = StartModTF.train_input_func(features=self.x_train, labels=self.y_train,
                                                       batch_size=self.batch_size, epochs=self.num_epochs)
        classifier.train(input_fn=train_input_func, steps=1000)

        # Evaluation
        eval_input_func = StartModTF.eval_input_fn(features=self.x_train, label=self.y_train,
                                                   batch_size=self.batch_size, epochs=self.num_epochs)
        result_eval = classifier.evaluate(input_fn=eval_input_func)

        # Prediction
        pred_input_func = StartModTF.pred_input_fn(test_features=self.x_test, batch_size=self.batch_size,
                                                   epochs=self.num_epochs)
        result_predict = classifier.predict(input_fn=pred_input_func)

        return result_eval, list(result_predict)

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModTF": StartModTF.__name__,
            "StartModTF.(data)": StartModTF.regressor_custom.__doc__
            }
        # info.update(StartML.info_help())

        return info


info_modtf = StartModTF(train_data)
info_modtf.info_help()
