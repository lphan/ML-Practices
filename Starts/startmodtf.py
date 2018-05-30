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
from keras.layers import Dense, Flatten, Conv1D, Dropout
from keras.optimizers import SGD
from keras.initializers import random_uniform

from Starts.startmod import *


class StartModTF(StartMod):
    """
        Description: StartModTF - Start Models Tensorflow
        Pre-made Estimators (regression, classification) and Custom Estimators

        References:
            https://www.tensorflow.org/api_docs/python/
            https://github.com/tensorflow/models/tree/master/samples/core/get_started
            https://github.com/tensorflow/models/tree/master/samples/cookbook/regression

        Start:
          jupyter notebook
          -> from startmodtf import *
          -> info_modtf
    """

    def __init__(self, data, label=None):
        """
        setup parameters for neuron network
        :param data: pandas.core.frame.DataFrame
        :param label: target_feature
        """
        self.data = data
        if label is None:
            self.n_classes = 2  # default setup 2 classes for binary classification (2 values e.g. 0 and 1)
            self.label = None

        elif len(label) == 1:
            self.n_classes = len(data[label].unique())
            self.label = label

        else:
            self.label = label  # regression

        self.hidden_units = [10, 10]    # default setup 10 neuron in 2 layers
        self.optimizer = "Adagrad"      # default Adagrad optimizer
        self.activation_fn = "relu"     # default 'relu' function
        self.learning_rate = 0.001      # default 0.001
        self.steps = 1000               # default training_steps 1000

        # (correspond with available system memory capacity to avoid outofmemory_error,
        # small for many features, big for performance)
        self.batch_size = 10            # default batch_size 10

        self.nr_epochs = 1             # default number of epochs 1
        self.feature_scl = False        # default turn off feature scaling

    # get and set methods for attributes
    def _get_attributes(self):
        return self.hidden_units, self.optimizer, self.activation_fn, self.learning_rate, \
               self.steps, self.batch_size, self.nr_epochs, self.feature_scl

    # reset all attributes in neural network
    def _set_attributes(self, dict_params):
        self.hidden_units = dict_params['hidden_units']
        self.optimizer = dict_params['optimizer']
        self.activation_fn = dict_params['activation_fn']
        self.learning_rate = dict_params['learning_rate']
        self.steps = dict_params['steps']
        self.batch_size = dict_params['batch_size']
        self.nr_epochs = dict_params['num_epochs']
        self.feature_scl = dict_params['feature_scl']

    def info_parameters(self):
        print("\nHidden_units: {}".format(self.hidden_units), "\n")
        print("Optimizer: {}".format(self.optimizer), "\n")
        print("Activation_function: {}".format(self.activation_fn), "\n")
        print("Learning_Rate: {}".format(self.learning_rate), "\n")
        print("Training_Steps: {}".format(self.steps), "\n")
        print("Batch_Size: {}".format(self.batch_size), "\n")
        print("Number_of_epochs: {}".format(self.nr_epochs), "\n")
        print("Feature_Scaling: {}".format(self.feature_scl), "\n")

    update_parameters = property(_get_attributes, _set_attributes)

    @classmethod
    def train_input_func(cls, features, labels, batch_size, nr_epochs):
        """
        An input function for training

        References:
            https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py

        :param features:
        :param labels:
        :param batch_size:
        :param nr_epochs:
        :return:
        """

        return tf.estimator.inputs.pandas_input_fn(x=features, y=labels, batch_size=batch_size,
                                                   num_epochs=nr_epochs, shuffle=True)
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

        References:
            https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn

        :param features:
        :param label:
        :param batch_size:
        :param epochs:
        :return:
        """
        return tf.estimator.inputs.pandas_input_fn(x=features, y=label, batch_size=batch_size,
                                                   num_epochs=epochs, shuffle=False)

    @classmethod
    def pred_input_fn(cls, test_features, batch_size):
        """
        An input function for prediction

        References:
            https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn

        :param test_features: from test_data (x_true)
        :param batch_size:
        :return:
        """
        """
        
        :return:
        """
        return tf.estimator.inputs.pandas_input_fn(x=test_features, batch_size=batch_size, shuffle=False)

    def setup_feature_columns(self, x_train):
        """
        Setup feature columns into TensorFlow format (numeric, bucketized, hash_bucket)

        References:
            https://www.tensorflow.org/get_started/feature_columns

        :param self:
        :param x_train: in Pandas DataFrame (containing all feature_columns)
        :return: feature_columns
        """

        # setup feature columns
        non_obj_feature, obj_feature = StartMod.feature_columns(x_train)

        # setup continuous features
        feature_columns = [tf.feature_column.numeric_column(key=fea) for fea in non_obj_feature]

        # setup categorical features (categorical_column_with_vocabulary_list) or (categorical_column_with_hash_bucket)
        if obj_feature:
            len_unique = len(x_train[self.label].unique())
            cat_column = tf.feature_column.categorical_column_with_hash_bucket(self.label,
                                                                               hash_bucket_size=len_unique)
            feature_columns.append(cat_column)

        # setup the bucketized_column
        # tbd: find boundaries (number of bars in histogram) to setup the bucketized_column
        # if label_bucket is not None:
        # bucketized_columns = [tf.feature_column.bucketized_column(fea, boundaries=[0, 5, 10, 15, 20, 25, 30])
        #                       for fea in numerics_columns if fea[0] == label_bucket]

        return feature_columns

    @classmethod
    def regressor_custom(cls, data):
        """
        tbd

        References:
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
        :param data: pandas.core.frame.DataFrame
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

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        # retrieve value of feature_columns

        # create model_obj

        # evaluation

        # predictions

        pass

    def classifier_estimator(self, model_lin=True):
        """
        Apply pre-made Estimator to classify data

        References:
            https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearClassifier
            https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier

        :param data: pandas.core.frame.DataFrame
        :param model_lin: default is LinearClassifier
        :return: LinearClassifier object (DNNClassifier object), y_true, y_predict
        """
        if self.label is None:
            print("Please choose other method as Clustering to process data\n")
            return

        # print(self.data.columns)
        # fea_cols = [tf.feature_column.numeric_column(key=fea) for fea in self.data.columns]

        x_train, x_true, y_train, y_true = StartMod.split_data(self.data, self.label)
        if self.feature_scl:
            x_train = StartMod.feature_scaling(x_train, type_pd=True)
            x_true = StartMod.feature_scaling(x_true, type_pd=True)

        # create feature_columns for model
        fea_cols = self.setup_feature_columns(x_train)

        if model_lin:
            classifier = tf.estimator.LinearClassifier(feature_columns=fea_cols, n_classes=self.n_classes,
                                                       optimizer='Ftrl')
        else:
            # Tbd with Dense Columns
            classifier = tf.estimator.DNNClassifier(feature_columns=fea_cols, hidden_units=self.hidden_units,
                                                    n_classes=self.n_classes, optimizer=self.optimizer,
                                                    activation_fn=self.activation_fn)

        # Train the Model
        train_input_func = StartModTF.train_input_func(features=x_train, labels=y_train,
                                                       batch_size=self.batch_size, nr_epochs=self.nr_epochs)
        classifier.train(input_fn=train_input_func, steps=self.steps)

        # Evaluation
        eval_input_func = StartModTF.eval_input_fn(features=x_train, label=y_train,
                                                   batch_size=self.batch_size, epochs=self.nr_epochs)
        result_eval = classifier.evaluate(input_fn=eval_input_func)
        print("\n", result_eval)

        # Prediction
        pred_input_func = StartModTF.pred_input_fn(test_features=x_true, batch_size=self.batch_size)
        y_predict = classifier.predict(input_fn=pred_input_func)

        # show metrics report
        # final_pred = [pred['class_ids'][0] for pred in list(result_predict)]
        # StartMod.metrics_report(self.y_test.values, final_pred)
        # print(type(self.y_test))
        # TODO: convert y_predict into numeric_values
        self.data['predicted'] = y_predict

        return classifier, y_true, y_predict

    def keras_sequential(self, data, hidden_layers=1, output_signals=1):
        """
        Setup Keras and run the Sequential method to predict value

        References:
            https://keras.io/getting-started/sequential-model-guide/

        :param data:
        :param dependent_label: sequential-object model, predicted value, actual (true) value
        :param hidden_layers: number of hidden layers (default 1)
        :param output_signals: default 1 (when there's only 1 categorical column)
        :return: Keras-Sequential object, the actual (true) value, the predicted value
        """
        # split data
        x_train, x_eval, y_train, y_eval = StartMod.split_data(data, self.label)

        # Initialising the ANN
        model = Sequential()

        # tbd: use parameter tuning to find the exact number of nodes in the hidden-layer
        # default = number of features
        input_signals = x_train.shape[1]

        # Adding the input layer and the first hidden layer, activation function as rectifier function
        model.add(Dense(activation="relu", input_dim=x_train.shape[1], units=input_signals,
                        kernel_initializer="uniform"))

        # Adding the second hidden layer, activation function as rectifier function
        for _ in range(hidden_layers):
            model.add(Dense(activation="relu", units=input_signals, kernel_initializer="uniform"))

        # Adding the output layer (in case of there's only one dependent_label), activation function as sigmoid function
        model.add(Dense(activation="sigmoid", units=output_signals, kernel_initializer="uniform"))

        # Compiling the ANN with optimizer='adam'
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit the keras_model to the training_data and see the real time training of model on data with result of loss
        # and accuracy. The smaller batch_size and higher epochs, the better the result. However, slow_computing!
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nr_epochs)

        # Predictions and evaluating the model
        y_pred = model.predict(x_eval)

        # Evaluate the model
        scores = model.evaluate(x_train, y_train)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        return model, y_eval, y_pred

    def keras_cnn_1d(self, momentum=0.2, seed=10, dropout_rate=0.2, n_filters=32, kernel_size=1, padding="same"):
        """
        Setup Keras hyper_parameters, Kernel_initializer 's parameter and find regression_value for (multiple) label(s)

        :param momentum: SGD's parameters
        :param seed: Setup hyperparameters (default is 10)
        :param dropout_rate: Regularization technique to prevent Neural Networks from Overfitting
        :param n_filters: number of filters (default is 32)
        :param kernel_size:
        :param padding: Output dim has the same size as Input dim (default is 'same')
        :return:

        Reference:
            https://keras.io/optimizers/
            https://keras.io/initializers/
            https://keras.io/layers/core/#dropout

        :return:
        """
        # Convert to the right dimension row, column, 1-channel in Numpy array
        X_data = self.data.drop(self.label, axis=1)
        Y_data = self.data[self.label]

        # Normalizing data
        scaler = MinMaxScaler()
        scaler.fit(X_data)
        X_data = pd.DataFrame(data=scaler.transform(X_data), columns=X_data.columns, index=X_data.index)

        # Split data into training_data and evaluation_data
        x_train, x_eval, y_train, y_eval = train_test_split(X_data, Y_data, test_size=0.2, random_state=101)

        # Number of feature columns in training_data
        input_dimension = len(x_train.columns)
        x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
        y_train = y_train.as_matrix()

        # Init hyper parameter
        np.random.seed(seed)
        hidden_initializer = random_uniform(seed=seed)

        # Number of layers and neurons in every layers (in numpy array e.g. [1000, 500, 250, 3])
        hidden_units = self.hidden_units

        # Create model and add hidden layers
        model = Sequential()

        # Conv1D needs 2 dimension -> input_shape=(1000,1), similarly Conv2D needs 3 dim, Conv3D needs 4 dim
        model.add(Conv1D(input_shape=(input_dimension, 1), activation=self.activation_fn, filters=n_filters,
                         kernel_size=kernel_size, padding=padding))
        model.add(Conv1D(activation=self.activation_fn, filters=n_filters, kernel_size=kernel_size))

        # Flattens the input
        model.add(Flatten())

        # Applies Dropout to the input
        model.add(Dropout(dropout_rate))

        # Use output of CNN as input of ANN, e.g. units = np.array([1000, 500, 250, 3])
        # print(hidden_units[0])
        model.add(Dense(units=hidden_units[0], input_dim=input_dimension, kernel_initializer=hidden_initializer,
                        activation=self.activation_fn))
        if len(hidden_units[1:-1])>0:
            for i, v in enumerate(hidden_units[1:-1]):
                # print(i, v)
                model.add(Dense(units=v, kernel_initializer=hidden_initializer, activation=self.activation_fn))
        # print(hidden_units[-1:][0])
        model.add(Dense(units=hidden_units[-1:][0], kernel_initializer=hidden_initializer))

        # reset optimizer
        sgd = SGD(lr=self.learning_rate, momentum=momentum)
        self.optimizer = sgd

        # compile and train model with training_data
        model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=self.nr_epochs, batch_size=self.batch_size)

        # Evaluate the model
        scores = model.evaluate(x_train, y_train)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # Predict value
        y_pred = model.predict(x_eval.values.reshape(x_eval.shape[0], x_eval.shape[1], 1))
        y_pred = pd.DataFrame(data=y_pred, columns=y_eval.columns)

        return model, y_eval, y_pred

    def keras_rnn_lstm(self):
        """
        Time Series model
        Setup Keras and find regression_value for (multiple) label(s)

        Reference:


        :return:
        """
        pass

    @classmethod
    def regularization(cls):
    	"""
        e.g. Dropout to prevent Neural Networks from Overfitting
            Grid_Search to tune the hyper_parameter
        """
    	pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModTF": StartModTF.__name__,
            "StartModTF.(data)": StartModTF.regressor_custom.__doc__
            }

        return info


smtf = StartModTF(train_data)
smtf.info_help()
# update parameters
# new_param={'hidden_units':[10,10,10], 'optimizer':'Adam', 'activation_fn':'sigmoid', 'learning_rate': 0.0001,
#            'steps':2000, 'batch_size':10, 'num_epochs':10, 'feature_scl':True}
# smtf.update_parameters=new_param
