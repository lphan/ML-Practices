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
# from tensorflow.contrib.keras import models, layers, optimizers, initializers
# from tensorflow.contrib.layers import fully_connected
from math import sqrt
from Starts.startmod import *
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Dropout, LSTM
from keras.optimizers import SGD
from keras.initializers import random_uniform


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

    def __init__(self, n_classes, dependent_label):
        """
        # Description: init parameters for neuron network

        :param dependent_label: target_feature

        References:
            https://keras.io/losses/
            https://keras.io/optimizers/

        """
        super().__init__(n_classes, dependent_label)   # StartMod.__init__(self, n_classes, dependent_label)
        self.input_units = 1
        self.hidden_units = [10, 10]        # default setup 10 neuron in 2 hidden layers
        self.output_units = 1

        self.optimizer = "Adagrad"          # default Adagrad optimizer
        self.activation_fn = "relu"         # default 'relu' function
        self.learning_rate = 0.001          # default 0.001 -> 0.01 -> 0.1
        self.steps = 1000                   # default training_steps 1000
        self.loss = 'mean_squared_error'    # option: binary_crossentropy, categorical_crossentropy, mean_absolute_error
        self.drop_out_rate = 0.2
        self.rec_drop_out = 0.2

        # (correspond with available system memory capacity to avoid out_of_memory_error,
        # small for many features, big for performance)
        self.batch_size = 10                # default batch_size 10

        self.nr_epochs = 1                  # default number of epochs 1 (for large dataset) and 10 (for small dataset)
        self.feature_scl = False            # default turn off feature scaling

        self.bias_initializer = 'random_uniform'
        self.depth_wise_initializer = 'random_uniform'
        self.seed = 10

        # CNN parameters
        self.kernel_size = 1
        self.filter_size = [3, 3]
        self.n_filters = 1
        self.n_padding = 1
        self.n_strides = 1
        self.momentum = 0.2  # SGD's parameters

        # others hyper parameters:
        #   mini batchsize,
        #
        # Reducing Overfitting:
        #   regularization for neural network: L1, L2
        #   dropout regularization: shrink weights
        #   other regularization: data augmentation, early stopping

        # choose NN architectures: RNN, CNN, others

    def _get_attributes(self):
        """
        # Description: get method to retrieve dict_parameters of attributes

        :return:
        """
        nn_attributes = {'input_units': self.input_units, 'hidden_units': self.hidden_units,
                         'output_units': self.output_units, 'optimizer': self.optimizer,
                         'activation_fn':self.activation_fn, 'learning_rate': self.learning_rate,
                         'steps': self.steps, 'batch_size': self.batch_size, 'num_epochs': self.nr_epochs,
                         'feature_scl': self.feature_scl, 'loss_fn': self.loss,
                         'drop_out': self.drop_out_rate, 'rec_drop_out': self.rec_drop_out,
                         'bias_initializer': self.bias_initializer,
                         'depth_wise_initializer': self.depth_wise_initializer, 'seed': self.seed,
                         'kernel_size': self.kernel_size, 'filter_size':self.n_filters, 'n_filters': self.n_filters,
                         'n_padding': self.n_padding, 'n_strides': self.n_strides, 'momentum': self.momentum
                         }
        return nn_attributes

    # reset all attributes in neural network
    def _set_attributes(self, dict_params):
        """
        # Reference:
            https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/

        :param dict_params:
        :return:
        """
        # Update Neuron Network Parameters
        self.input_units = dict_params['input_units']
        self.hidden_units = dict_params['hidden_units']
        self.output_units = dict_params['output_units']
        self.optimizer = dict_params['optimizer']
        self.activation_fn = dict_params['activation_fn']
        self.learning_rate = dict_params['learning_rate']
        self.steps = dict_params['steps']
        self.batch_size = dict_params['batch_size']
        self.nr_epochs = dict_params['num_epochs']
        self.feature_scl = dict_params['feature_scl']
        self.loss = dict_params['loss_fn']
        self.drop_out_rate = dict_params['drop_out']
        self.rec_drop_out = dict_params['rec_drop_out']
        self.bias_initializer = dict_params['bias_initializer']
        self.depth_wise_initializer = dict_params['depth_wise_initializer']
        self.seed = dict_params['seed']

        # update CNN Hype Parameters
        self.kernel_size = dict_params['kernel_size']
        self.filter_size = dict_params['filter_size']
        self.n_filters = dict_params['n_filters']
        self.n_padding = dict_params['n_padding']
        self.n_strides = dict_params['n_strides']
        self.momentum = dict_params['momentum']

    def info_parameters(self, CNN=True):
        print("\nInput_units: {}".format(self.input_units), "\n")
        print("Hidden_units: {}".format(self.hidden_units), "\n")
        print("Output_units: {}".format(self.output_units), "\n")
        print("Optimizer: {}".format(self.optimizer), "\n")
        print("Activation_function: {}".format(self.activation_fn), "\n")
        print("Learning_Rate: {}".format(self.learning_rate), "\n")
        print("Training_Steps: {}".format(self.steps), "\n")
        print("Batch_Size: {}".format(self.batch_size), "\n")
        print("Number_of_epochs: {}".format(self.nr_epochs), "\n")
        print("Feature_Scaling: {}".format(self.feature_scl), "\n")
        print("Loss_function: {}".format(self.loss), "\n")
        print("Drop_out: {}".format(self.drop_out_rate), "\n")
        print("Recurrent_drop_out: {}".format(self.rec_drop_out), "\n")
        print("Bias_Initializer: {}".format(self.bias_initializer), "\n")
        print("Depth_wise_Initializer: {}".format(self.depth_wise_initializer), "\n")
        print("Seed: {}".format(self.seed), "\n")

        if CNN:
            print("Kernel_size: {}".format(self.kernel_size), "\n")
            print("Filter_size: {}".format(self.filter_size), "\n")
            print("n_filters: {}".format(self.n_filters), "\n")
            print("n_padding: {}".format(self.n_padding), "\n")
            print("n_strides: {}".format(self.n_strides), "\n")
            print("momentum: {}".format(self.momentum), "\n")

    update_parameters = property(_get_attributes, _set_attributes)

    @classmethod
    def train_input_func(cls, features, dependent_label, batch_size, nr_epochs):
        """
        # Description: input function for training

        # References:
            https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py

        :param features:
        :param dependent_label:
        :param batch_size:
        :param nr_epochs:
        :return:
        """
        # Alternatives:
        # tensors = (dict(features), labels)
        #
        # dataset = tf.data.Dataset.from_tensor_slices(tensors)
        #
        # # Shuffle, repeat, and batch the examples.
        # return dataset.shuffle(1000).repeat().batch(batch_size)
        return tf.estimator.inputs.pandas_input_fn(x=features, y=dependent_label, batch_size=batch_size,
                                                   num_epochs=nr_epochs, shuffle=True)

    @classmethod
    def eval_input_fn(cls, features, label, batch_size, epochs):
        """
        # Description: input function for evaluation

        # References:
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
        # Description: setup feature columns into TensorFlow format (numeric, bucketized, hash_bucket)

        # References:
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
            len_unique = len(x_train[self.dependent_label].unique())
            cat_column = tf.feature_column.categorical_column_with_hash_bucket(self.dependent_label,
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
        # Description: apply Estimator API

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

    def classifier_estimator(self, data, dependent_label, model_lin=True):
        """
        # Description: apply pre-made Estimator to classify data

        # References:
            https://www.tensorflow.org/api_docs/python/tf/contrib/learn/LinearClassifier
            https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier

        :param data: pandas.core.frame.DataFrame
        :param dependent_label:
        :param model_lin: default is LinearClassifier
        :return: LinearClassifier object (DNNClassifier object), y_true, y_predict
        """
        if dependent_label is None:
            print("Please choose other method as Clustering to process data\n")
            return

        # print(self.data.columns)
        # fea_cols = [tf.feature_column.numeric_column(key=fea) for fea in self.data.columns]

        x_train, x_true, y_train, y_true = self.split_data(data, dependent_label)
        if self.feature_scl:
            x_train = StartMod.feature_scaling(x_train, type_pd=True)
            x_true = StartMod.feature_scaling(x_true, type_pd=True)

        # create feature_columns for model
        fea_cols = self.setup_feature_columns(x_train)

        if model_lin:
            print(self.n_classes)
            classifier = tf.estimator.LinearClassifier(feature_columns=fea_cols, n_classes=self.n_classes,
                                                       optimizer='Ftrl')
        else:
            # Tbd with Dense Columns
            classifier = tf.estimator.DNNClassifier(feature_columns=fea_cols, hidden_units=self.hidden_units,
                                                    n_classes=self.n_classes, optimizer=self.optimizer,
                                                    activation_fn=self.activation_fn)

        # Train the Model
        train_input_func = StartModTF.train_input_func(features=x_train, dependent_label=y_train,
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
        # convert y_predict into numeric_values, Convert the predicted value y_pred and show the metrics_report
        final_pred = [pred['class_ids'][0] for pred in list(y_predict)]

        # self.data['predicted'] = y_predict
        return classifier, y_true, final_pred

    @classmethod
    def regularization_nn(cls):
        """
        e.g. Dropout to prevent Neural Networks from Overfitting
            Grid_Search to tune the hyper_parameter
        """
        pass

    @classmethod
    def auto_encoder(cls):
        """
        The number of input neurons = the number of output neurons (symmetric network)
        Input size = output size
        Encoder <-> Decoder
        The weight at the input layer = The weight at the output layer
        :return:
        """
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModTF": StartModTF.__name__,
            "StartModTF.(data)": StartModTF.regressor_custom.__doc__
            }

        return info


class StartModTFANN(StartModTF):

    def __init__(self, n_classes, dependent_label):
        super().__init__(n_classes, dependent_label)  # StartModTF.__init__(self, n_classes, dependent_label)

    def keras_sequential(self, data, output_signals=1):
        """
        # Description: setup Keras and run the Sequential method to predict value

        # References:
            https://keras.io/getting-started/sequential-model-guide/
            https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc#

        :param data: pandas.core.frame.DataFrame
        :param output_signals: default 1 (when there's only 1 categorical column)
        :return: Keras-Sequential object, the actual (true) value, the predicted value
        """
        # split data
        x_train, x_eval, y_train, y_eval = StartMod.split_data(data, dependent_label=self.dependent_label)

        # Initialising the ANN
        model = Sequential() # model = models.Sequential()

        # tbd: use parameter tuning to find the exact number of nodes in the hidden-layer
        # Number of layers and neurons in every layers (in numpy array e.g. [1000, 500, 250, 3])
        hidden_units = self.hidden_units

        # Init hyper parameter
        # np.random.seed(10)
        hidden_initializer = random_uniform(seed=self.seed)  # initializers.RandomUniform(seed=self.seed)
        input_dimension = self.input_units

        # default = number of features, input_signals = x_train.shape[1] = hidden_units[0]
        # Adding the input layer and the first hidden layer, activation function as rectifier function
        model.add(Dense(units=self.input_units, input_dim=input_dimension, kernel_initializer=hidden_initializer,
                        activation=self.activation_fn, bias_initializer=self.bias_initializer))

        # Adding the second hidden layer, activation function as rectifier function
        # print(hidden_units)
        for i in range(len(hidden_units)):
            # print(i, hidden_units[i])
            model.add(Dense(activation=self.activation_fn, units=hidden_units[i], kernel_initializer=hidden_initializer))

        # Adding the output layer (in case of there's only one dependent_label),
        # n_classes = 2 (binary), then activation function is chosen as sigmoid function
        # n_classes > 2 (not binary), then activation function is chosen as softmax function
        if self.n_classes == 2:
            output_activation = "sigmoid"
        else:
            output_activation = "softmax"

        model.add(Dense(units=self.output_units, kernel_initializer=hidden_initializer, activation=output_activation))

        # Compiling the ANN with optimizer='adam'
        model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])

        # Fit the keras_model to the training_data and see the real time training of model on data with result of loss
        # and accuracy. The smaller batch_size and higher epochs, the better the result. However, slow_computing!
        print(x_train.shape, y_train.shape)
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nr_epochs)

        # Predictions and evaluating the model
        y_pred = model.predict(x_eval)

        # Evaluate the model
        scores, accuracy = model.evaluate(x_train, y_train)
        print("\nModel %s: Scores: %.2f%%, Accuracy: %.2f%%" % (model, scores*100, accuracy*100))

        return model, y_eval, y_pred


class StartModTFCNN(StartModTF):
    def __init__(self, n_classes, dependent_label):
        super().__init__(n_classes, dependent_label)  # StartModTF.__init__(self, n_classes, dependent_label)

        # Types of layer:
        # Convolution (conv), Pooling (pool), Fully connected (fc)
        # Max Pooling, Average Pooling
        # E.g.:
        #   LeNet-5, AlexNet, VGG
        #   ResNet (Residual Network)
        #   Inception Network

    def keras_cnn_1d(self, data):
        """
        # Description:
            setup Keras hyper_parameters, Kernel_initializer 's parameter and find regression_value
            for (multiple) label(s)

        :param data: pandas.core.frame.DataFrame
        :return:

        # Reference:
            https://keras.io/optimizers/
            https://keras.io/initializers/
            https://keras.io/layers/core/#dropout

        :return:
        """
        # Convert to the right dimension row, column, 1-channel in Numpy array
        X_data = data.drop(self.dependent_label, axis=1)
        Y_data = data[self.dependent_label]

        # Normalizing data
        scaler = MinMaxScaler()
        scaler.fit(X_data)
        X_data = pd.DataFrame(data=scaler.transform(X_data), columns=X_data.columns, index=X_data.index)

        # Split data into training_data and evaluation_data
        x_train, x_eval, y_train, y_eval = train_test_split(X_data, Y_data, test_size=0.2, random_state=101)

        # Number of feature columns in training_data
        input_dimension = len(x_train.columns)  # input_dimension = self.input_units
        x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
        y_train = y_train.as_matrix()

        # Init hyper parameter
        np.random.seed(self.seed)
        hidden_initializer = random_uniform(seed=self.seed)
        # self.n_filters = 32  # reset number of filters (default is 32)

        # Create model and add hidden layers
        model = Sequential()

        # Conv1D needs 2 dimension -> input_shape=(input_dimension, 1), similarly Conv2D needs 3 dim, Conv3D needs 4 dim
        # Output dim has the same size as Input dim(default is 'same')
        model.add(Conv1D(input_shape=(input_dimension, 1), activation=self.activation_fn, filters=self.n_filters,
                         kernel_size=self.kernel_size, padding='same'))
        model.add(Conv1D(activation=self.activation_fn, filters=self.n_filters, kernel_size=self.kernel_size))

        # Flattens the input
        model.add(Flatten())

        # Regularization technique to prevent Neural Networks from Overfitting
        model.add(Dropout(rate=self.drop_out_rate))

        # Use output of CNN as input of ANN, e.g. units = np.array([1000, 500, 250, 3])
        model.add(Dense(units=self.input_units, input_dim=input_dimension, kernel_initializer=hidden_initializer,
                        activation=self.activation_fn))

        # Hidden layers, number of layers and neurons in every layers (in numpy array e.g. [1000, 500, 250, 3])
        for unit in self.hidden_units:
            model.add(Dense(units=unit, kernel_initializer=hidden_initializer, activation=self.activation_fn))

        # Output signal = output_units
        # output_activation_fn = 'softmax', no need activation in this case?
        model.add(Dense(units=self.output_units, kernel_initializer=hidden_initializer))

        # reset optimizer
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum)
        self.optimizer = sgd

        # compile and train model with training_data
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=self.nr_epochs, batch_size=self.batch_size)

        # Evaluate the model
        # scores, accuracy = model.evaluate(x_train, y_train)
        # print("\nModel %s: Scores: %.2f%%, Accuracy: %.2f%%" % (model.metrics_names[1], scores[1]*100, accuracy*100))

        # Evaluate the model
        scores = model.evaluate(x_train, y_train)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # Predict value
        y_pred = pd.DataFrame(data=model.predict(x_eval.values.reshape(x_eval.shape[0], x_eval.shape[1], 1)),
                              columns=y_eval.columns)

        return model, y_eval, y_pred


class StartModTFRNN(StartModTF):

    def __init__(self, n_classes, dependent_label):
        super().__init__(n_classes, dependent_label)  # StartModTF.__init__(self, n_classes, dependent_label)

    def keras_rnn_lstm_onestep_univ(self, data, repeats=10):
        """
        # Description:
            build recurrent neural network RNN using Long-Short Term Memory LSTM for a one-step univariate time series
            forecasting problem

        # Reference:
            https://keras.io/layers/recurrent/#lstm
            https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

        :param data:
        :param repeats:
        :return:
        """

        # frame a sequence as a supervised learning problem
        def timeseries_to_supervised(dat, lag=1):
            df = pd.Series(dat)
            columns = pd.Series(data=[df.shift(i) for i in range(1, lag + 1)])
            new_df = pd.concat([columns[0], df], axis=1)
            new_df.fillna(0, inplace=True)
            return new_df

        # create a different series
        def difference(dat, interval=1):
            diff = list()
            for i in range(interval, len(dat)):
                value = dat[i] - dat[i - interval]
                diff.append(value)
            return pd.Series(diff)

        # invert difference value
        def inverse_difference(history, yhat, interval=1):
            return yhat + history[-interval]

        # inverse scaling for a forecasted value
        def invert_scale(scaler, X, value):
            new_row = [x for x in X] + [value]
            array = np.array(new_row)
            array = array.reshape(1, len(array))
            inverted = scaler.inverse_transform(array)
            return inverted[0, -1]

        # fit an LSTM network to training data
        def fit_lstm(train):
            """
            The LSTM layer expects input to be in a matrix
            :param train:
            :return:
            """
            X, y = train[:, 0:-1], train[:, -1]
            # reshape X to the 3D format (Samples, TimeSteps, Features) for LSTM, so keep it simple as
            # one separate sample, with one timestep and one feature.
            X = X.reshape(X.shape[0], 1, X.shape[1])

            # init Keras Sequential model with 1 hidden layer with x_given neurons and 1 output
            # batch_input_shape = tuple that specifies the expected number of observations to read each batch,
            # the number of time steps, and the number of features.
            model = Sequential()
            model.add(LSTM(self.hidden_units[0], batch_input_shape=(self.batch_size, X.shape[1], X.shape[2]),
                           stateful=True, dropout=self.drop_out_rate, recurrent_dropout=self.rec_drop_out))
            model.add(Dense(self.hidden_units[1]))  # 1 for 1 output

            # set loss='mean_squared_error', optimizer='adam'
            model.compile(loss=self.loss, optimizer=self.optimizer)
            for i in range(self.nr_epochs):
                model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=False)
                model.reset_states()
            return model

        # make a one-step forecast
        def forecast_lstm(model, X):
            X = X.reshape(1, 1, len(X))
            yhat = model.predict(X, batch_size=self.batch_size)
            return yhat[0, 0]

        # transform data to be stationary
        raw_values = data[self.dependent_label].values
        diff_values = difference(raw_values)

        # transform data to be supervised learning
        supervised = timeseries_to_supervised(diff_values, 1)
        supervised_values = supervised.values

        # transform the scale of the data
        scaler, supervised_values_scaled = StartMod.feature_scaling(supervised_values, feature_range=(-1, 1),
                                                                    type_pd=False)

        # split data into train and test set
        train_scaled, test_scaled = StartMod.split_data(data=supervised_values_scaled, type_pd=False, test_size=0.1)

        # repeat experiment from parameter repeat
        error_scores = list()
        for r in range(repeats):
            # fit the model with different parameters fit_lstm(train_scaled, 3000, 4), or (train_scaled, 1500, 1)
            lstm_model = fit_lstm(train_scaled)

            # forecast the entire training dataset to build up state for forecasting ??
            # ->  train_scaled[:, 0] is the original value in first column
            # ->  train_scaled[:, 1] is the shifted one-time step value
            train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
            lstm_model.predict(train_reshaped, batch_size=self.batch_size)

            # walk-forward validation on the test data by predicting on test_scaled to measure the model's performance
            y_pred = list()

            for i in range(len(test_scaled)):
                # make one-step forecast as supervised learning
                # y = test_scaled[i, 1] is the shifted one-time step values
                # X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
                # convert X into numpy array to get its shape and its length
                X, y = np.array([test_scaled[i, 0]]), test_scaled[i, 1]
                X = X.reshape(1, 1, len(X))
                yhat = lstm_model.predict(X, batch_size=self.batch_size)
                yhat = yhat[0, 0]

                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # print("------ yhat invert_scale", yhat)

                # invert differencing
                yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
                # print("------ yhat invert_difference", yhat)

                # store forecast
                y_pred.append(yhat)
                expected = raw_values[len(train_scaled) + i + 1]
                print('At time point=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

            # report performance by measuring RMSE from truth value in test_scaled and predicted value for every repeat
            y_eval = raw_values[-len(test_scaled):]
            rmse = sqrt(mean_squared_error(y_eval, y_pred))
            print('%d) Test RMSE: %.3f' % (r + 1, rmse), '\n')
            # print(len(raw_values[len(train_scaled)+1:]), len(predictions), len(raw_values[-len(test_scaled):]))
            error_scores.append(rmse)

        # summarize results
        results = pd.DataFrame()
        results['rmse'] = error_scores
        print(results.describe())
        results.boxplot()

        return lstm_model, raw_values[-len(test_scaled):], y_pred

    def keras_rnn_lstm_onestep_multiv(self, data, dependent_label):
        """
        # Description:
            build recurrent neural network RNN using Long-Short Term Memory LSTM for a one-step multivariate time series
            forecasting problem

        # Reference:
            https://keras.io/layers/recurrent/#lstm
            https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        :return:
        """
        pass

    def keras_rnn_lstm_multisteps(self, data, dependent_label):
        """
        # Description:
            build recurrent neural network RNN using Long-Short Term Memory LSTM for a multi-step time series
            forecasting problem

        # Reference:
            https://keras.io/layers/recurrent/#lstm
            https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        :return:
        """
        pass

    def exp_weighted_avg(self, data, dependent_label):
        """
        # Description:
            compute exponential moving average of group of certain time series values
            (e.g. in 5 days, 10 days, 1 week, 1 month, 1 year, etc.)

        :return:
        """
        pass

# update parameters
# new_param={'input_units': 1000, 'hidden_units': [500, 250], 'output_units': 3, 'optimizer':'Adam',
#            'activation_fn': 'relu', 'learning_rate': 0.0025,
#            'steps': 5000, 'batch_size': 10, 'num_epochs': 100, 'feature_scl': True,
#            'loss_fn': 'binary_crossentropy', 'drop_out': 0.5, 'rec_drop_out': 0.6,
#            'bias_initializer': 'random_uniform', 'depth_wise_initializer': 'random_uniform', 'seed': 10,
#            'kernel_size': 1, 'filter_size': 1, 'momentum': 0.2, 'n_filters': 32, 'n_padding': 1, 'n_strides': 1}
# StartModtf smtf
# smtf.update_parameters=new_param
