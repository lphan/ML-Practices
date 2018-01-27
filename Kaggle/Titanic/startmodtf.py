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

from keras.models import Sequential
from keras.layers import Dense
from startmod import *
from startmodskl import StartModSKL


class StartModTF(StartMod):
    """
      Description: StartModTF - Start Models Tensorflow
        regression, classification

      Start:
          jupyter notebook
          -> from startmodtf import *
          -> info_modtf
    """

    def __init__(self):
        pass

    @classmethod
    def keras_sequential(cls, data, dependent_label):
        """
        Setup Keras and run the Sequential method to predict value
        :param data:
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
    def regression(cls, data):
        """
        tbd
        :param data:
        :return:
        """
        pass

    @classmethod
    def classification(cls, data):
        """
        tbd
        :param data:
        :return:
        """
        pass

    @staticmethod
    def info_help():
        info = {
            "info_help_StartModTF": StartModTF.__name__,
            "StartMod.(data)": StartModTF.regression_input.__doc__
            }
        # info.update(StartML.info_help())

        return info


info_modtf = StartMod.info_help()
