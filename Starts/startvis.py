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
# import pyqtgraph
# import graphviz
# import seaborn
# import plotly
# import plotnine
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
from pd.plotting import scatter_matrix
from matplotlib.pylab import rcParams
from Starts.startml import *
rcParams['figure.figsize'] = 20, 6


class StartVis(StartML):
    """
        Description: StartVis - Start Visualization
        Visualize data in different chart (Bar-Chart, Histograms, Scatter, TimeSeries, etc.)
        Reference: https://www.kaggle.com/learn/data-visualisation

        Start:
          jupyter notebook
          -> from startvis import *
          -> info_vis
    """

    @classmethod
    def vis_bar(cls, data, columns, x_label='', y_label='', title='', rot=0, bar=True):
        """
        visualize the number of counted values in the given columns in bar-chart

        :param data: pandas.core.frame.DataFrame
        :param columns:
        :param x_label:
        :param y_label:
        :param title:
        :param rot: rotation
        :param bar: display bar char (default True)
        :return:
        """
        if bar:
            for column in columns:
                # data[column].head().value_counts().plot(kind='bar')
                # TBD: compute %value on number of index
                (data[column].value_counts()/ len(data[column])).sort_index().plot(kind='bar')
            plt.title("Bar Chart " + str(column) + " " + title)
        else:
            for column in columns:
                (data[column].value_counts() / len(data[column])).sort_index().plot.line()
            plt.title("Line Chart " + str(column) + " " + title)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(rotation=rot)
        # plt.legend()
        plt.show()

    @classmethod
    def vis_bar_groupby(cls, data, columns, gb_columns, x_label='', y_label='', title='', rot=0):
        """
        Visualize groupby-object in bar chart.

        :param data: pandas.core.frame.DataFrame
        :param columns:
        :param gb_columns: groupby columns
        :param x_label:
        :param y_label:
        :param title:
        :param rot: rotation (default = 0)
        :return:
        """

        grouped_data = StartML.groupby_columns(data, columns, gb_columns)  # dict-type

        # compute left and height for bar-chart
        le = np.arange(len(grouped_data.keys()))
        he = [len(grouped_data[k]) for k in grouped_data.keys()]

        print("Values:", he)
        plt.bar(left=le, height=he, width=0.5)
        plt.xticks(le, (grouped_data.keys()), rotation=rot)
        plt.title("Bar Chart group_by " + str(columns) + " " + title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    @classmethod
    def vis_hist(cls, data, columns, x_label='', y_label='', title='', func_filter=None, rot=0):
        """
        Description: Display Histogram of data and labels, with filter-function

        :param data: pandas.core.frame.DataFrame
        :param columns: all column name
        :param func_filter: object type Pandas-Series
        :param x_label: label x_axis
        :param y_label: label y_axis
        :param title: chart title
        :param rot: rotation (default = 0)

        :return: Histogram
        """

        try:
            if func_filter:
                for column in columns:
                    data[func_filter][column].plot.hist()
            else: 
                data.hist()

            """ for column in columns:
                if func_filter is None:
                    data[column].plot.hist()
                else:
                    data[func_filter][column].plot.hist() """

            plt.title("Histogram " + str(column) + " " + title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=rot)
            plt.legend()
            plt.show()
        except TypeError:
            print("No numeric data to plot")
            return

    @classmethod
    def vis_density(cls, data):
        """
        Description: getting a idea of the distribution for each attribute/ column
        :param data: pandas.core.frame.DataFrame
        """
        data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
        plt.show()

    @classmethod
    def vis_boxplot(cls, data):
        """
        visual boxplot all data features to detect the possible outliers in every features

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)

        data_to_plot = [data[col].values for col in data.columns]

        ax.boxplot(data_to_plot)
        plt.title("Visual Boxplot")
        plt.show()

    @classmethod
    def vis_subplot(cls, data):
        """
        show subplot of all features columns in data
        :param data: pandas.core.frame.DataFrame
        :return:
        """
        idx = 1
        plt.figure(figsize=(15, 10))
        for col in data.columns:
            plt.subplot(len(data.columns), 1, idx)
            plt.plot(data[col])  # plt.plot(nonan_data[col].values)
            plt.title(col, y=0.2, loc='right')
            idx += 1
        plt.show()

    @classmethod
    def vis_scatter(cls, data, features):
        plt.figure()
        x_data = data[features[0]]
        y_data = data[features[1]]
        label_data = data[features[2]]
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.scatter(x_data, y_data, c=label_data)

    @classmethod
    def vis_obj_predict(cls, x_index, y_true, y_pred, title):
        """
        Visualizing result of the predicting object

        :param x_index:
        :param y_true:
        :param y_pred:
        :param title:
        :return:
        """
        plt.figure()
        plt.bar(x_index, y_true, align='center', alpha=1)
        plt.bar(x_index, y_pred, align='center', alpha=1)
        plt.title(title)
        plt.xlabel(str(type(x_index)))
        plt.ylabel(str(type(y_true)))
        plt.legend()
        plt.show()

    @classmethod
    def vis_crossEntropy(cls, predicted_prob, log_loss):
        """
        Description:
            visualize the cross-entropy (log loss)

        References: 
            https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

        :param predicted_prob: list of all the predicted probabilities values (must be in range 0 and 1)
        :param log_loss: list of all the actual logarithmic loss values
        """        
        plt.xlabel('Predicted probability')
        plt.ylabel('Log loss')
        plt.scatter(predicted_prob, log_loss)
        plt.legend()
        plt.show()

    @classmethod
    def vis_clustering(cls, data, y_clusters, x_label='', y_label='', ts=False):
        """
        plot clustering out with limited to data in 2 columns 0, 1)

        :param data:
        :param y_clusters:
        :param x_label:
        :param y_label:
        :param ts: set True if index is TimeSeries's type
        :return:
        """
        plt.figure(figsize=(16, 14))
        if ts:
            data.loc[:, 'Clusters'] = y_clusters
            for i in range(max(np.unique(y_clusters)) + 1):
                plt.scatter(StartML.lookup_value(data[['Clusters']], i, tup=False),
                            data.values[y_clusters == i, 0], s=10,
                            label='Cluster '+str(i))
            plt.xticks(rotation=45)
        else:
            for i in range(max(np.unique(y_clusters))+1):
                plt.scatter(data.values[y_clusters == i, 0], data.values[y_clusters == i, 1], s=10,
                            label='Cluster '+str(i))

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        title = ''
        for col in data.columns:
            title = title + '-' + str(col)
        plt.title(title)
        plt.show()

    @classmethod
    def vis_basemap(cls, data, magnitude=False):
        """
        Visual the Geo-coordinates Latitude Longitude
        References:
            https://matplotlib.org/basemap/

        :param data: pandas.core.frame.DataFrame (with geospatial coordinates 'Longitude' and 'Latitutde'
        :return:
        """
        try:
            latitude = data['Latitude'].tolist()
            longitude = data['Longitude'].tolist()
        except KeyError:
            print("Coordinates data Latitude and Longitude not sufficient")

        if 'Magnitude' in data.columns:
            mag = data['Magnitude'].tolist()
        else:
            mag = []

        earth = Basemap(projection='mill', resolution='c')
        x, y = earth(longitude, latitude)

        plt.figure(figsize=(16, 14))
        plt.title("Observation locations")

        if mag:
            print("Scatter function ...")
            size = [s**2 for s in mag]
            earth.scatter(x, y, s=size, marker='.', color='red')
        else:
            print("Plot function ...")
            earth.scatter(x, y, s=5, marker='x', color='red')

        # setup basemap
        # earth.etopo(alpha=0.1)
        # earth.bluemarble(alpha=0.42)
        earth.drawcoastlines()
        earth.fillcontinents(color='coral', lake_color='aqua')
        earth.drawmapboundary()
        earth.drawcountries()

        plt.show()

    @classmethod
    def vis_3d_scatter(cls, data, features):
        """
        Description: tbd

        :param data: pandas.core.frame.DataFrame
        :param features:
        :return:
        """
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        data_x = data[features[0]]  # x_data
        data_y = data[features[1]]  # y_data
        data_z = data[features[2]]  # z_data
        ax.scatter(data_x, data_y, data_z, c=data[features[2]])

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])

        plt.show()

    @classmethod
    def vis_nd_scatter(cls, data):
        """
        Idea to visualize data in n_dimension (n > 3)

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        pass

    @classmethod
    def vis_contourf(cls, data):
        """
        References:
            https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.contourf.html

        :param data: pandas.core.frame.DataFrame
        :return:
        """
        pass

    @classmethod
    def vis_square_matrix_plot(cls, data):
        """
        Description: plot Correlation Matrix to show the strengthen connection between data points
            Input data is a square matrix and value of every item show the relationship between pairwise data points
            in directed graph (set: alpha parameter to display the transparency), respectively idea of which variables
            have a high correlation with each other.

        Used in Graph Analytics
        :param data: pandas.core.frame.DataFrame
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(data.corr(), vmin=-1, vmax=1)
        fig.colorbar(cax)        
        ax.set_xticks(np.arange(0, 9, 1))
        ax.set_yticks(np.arange(0, 9, 1))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)
        plt.show()

    @classmethod
    def vis_scatter_matrix(cls, data):
        """
        Description: scatter plot for each pair of columns attributes for spotting structured relationships 
        """
        scatter_matrix(data)
        plt.show()

    @classmethod
    def vis_features_acc(cls):
        """
        Description:
            plot which shows the correlation between number of features and accuracy performance.
            Adding more features over the threshold's performance contributes very little and sometimes even leads to
            overfitting and degrades performance.

        Reference:
            https://developers.google.com/machine-learning/guides/text-classification/step-3

        :return:
        """
        pass

    @classmethod
    def vis_learning_curve(cls):
        """
        Description:
            plot the learning curve to see the changes and improvement of models when the training set is being changes.
            visulize the effect of the number of training data and performance of models
        
        Reference:
            https://www.dataquest.io/blog/learning-curves-machine-learning/
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
            https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
        """
        pass

    @classmethod
    def vis_more_dim(cls):
        """
        Description:
            use dimensional reduction (SVD, PCA, LDA) to reduce the number of features (>3) into 2 or 3 dimension
            and plot
        """
        pass
    
    @staticmethod
    def info_help():
        info = {
            "info_help_StartVis": StartVis.__name__,
            "StartVis.vis_bar_groupby ": StartVis.vis_bar_groupby.__doc__,
            "StartVis.vis_scatter ": StartVis.vis_scatter.__doc__,
            "StartVis.vis_hist ": StartVis.vis_hist.__doc__,
            }

        return info


info_vis = StartVis.info_help()
