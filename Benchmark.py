#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


from datetime import datetime
from datetime import timedelta


# In[ ]:


import tensorflow as tf


# # Data Transformer

# In[2]:


import numpy as np


class Transformer:

    def __init__(self, decision_size, decision_overlap, segments_size=90, segments_overlap=45, sampling=2):
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.sampling = sampling
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap

    def transfer(self, dataset, features, method):
        print("segmenting data with " + str(len(dataset)) + " points")
        segments, labels = self.__segment_signal(dataset, features)
        print("making " + str(len(segments)) + " segments")
        if method == "table":
            segments_dataset = self.__transfer_table(segments, features)
        elif method == "1d":
            segments_dataset = self.__transfer_1d(segments, features)
        elif method == "2d":
            segments_dataset = self.__transfer_2d(segments, features)
        elif method == "3d_1ch":
          segments_dataset = self.__transfer_2d_1ch(segments, features)
        elif method == "3d":
            segments_dataset = self.__transfer_3d(segments, features)
        elif method == "4d":
            segments_dataset = self.__transfer_4d(segments, features)
        elif method == "rnn_2d":
            segments_dataset, labels =  self.__transfer_rnn_2d(segments, labels, features)
        elif method == "rnn_3d_1ch":
            segments_dataset, labels =  self.__transfer_rnn_3d_1ch(segments, labels, features)
        return segments_dataset, labels

    @staticmethod
    def data_shape(method, n_features, segments_size, segments_overlap=None, decision_size=None):
        if method == "table":
            return (None, n_features * segments_size)
        elif method == "1d":
            return (None, 1, n_features * segments_size, 1)
        elif method == "2d":
            return (None, n_features, segments_size)
        elif method == "3d_1ch":
            return (None, n_features, segments_size, 1)
        elif method == "3d":
            return (None, 1, segments_size, n_features)
        elif method == "4d":
            return (n_features, None, 1, segments_size, 1)
        elif method == "rnn_2d":
            s_b = Transformer.get_segments_a_decision_window(segments_size,
                                                             int(segments_size * segments_overlap),
                                                             decision_size)
            return (None, s_b, segments_size*n_features)
        elif method == "rnn_3d_1ch":
            s_b = Transformer.get_segments_a_decision_window(segments_size,
                                                             int(segments_size * segments_overlap),
                                                             decision_size)
            return (None, s_b, 1, segments_size*n_features)
        return ()

    @staticmethod
    def get_segments_a_decision_window(segment_size, segment_overlap_size, decision_size):
        return int((decision_size - segment_size) / (segment_size - segment_overlap_size) + 1)

    def __transfer_rnn_2d(self, segments, labels, features):
        y_output = []
        x_output = []
        c = len(np.unique(labels))
        s = Transformer.get_segments_a_decision_window(self.segments_size,
                                                       self.segments_overlap,
                                                       self.decision_size)
        r = int(np.floor(s * self.decision_overlap))
        for _id in np.unique(labels):
          subset = segments[np.where(labels == _id)]
          n = subset.shape[0]
          o = int(np.floor((n - r) / (s - r)))
          for i in range(o):
            row = []
            for j in range(s):
              A = subset[i * (s-r) + j]
              A = A.reshape(A.shape[0]*A.shape[1])
              row.append(A)
            y_output.append(_id)
            x_output.append(row)
        x_output = np.array(x_output)
        y_output = np.array(y_output)
        return x_output, y_output

    def __transfer_rnn_3d_1ch(self, segments, labels, features):
        # (samples, time, channels=1, rows)
        y_output = []
        x_output = []
        c = len(np.unique(labels))
        s = Transformer.get_segments_a_decision_window(self.segments_size,
                                                       self.segments_overlap,
                                                       self.decision_size)
        r = int(np.floor(s * self.decision_overlap))
        for _id in np.unique(labels):
          subset = segments[np.where(labels == _id)]
          n = subset.shape[0]
          o = int(np.floor((n - r) / (s - r)))
          for i in range(o):
            row = []
            for j in range(s):
              A = subset[i * (s-r) + j]
              A = A.reshape(A.shape[0]*A.shape[1])
              row.append([A])
            y_output.append(_id)
            x_output.append(row)
        x_output = np.array(x_output)
        y_output = np.array(y_output)
        return x_output, y_output

    def __transfer_table(self, segments, features):
        new_dataset = []
        for segment in segments:
            row = []
            for feature_i in range(len(features)):
                for i in range(len(segment[feature_i])):
                    row.append(segment[feature_i][i])
            new_dataset.append(row)

        new_dataset = np.array(new_dataset)
        return new_dataset

    def __transfer_1d(self, segments, features):
        new_dataset = []
        for segment in segments:
            row = []
            for feature_i in range(len(features)):
                for i in range(len(segment[feature_i])):
                    row.append(segment[feature_i][i])
            new_dataset.append([row])

        new_dataset = np.array(new_dataset)
        return np.expand_dims(new_dataset, axis=3)

    def __transfer_2d(self, segments, features):
        new_dataset = []
        for segment in segments:
            row = []
            for feature_i in range(len(features)):
                row.append(segment[feature_i])
            new_dataset.append(row)

        new_dataset = np.array(new_dataset)
        return new_dataset

    def __transfer_3d_1ch(self, segments, features):
        new_dataset = []
        for segment in segments:
            row = []
            for feature_i in range(len(features)):
                row.append(segment[feature_i])
            new_dataset.append(row)

        new_dataset = np.array(new_dataset)
        return np.expand_dims(new_dataset, axis=3)

    def __transfer_3d(self, segments, features):
        new_dataset = []
        for segment in segments:
            row = []
            for i in range(len(segment[0])):
                cell = []
                for feature_i in range(len(features)):
                    cell.append(segment[feature_i][i])
                row.append(cell)
            new_dataset.append([row])

        new_dataset = np.array(new_dataset)
        return new_dataset

    def __transfer_4d(self, segments, features):
        new_dataset = []
        for feature_i in range(len(features)):
            row = []
            for segment in segments:
                cell = []
                for element in segment[feature_i]:
                    cell.append([element])
                row.append([cell])
            new_dataset.append(row)

        new_dataset = np.array(new_dataset)
        return new_dataset

    def __windows(self, data):
        start = 0
        while start < data.count():
            yield int(start), int(start + self.segments_size)
            start += (self.segments_size - self.segments_overlap)

    def __segment_signal(self, dataset, features):
        segments = []
        labels = []
        for class_i in np.unique(dataset["id"]):
            subset = dataset[dataset["id"] == class_i]
            for (start, end) in self.__windows(subset["id"]):
                feature_slices = []
                for feature in features:
                    feature_slices.append(subset[feature][start:end].tolist())
                if len(feature_slices[0]) == self.segments_size:
                    segments.append(feature_slices)
                    labels.append(class_i)
        return np.array(segments), np.array(labels)


# # Dataset

# In[3]:


import glob
import os
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn import preprocessing


class Dataset:

    def __init__(self, db_path, sample_rate, features, 
                 window_time, window_overlap_percentage, 
                 decision_time, decision_overlap_percentage, 
                 add_noise, noise_rate,
                 train_blocks: list, valid_blocks: list, test_blocks: list, 
                 data_length_time=-1):
        """
        :param db_path:
        :param sample_rate:
        :param features:
        :param window_time: in seconds
        :param window_overlap_percentage: example: 0.75 for 75%
        :param add_noise: True or False
        :param noise_rate:
        :param train_blocks:
        :param valid_blocks:
        :param test_blocks:
        :param data_length_time: the amount of data from each class in seconds. -1 means whole existing data.
        """
        self.db_path = db_path
        self.features = features
        self.sample_rate = sample_rate
        self.window_size = window_time * sample_rate
        self.window_overlap_size = int(self.window_size * window_overlap_percentage)
        self.decision_size = decision_time * sample_rate
        self.decision_overlap_size = int(self.decision_size * decision_overlap_percentage)
        self.decision_overlap_percentage = decision_overlap_percentage
        self.add_noise = add_noise
        self.noise_rate = noise_rate
        self.train_blocks = train_blocks
        self.valid_blocks = valid_blocks
        self.test_blocks = test_blocks
        self.data_length_size = data_length_time * sample_rate if data_length_time != -1 else -1

        # Initialization
        self.train_dataset = pd.DataFrame()
        self.valid_dataset = pd.DataFrame()
        self.test_dataset = pd.DataFrame()
        self.n_train_dataset = pd.DataFrame()
        self.n_valid_dataset = pd.DataFrame()
        self.n_test_dataset = pd.DataFrame()
        self.X_train = np.array([])
        self.y_train = np.array([])
        self.X_valid = np.array([])
        self.y_valid = np.array([])
        self.X_test = np.array([])
        self.y_test = np.array([])

    def load_data(self, n_classes, method):
        segments_path = self.db_path +                         "segments/" +                         "method: " + str(method) + os.sep +                         "wl: " + str(self.window_size) + os.sep +                         "wo: " + str(self.window_overlap_size) + os.sep +                         "dl: " + str(self.decision_size) + os.sep +                         "do: " + str(self.decision_overlap_size) + os.sep +                         "train: " + str(self.train_blocks) + os.sep +                         "valid: " + str(self.valid_blocks) + os.sep +                         "test: " + str(self.test_blocks) + os.sep
        if os.path.exists(segments_path + 'X_train.npy')                 and os.path.exists(segments_path + 'y_train.npy')                 and os.path.exists(segments_path + 'X_valid.npy')                 and os.path.exists(segments_path + 'y_valid.npy')                 and os.path.exists(segments_path + 'X_test.npy')                 and os.path.exists(segments_path + 'y_test.npy'):
            print("Dataset is already")
            self.X_train = np.load(segments_path + 'X_train.npy')
            self.y_train = np.load(segments_path + 'y_train.npy')
            self.X_valid = np.load(segments_path + 'X_valid.npy')
            self.y_valid = np.load(segments_path + 'y_valid.npy')
            self.X_test = np.load(segments_path + 'X_test.npy')
            self.y_test = np.load(segments_path + 'y_test.npy')
        else:
            self.__preprocess(n_classes, method)
            # Save Dataset
            if not os.path.exists(segments_path):
                os.makedirs(segments_path)
            np.save(segments_path + 'X_train.npy', self.X_train)
            np.save(segments_path + 'y_train.npy', self.y_train)
            np.save(segments_path + 'X_valid.npy', self.X_valid)
            np.save(segments_path + 'y_valid.npy', self.y_valid)
            np.save(segments_path + 'X_test.npy', self.X_test)
            np.save(segments_path + 'y_test.npy', self.y_test)

        def to_dic(data):
            dic = {}
            for i, x in enumerate(data):
                dic[str(i)] = x
            return dic

        if len(self.X_train.shape) == 5:
            self.X_train = to_dic(self.X_train)
            self.X_valid = to_dic(self.X_valid)
            self.X_test = to_dic(self.X_test)

    def __preprocess(self, n_classes, method):
        csv_paths = np.random.choice(glob.glob(self.db_path + "*.csv"), n_classes, replace=False)

        self.class_names = {}
        for i, csv_path in enumerate(csv_paths):
            label = os.path.basename(csv_path).split('.')[0]
            self.class_names[label] = i
            train, valid, test = self.__read_data(csv_path, self.features, label)
            train['id'] = i
            valid['id'] = i
            test['id'] = i
            self.train_dataset = pd.concat([self.train_dataset, train])
            self.valid_dataset = pd.concat([self.valid_dataset, valid])
            self.test_dataset = pd.concat([self.test_dataset, test])

        self.__standardization()
        self.__segmentation(method=method)

    def __read_data(self, path, features, label):
        data = pd.read_csv(path, low_memory=False)
        data = data[features]
        data = data.fillna(data.mean())
        length = self.data_length_size if self.data_length_size != -1 else data.shape[0]
        print('class: %5s, data size: %s, selected data size: %s' % (
            label, str(timedelta(seconds=int(data.shape[0] / self.sample_rate))),
            str(timedelta(seconds=int(length / self.sample_rate)))))
        return self.__split_to_train_valid_test(data)

    def __split_to_train_valid_test(self, data):
        n_blocks = max(self.train_blocks + self.valid_blocks + self.test_blocks) + 1
        block_length = int(len(data[:self.data_length_size]) / n_blocks)

        train_data = pd.DataFrame()
        for i in range(len(self.train_blocks)):
            start = self.train_blocks[i] * block_length
            end = self.train_blocks[i] * block_length + block_length - 1
            if train_data.empty:
                train_data = data[start:end]
            else:
                train_data = pd.concat([data[start:end], train_data])

        valid_data = pd.DataFrame()
        for i in range(len(self.valid_blocks)):
            start = self.valid_blocks[i] * block_length
            end = self.valid_blocks[i] * block_length + block_length - 1
            if valid_data.empty:
                valid_data = data[start:end]
            else:
                valid_data = pd.concat([data[start:end], valid_data])

        test_data = pd.DataFrame()
        for i in range(len(self.test_blocks)):
            start = self.test_blocks[i] * block_length
            end = self.test_blocks[i] * block_length + block_length - 1
            if test_data.empty:
                test_data = data[start:end]
            else:
                test_data = pd.concat([data[start:end], test_data])

        if self.add_noise:
            test_data = self.__add_noise_to_data(test_data)

        return train_data, valid_data, test_data

    def __add_noise_to_data(self, x):
        x_power = x ** 2
        sig_avg_watts = np.mean(x_power)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        noise_avg_db = sig_avg_db - self.target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), size=x.shape)
        return x + noise_volts

    def __standardization(self):
        scaler = preprocessing.StandardScaler()
        scaler = scaler.fit(self.train_dataset.iloc[:, :-1])
        n_train_dataset = scaler.transform(self.train_dataset.iloc[:, :-1])
        n_valid_dataset = scaler.transform(self.valid_dataset.iloc[:, :-1])
        n_test_dataset = scaler.transform(self.test_dataset.iloc[:, :-1])

        self.n_train_dataset = pd.DataFrame(n_train_dataset, columns=self.features)
        self.n_valid_dataset = pd.DataFrame(n_valid_dataset, columns=self.features)
        self.n_test_dataset = pd.DataFrame(n_test_dataset, columns=self.features)
        self.n_train_dataset['id'] = self.train_dataset.iloc[:, -1].tolist()
        self.n_valid_dataset['id'] = self.valid_dataset.iloc[:, -1].tolist()
        self.n_test_dataset['id'] = self.test_dataset.iloc[:, -1].tolist()

    def __segmentation(self, method):
        transformer = Transformer(segments_size=self.window_size, 
                                  segments_overlap=self.window_overlap_size,
                                  decision_size=self.decision_size,
                                  decision_overlap=self.decision_overlap_percentage)
        self.X_train, self.y_train = transformer.transfer(self.n_train_dataset, self.features, method=method)
        self.X_valid, self.y_valid = transformer.transfer(self.n_valid_dataset, self.features, method=method)
        self.X_test, self.y_test = transformer.transfer(self.n_test_dataset, self.features, method=method)


# # RECURRENT MODELS

# In[4]:


class LSTM():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.input_shape = self.get_input_shape()
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_l()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(),
                                            n_features=self.n_features,
                                            segments_size=self.segments_size,
                                            segments_overlap=self.segments_overlap,
                                            decision_size=self.decision_size)
        return data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "rnn_2d"

    def count_params(self):
        return self.model.count_params()

    def build_model_l(self):
        print(self.input_shape)
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.LSTM(48)(input_)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# In[5]:


class ConvLSTM():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.input_shape = self.get_input_shape()
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_l()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(),
                                            n_features=self.n_features,
                                            segments_size=self.segments_size,
                                            segments_overlap=self.segments_overlap,
                                            decision_size=self.decision_size)
        return data_shape[-3], data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "rnn_3d_1ch"

    def count_params(self):
        return self.model.count_params()

    def build_model_l(self):
        print(self.input_shape)
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.ConvLSTM1D(2, kernel_size=3, padding='valid',
                                       data_format="channels_first",
                                       return_sequences=True)(input_)
        x = tf.keras.layers.ConvLSTM1D(4, kernel_size=3, padding='valid'
                                       , data_format="channels_first"
                                       , activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# In[6]:


class GRU():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.input_shape = self.get_input_shape()
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_l()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(),
                                            n_features=self.n_features,
                                            segments_size=self.segments_size,
                                            segments_overlap=self.segments_overlap,
                                            decision_size=self.decision_size)
        return data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "rnn_2d"

    def count_params(self):
        return self.model.count_params()

    def build_model_l(self):
        print("input_shape:",self.input_shape)
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.GRU(48)(input_)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# In[7]:


class BiLSTM():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.input_shape = self.get_input_shape()
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_l()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(),
                                            n_features=self.n_features,
                                            segments_size=self.segments_size,
                                            segments_overlap=self.segments_overlap,
                                            decision_size=self.decision_size)
        return data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "rnn_2d"

    def count_params(self):
        return self.model.count_params()

    def build_model_l(self):
        print(self.input_shape)
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48, return_sequences=True))(input_)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48))(x)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(x)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# # Classic Models

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


# Ensembles of Convolutinal Neural Network
class MULTI_CNN():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_multi()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-3], data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "4d"

    def count_params(self):
        return self.model.count_params()

    def build_model_multi(self):
        input_ = [tf.keras.layers.Input(shape=self.input_shape, name=str(i)) for i in range(self.n_features)]
        models = [self.build_cn_part(input_[i]) for i in range(self.n_features)]

        combined = tf.keras.layers.Concatenate(axis=1)(models)

        dense = tf.keras.layers.Flatten()(combined)
        dense = tf.keras.layers.Dense(1024)(dense)
        dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(dense)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def build_cn_part(self, x):
        x = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        return x

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# Convolutinal Neural Network
class CNN_L():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model_l()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-3], data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "3d"

    def count_params(self):
        return self.model.count_params()

    def build_model_l(self):
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        x = tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(input_)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=1, padding="same", activation='relu')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        dense = tf.keras.layers.Flatten()(x)
        dense = tf.keras.layers.Dense(1024)(dense)
        dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(dense)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# Multi Linear Perceptron
class MLP():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        self.model = self.build_model()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-3], data_shape[-2], data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "3d"

    def count_params(self):
        return self.model.count_params()

    def build_model(self):
        input_ = tf.keras.layers.Input(shape=self.input_shape)

        dense = tf.keras.layers.Flatten()(input_)
        dense = tf.keras.layers.Dense(1024)(dense)
        dense = tf.keras.layers.LeakyReLU(alpha=0.2)(dense)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        dense = tf.keras.layers.Dense(self.classes, activation='softmax')(dense)

        model = tf.keras.models.Model(inputs=input_, outputs=[dense])
        return model

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True)

        return self.model.predict(X_test)


# K-Nearest Neighbors
class KNN():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap

        # Build and compile the model
        self.model = KNeighborsClassifier()

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "table"

    def count_params(self):
        return -1

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)

        self.model.fit(X_train, y_train_onehot)

        return self.model.predict(X_test)


# Logistic Regression
class LR():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap

        # Build and compile the model
        self.model = LogisticRegression(solver='sag', multi_class='multinomial')

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "table"

    def count_params(self):
        return -1

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train)

        enc = preprocessing.OneHotEncoder()
        enc.fit(y_train.reshape(-1, 1))
        return enc.transform(self.model.predict(X_test).reshape(-1, 1)).toarray()


# Random Forest
class RF():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap

        # Build and compile the model
        self.model = RandomForestClassifier()

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "table"

    def count_params(self):
        return -1

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)

        self.model.fit(X_train, y_train_onehot)

        return self.model.predict(X_test)


# Support Vector Machine
class SVM():
    def __init__(self, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        self.classes = classes
        self.n_features = n_features
        self.segments_size = segments_size
        self.input_shape = self.get_input_shape()
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap

        # Build and compile the model
        self.model = LinearSVC()

    def get_input_shape(self):
        data_shape = Transformer.data_shape(method=self.get_data_arrangement(), n_features=self.n_features,
                                            segments_size=self.segments_size)
        return data_shape[-1]

    @staticmethod
    def get_data_arrangement():
        return "table"

    def count_params(self):
        return -1

    def train(self, epochs, X_train, y_train, X_valid, y_valid, X_test, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        self.model.fit(X_train, y_train)

        enc = preprocessing.OneHotEncoder()
        enc.fit(y_train.reshape(-1, 1))
        return enc.transform(self.model.predict(X_test).reshape(-1, 1)).toarray()


# # Evaluation

# In[ ]:


import os
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def analysis_model(y_pred, y_real_raw):
    result = {}
    loss_fn = MeanSquaredError()

    result['mse_loss'] = loss_fn(np.asarray(pd.get_dummies(y_real_raw), dtype=np.int8),
                                         np.asarray(y_pred, dtype=np.float)).numpy()
    y_pred_arg = np.argmax(y_pred, axis=1)
    result['accuracy'] = accuracy_score(y_real_raw, y_pred_arg)
    result['precision'] = precision_score(y_real_raw, y_pred_arg, average='macro')
    result['recall'] = recall_score(y_real_raw, y_pred_arg, average='macro')
    result['f1'] = f1_score(y_real_raw, y_pred_arg, average='macro')

    return result


# # Train

# In[ ]:


def train_model(dataset: Dataset, classifier, epochs, batch_size):
    y_test_prediction = classifier.train(epochs=epochs,
                                         X_train=dataset.X_train,
                                         y_train=dataset.y_train,
                                         X_valid=dataset.X_valid,
                                         y_valid=dataset.y_valid,
                                         X_test=dataset.X_test,
                                         batch_size=batch_size)

    result_test = analysis_model(y_pred=y_test_prediction,
                                 y_real_raw=dataset.y_test)

    print('Test(%s):%5.2f' % (str(classifier), result_test['accuracy']))

    return result_test


# # Spliting

# In[ ]:


def h_block_analyzer(db_path, sample_rate, features, n_classes, noise_rate, segments_time,
                     segments_overlap, 
                     decision_time, decision_overlap,
                     classifier, epochs, batch_size, data_length_time,
                     n_h_block, n_train_h_block, n_valid_h_block, n_test_h_block, h_moving_step=1):
    """
    :param db_path: the address of dataset directory
    :param sample_rate: the sampling rate of signals
    :param features: the signals of original data
    :param n_classes: the number of classes
    :param noise_rate: the rate of noises injected to test data
    :param segments_time: the length of each segment in seconds.
    :param segments_overlap: the overlap of each segment
    :param classifier: the neural network
    :param epochs: the number of training epochs
    :param batch_size: the number of segments in each batch
    :param n_h_block: the number of all hv blocks
    :param n_train_h_block: the number of hv blocks to train network
    :param n_valid_h_block: the number of hv blocks to validate network
    :param n_test_h_block: the number of hv blocks to test network
    :param h_moving_step: the number of movement of test and validation blocks in each iteration
    :return:
    """

    statistics = {}
    add_noise = noise_rate < 100

    # Create hv blocks
    data_blocks = [i for i in range(n_h_block)]
    n_vt = (n_valid_h_block + n_test_h_block)
    n_iteration = int((n_h_block - n_vt) / h_moving_step)
    for i in range(n_iteration + 1):
        print('iteration: %d/%d' % (i + 1, n_iteration + 1))

        training_container = data_blocks[0:i] + data_blocks[i + n_vt:n_h_block]
        train_blocks = training_container[:n_train_h_block]
        valid_blocks = data_blocks[i: i + n_valid_h_block]
        test_blocks = data_blocks[i + n_valid_h_block: i + n_vt]

        dataset = Dataset(db_path,
                          sample_rate,
                          features=features,
                          window_time=segments_time,
                          window_overlap_percentage=segments_overlap,
                          decision_time=decision_time,
                          decision_overlap_percentage=decision_overlap,
                          add_noise=add_noise,
                          noise_rate=noise_rate,
                          train_blocks=train_blocks,
                          valid_blocks=valid_blocks,
                          test_blocks=test_blocks,
                          data_length_time=data_length_time)

        dataset.load_data(n_classes=n_classes, method=classifier.get_data_arrangement())
        result = train_model(dataset=dataset, classifier=classifier, epochs=epochs,
                             batch_size=batch_size)

        for key in result.keys():
            if not key in statistics:
                statistics[key] = []
            statistics[key].append(result[key])
        
        print(statistics)
    return statistics


# # Save Result

# In[ ]:


def save_result(log_dir, data: dict):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save to file
    with open(log_dir + 'statistics.txt', 'a') as f:
        f.write('\n==========***==========\n')
        f.write(str(data))
        f.write('\n')

    csv_file = log_dir + 'statistics.csv'
    file_exists = os.path.isfile(csv_file)
    try:
        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(data.keys())
            writer.writerow(data.values())
            csvfile.close()
    except IOError:
        print("I/O error")


# # Main

# In[ ]:


problems = {'DriverIdentification':{},
            'ConfLongDemo_JSI':{},
            'Healthy_Older_People':{},
            'Motor_Failure_Time':{},
            'Power_consumption':{},
            'PRSA2017':{},
            'RSSI':{},
            'User_Identification_From_Walking':{},
            'WISDM':{},
           }
problems['DriverIdentification']['dataset'] = './Spatio-Temporal-Classifier/datasets/DriverIdentification/'
problems['DriverIdentification']['n_classes'] = 10
problems['DriverIdentification']['features'] = ['x-accelerometer', 'y-accelerometer', 'z-accelerometer', 'x-gyroscope', 'y-gyroscope', 'z-gyroscope']
problems['DriverIdentification']['sample_rate'] = 2
problems['DriverIdentification']['data_length_time'] = -1
problems['DriverIdentification']['n_h_block'] = 15
problems['DriverIdentification']['n_train_h_block'] = 9
problems['DriverIdentification']['n_valid_h_block'] = 2
problems['DriverIdentification']['n_test_h_block'] = 4
problems['DriverIdentification']['h_moving_step'] = 1
problems['DriverIdentification']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60]
problems['DriverIdentification']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60]

problems['ConfLongDemo_JSI']['dataset'] = './Spatio-Temporal-Classifier/datasets/ConfLongDemo_JSI/'
problems['ConfLongDemo_JSI']['n_classes'] = 5
problems['ConfLongDemo_JSI']['features'] = ["x", "y", "z"]
problems['ConfLongDemo_JSI']['sample_rate'] = 30
problems['ConfLongDemo_JSI']['data_length_time'] = -1
problems['ConfLongDemo_JSI']['n_h_block'] = 15
problems['ConfLongDemo_JSI']['n_train_h_block'] = 9
problems['ConfLongDemo_JSI']['n_valid_h_block'] = 2
problems['ConfLongDemo_JSI']['n_test_h_block'] = 4
problems['ConfLongDemo_JSI']['h_moving_step'] = 1
problems['ConfLongDemo_JSI']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60]
problems['ConfLongDemo_JSI']['segments_times'] = [3,4,5,6,7,8,9,10,30,60]

problems['Healthy_Older_People']['dataset'] = './Spatio-Temporal-Classifier/datasets/Healthy_Older_People/'
problems['Healthy_Older_People']['n_classes'] = 12
problems['Healthy_Older_People']['features'] = ["X", "Y", "Z"]
problems['Healthy_Older_People']['sample_rate'] = 1
problems['Healthy_Older_People']['data_length_time'] = -1
problems['Healthy_Older_People']['n_h_block'] = 15
problems['Healthy_Older_People']['n_train_h_block'] = 9
problems['Healthy_Older_People']['n_valid_h_block'] = 2
problems['Healthy_Older_People']['n_test_h_block'] = 4
problems['Healthy_Older_People']['h_moving_step'] = 1
problems['Healthy_Older_People']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60]
problems['Healthy_Older_People']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60]

problems['Motor_Failure_Time']['dataset'] = './Spatio-Temporal-Classifier/datasets/Motor_Failure_Time/'
problems['Motor_Failure_Time']['n_classes'] = 3
problems['Motor_Failure_Time']['features'] = ['x', 'y', 'z']
problems['Motor_Failure_Time']['sample_rate'] = 18
problems['Motor_Failure_Time']['data_length_time'] = -1
problems['Motor_Failure_Time']['n_h_block'] = 15
problems['Motor_Failure_Time']['n_train_h_block'] = 9
problems['Motor_Failure_Time']['n_valid_h_block'] = 2
problems['Motor_Failure_Time']['n_test_h_block'] = 4
problems['Motor_Failure_Time']['h_moving_step'] = 1
problems['Motor_Failure_Time']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60]
problems['Motor_Failure_Time']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60]

problems['Power_consumption']['dataset'] = './Spatio-Temporal-Classifier/datasets/Power_consumption/'
problems['Power_consumption']['n_classes'] = 3
problems['Power_consumption']['features'] = ['Temperature', 'Humidity', 'Wind Speed', 
                                             'general diffuse flows', 'diffuse flows', 
                                             'Consumption']
problems['Power_consumption']['sample_rate'] = 1
problems['Power_consumption']['data_length_time'] = -1
problems['Power_consumption']['n_h_block'] = 15
problems['Power_consumption']['n_train_h_block'] = 9
problems['Power_consumption']['n_valid_h_block'] = 2
problems['Power_consumption']['n_test_h_block'] = 4
problems['Power_consumption']['h_moving_step'] = 1
problems['Power_consumption']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60,20*60,30*60,40*60]
problems['Power_consumption']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60,20*60]

problems['PRSA2017']['dataset'] = './Spatio-Temporal-Classifier/datasets/PRSA2017/'
problems['PRSA2017']['n_classes'] = 12
problems['PRSA2017']['features'] = ['PM2.5','PM10','SO2','NO2','CO','O3','TEMP','PRES','DEWP','RAIN','wd','WSPM']
problems['PRSA2017']['sample_rate'] = 1
problems['PRSA2017']['data_length_time'] = -1
problems['PRSA2017']['n_h_block'] = 15
problems['PRSA2017']['n_train_h_block'] = 9
problems['PRSA2017']['n_valid_h_block'] = 2
problems['PRSA2017']['n_test_h_block'] = 4
problems['PRSA2017']['h_moving_step'] = 1
problems['PRSA2017']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60,20*60,30*60,40*60]
problems['PRSA2017']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60,20*60]

problems['RSSI']['dataset'] = './Spatio-Temporal-Classifier/datasets/RSSI/'
problems['RSSI']['n_classes'] = 12
problems['RSSI']['features'] = ['rssiOne', 'rssiTwo']
problems['RSSI']['sample_rate'] = 1
problems['RSSI']['data_length_time'] = -1
problems['RSSI']['n_h_block'] = 6
problems['RSSI']['n_train_h_block'] = 4
problems['RSSI']['n_valid_h_block'] = 1
problems['RSSI']['n_test_h_block'] = 1
problems['RSSI']['h_moving_step'] = 1
problems['RSSI']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60]
problems['RSSI']['segments_times'] = [3,4,5,6,7,8,9,10,30,60]

problems['User_Identification_From_Walking']['dataset'] = './Spatio-Temporal-Classifier/datasets/User_Identification_From_Walking/'
problems['User_Identification_From_Walking']['n_classes'] = 13
problems['User_Identification_From_Walking']['features'] = [' x acceleration', ' y acceleration', ' z acceleration']
problems['User_Identification_From_Walking']['sample_rate'] = 32
problems['User_Identification_From_Walking']['data_length_time'] = -1
problems['User_Identification_From_Walking']['n_h_block'] = 10
problems['User_Identification_From_Walking']['n_train_h_block'] = 5
problems['User_Identification_From_Walking']['n_valid_h_block'] = 2
problems['User_Identification_From_Walking']['n_test_h_block'] = 3
problems['User_Identification_From_Walking']['h_moving_step'] = 1
problems['User_Identification_From_Walking']['decision_times'] = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
problems['User_Identification_From_Walking']['segments_times'] = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

problems['WISDM']['dataset'] = './Spatio-Temporal-Classifier/datasets/WISDM/'
problems['WISDM']['n_classes'] = 10
problems['WISDM']['features'] = ['X-accel', 'Y-accel', 'Z-accel']
problems['WISDM']['sample_rate'] = 20
problems['WISDM']['data_length_time'] = -1
problems['WISDM']['n_h_block'] = 15
problems['WISDM']['n_train_h_block'] = 9
problems['WISDM']['n_valid_h_block'] = 4
problems['WISDM']['n_test_h_block'] = 2
problems['WISDM']['h_moving_step'] = 1
problems['WISDM']['decision_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60]
problems['WISDM']['segments_times'] = [3,4,5,6,7,8,9,10,30,60,2*60,3*60,4*60,5*60,6*60,7*60]


# In[ ]:


noise_rate = 100
epochs = 50
batch_size = 32


# In[ ]:


problem = 'Healthy_Older_People'
dataset = problems[problem]['dataset']
n_classes = problems[problem]['n_classes']
features = problems[problem]['features']
sample_rate = problems[problem]['sample_rate']
data_length_time = problems[problem]['data_length_time']
n_h_block = problems[problem]['n_h_block']
n_train_h_block = problems[problem]['n_train_h_block']
n_valid_h_block = problems[problem]['n_valid_h_block']
n_test_h_block = problems[problem]['n_test_h_block']
h_moving_step = problems[problem]['h_moving_step']
decision_times = problems[problem]['decision_times']
segments_times = problems[problem]['segments_times']


# # Classic Configuration

# In[ ]:


# models = ['KNN','MLP','LR','RF','SVM','CNN_L']
# # 'CNN_L','MLP','KNN','LR','RF','SVM'
# decision_time = 0
# decision_overlap = 0
# log_dir = f"./comparisons/log/{problem}/classic/"

# for model in models:
#     for segments_time in decision_times:
#         for segments_overlap in [0.0]:
#             classifier = eval(model)(classes=n_classes,
#                                      n_features=len(features),
#                                      segments_size=int(segments_time * sample_rate),
#                                      segments_overlap=segments_overlap,
#                                      decision_size=int(decision_time * sample_rate),
#                                      decision_overlap=decision_overlap)

#             # cross-validation
#             start = datetime.now()
#             statistics = h_block_analyzer(db_path=dataset,
#                                           sample_rate=sample_rate,
#                                           features=features,
#                                           n_classes=n_classes,
#                                           noise_rate=noise_rate,
#                                           segments_time=segments_time,
#                                           segments_overlap=segments_overlap,
#                                           decision_time=decision_time,
#                                           decision_overlap=decision_overlap,
#                                           classifier=classifier,
#                                           epochs=epochs,
#                                           batch_size=batch_size,
#                                           data_length_time=data_length_time,
#                                           n_h_block=n_h_block,
#                                           n_train_h_block=n_train_h_block,
#                                           n_valid_h_block=n_valid_h_block,
#                                           n_test_h_block=n_test_h_block,
#                                           h_moving_step=h_moving_step)
#             end = datetime.now()
#             running_time = end - start

#             # Summarizing the results of cross-validation
#             data = {}
#             data['dataset'] = dataset
#             data['class'] = str(n_classes)
#             data['features'] = str(features)
#             data['sample_rate'] = str(sample_rate)
#             data['noise_rate'] = str(noise_rate)
#             data['epochs'] = str(epochs)
#             data['batch_size'] = str(batch_size)
#             data['data_length_time'] = str(data_length_time)
#             data['n_h_block'] = str(n_h_block)
#             data['n_train_h_block'] = str(n_train_h_block)
#             data['n_valid_h_block'] = str(n_valid_h_block)
#             data['n_test_h_block'] = str(n_test_h_block)
#             data['h_moving_step'] = str(h_moving_step)
#             data['segments_time'] = str(segments_time)
#             data['segments_overlap'] = str(segments_overlap)
#             data['inner_classifier'] = str(model)
#             data['datetime'] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
#             data['running_time'] = str(running_time.seconds) + " seconds"
#             data['n_params'] = classifier.count_params()
#             data['segments_time'] = timedelta(seconds=int(segments_time))
#             data['segments_overlap'] = segments_overlap
#             data['decision_time'] = timedelta(seconds=int(decision_time))
#             data['decision_overlap'] = decision_overlap
#             statistics_summary = {}
#             for key in statistics.keys():
#                 statistics_summary[key + '_mean'] = np.average(statistics[key])
#                 statistics_summary[key + '_std'] = np.std(statistics[key])
#                 statistics_summary[key + '_max'] = np.max(statistics[key])
#                 statistics_summary[key + '_min'] = np.min(statistics[key])
#             data.update(statistics_summary)
#             # Save information
#             save_result(log_dir=log_dir, data=data)


# # Recurrent Configuration

# In[ ]:


models = ['BiLSTM']
# 'GRU','LSTM','ConvLSTM','BiLSTM'
# 'Power_consumption','PRSA2017','RSSI','User_Identification_From_Walking','WISDM','Motor_Failure_Time'
for problem in ['Motor_Failure_Time']:
    dataset = problems[problem]['dataset']
    n_classes = problems[problem]['n_classes']
    features = problems[problem]['features']
    sample_rate = problems[problem]['sample_rate']
    data_length_time = problems[problem]['data_length_time']
    n_h_block = problems[problem]['n_h_block']
    n_train_h_block = problems[problem]['n_train_h_block']
    n_valid_h_block = problems[problem]['n_valid_h_block']
    n_test_h_block = problems[problem]['n_test_h_block']
    h_moving_step = problems[problem]['h_moving_step']
    decision_times = problems[problem]['decision_times']
    decision_times = [3*60,4*60,5*60,6*60,7*60,8*60,9*60,10*60]
    segments_times = problems[problem]['segments_times']
    
    log_dir = f"./comparisons/log/{problem}/recurrent/"
    
    for model in models:
        for decision_time in decision_times:
            for decision_overlap in [0.0]:
                for segments_time in segments_times:
                    for segments_overlap in [0.75]:
                        if float(decision_time*0.75) < float(segments_time):
                            continue
                        classifier = eval(model)(classes=n_classes,
                                                 n_features=len(features),
                                                 segments_size=int(segments_time * sample_rate),
                                                 segments_overlap=segments_overlap,
                                                 decision_size=int(decision_time * sample_rate),
                                                 decision_overlap=decision_overlap)
    
                        # cross-validation
                        start = datetime.now()
                        statistics = h_block_analyzer(db_path=dataset,
                                                      sample_rate=sample_rate,
                                                      features=features,
                                                      n_classes=n_classes,
                                                      noise_rate=noise_rate,
                                                      segments_time=segments_time,
                                                      segments_overlap=segments_overlap,
                                                      decision_time=decision_time,
                                                      decision_overlap=decision_overlap,
                                                      classifier=classifier,
                                                      epochs=epochs,
                                                      batch_size=batch_size,
                                                      data_length_time=data_length_time,
                                                      n_h_block=n_h_block,
                                                      n_train_h_block=n_train_h_block,
                                                      n_valid_h_block=n_valid_h_block,
                                                      n_test_h_block=n_test_h_block,
                                                      h_moving_step=h_moving_step)
                        end = datetime.now()
                        running_time = end - start
    
                        # Summarizing the results of cross-validation
                        data = {}
                        data['dataset'] = dataset
                        data['class'] = str(n_classes)
                        data['features'] = str(features)
                        data['sample_rate'] = str(sample_rate)
                        data['noise_rate'] = str(noise_rate)
                        data['epochs'] = str(epochs)
                        data['batch_size'] = str(batch_size)
                        data['data_length_time'] = str(data_length_time)
                        data['n_h_block'] = str(n_h_block)
                        data['n_train_h_block'] = str(n_train_h_block)
                        data['n_valid_h_block'] = str(n_valid_h_block)
                        data['n_test_h_block'] = str(n_test_h_block)
                        data['h_moving_step'] = str(h_moving_step)
                        data['segments_time'] = str(segments_time)
                        data['segments_overlap'] = str(segments_overlap)
                        data['inner_classifier'] = str(model)
                        data['datetime'] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
                        data['running_time'] = str(running_time.seconds) + " seconds"
                        data['n_params'] = classifier.count_params()
                        data['segments_time'] = timedelta(seconds=int(segments_time))
                        data['segments_overlap'] = segments_overlap
                        data['decision_time'] = timedelta(seconds=int(decision_time))
                        data['decision_overlap'] = decision_overlap
                        statistics_summary = {}
                        for key in statistics.keys():
                            statistics_summary[key + '_mean'] = np.average(statistics[key])
                            statistics_summary[key + '_std'] = np.std(statistics[key])
                            statistics_summary[key + '_max'] = np.max(statistics[key])
                            statistics_summary[key + '_min'] = np.min(statistics[key])
                        data.update(statistics_summary)
                        # Save information
                        save_result(log_dir=log_dir, data=data)


# In[ ]:


print("End")


# In[ ]:




