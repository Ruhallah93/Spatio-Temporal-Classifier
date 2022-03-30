import pandas as pd
import numpy as np
import tensorflow as tf
from regularizer2 import RestoringBest, ModelAnalyser
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from transformer import Transformer


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
        if restore_best:
            metric = ModelAnalyser(segment_size=self.segments_size,
                                   segment_overlap=self.segments_overlap,
                                   decision_size=self.decision_size,
                                   decision_overlap=self.decision_overlap,
                                   X=X_valid,
                                   y=y_valid,
                                   X_t=X_train,
                                   y_t=y_train)
            restoring_best = RestoringBest(metric=metric, monitor='ms_loss')
            self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True,
                           callbacks=[restoring_best])
        else:
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
        if restore_best:
            metric = ModelAnalyser(segment_size=self.segments_size,
                                   segment_overlap=self.segments_overlap,
                                   decision_size=self.decision_size,
                                   decision_overlap=self.decision_overlap,
                                   X=X_valid,
                                   y=y_valid,
                                   X_t=X_train,
                                   y_t=y_train)
            restoring_best = RestoringBest(metric=metric, monitor='ms_loss')
            self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True,
                           callbacks=[restoring_best])
        else:
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
        if restore_best:
            metric = ModelAnalyser(segment_size=self.segments_size,
                                   segment_overlap=self.segments_overlap,
                                   decision_size=self.decision_size,
                                   decision_overlap=self.decision_overlap,
                                   X=X_valid,
                                   y=y_valid,
                                   X_t=X_train,
                                   y_t=y_train)
            restoring_best = RestoringBest(metric=metric, monitor='ms_loss')
            self.model.fit(X_train, y_train_onehot,
                           validation_data=(X_valid, y_valid_onehot),
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle=True,
                           callbacks=[restoring_best])
        else:
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
