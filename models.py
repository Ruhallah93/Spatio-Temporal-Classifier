import pandas as pd
import numpy as np
import tensorflow as tf
from regularizer import RestoringBest, ModelAnalyser


class ConvNet():
    def __init__(self, data_arrangement, model_size, input_shape, classes, n_features,
                 segments_size, segments_overlap, decision_size, decision_overlap):
        """
        :param input_shape: input_shape: (rows, cols, channels)
        """
        self.classes = classes
        self.n_features = n_features
        self.input_shape = input_shape
        self.segments_size = segments_size
        self.segments_overlap = segments_overlap
        self.decision_size = decision_size
        self.decision_overlap = decision_overlap
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Build and compile the model
        if data_arrangement == "4d":
            self.model = self.build_model_multi()
            # self.model.load_weights("initial_weights_4d.h5")
        elif model_size == "large":
            self.model = self.build_model_l()
        elif model_size == "medium":
            self.model = self.build_model_m()
        elif model_size == "small":
            self.model = self.build_model_s()

        optimizer = tf.keras.optimizers.Adam(0.001, 0.5)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def build_model_l(self):
        model = tf.keras.layers.Sequential()

        model.add(tf.keras.layers.Conv2D(64, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))

        img = tf.keras.layers.Input(shape=self.input_shape)
        validity = model(img)

        return tf.keras.layers.Model(img, validity)

    def build_model_m(self):
        model = tf.keras.layers.Sequential()

        model.add(tf.keras.layers.Conv2D(32, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))

        img = tf.keras.layers.Input(shape=self.input_shape)
        validity = model(img)

        return tf.keras.layers.Model(img, validity)

    def build_model_s(self):
        model = tf.keras.layers.Sequential()

        model.add(tf.keras.layers.Conv2D(16, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Conv2D(8, kernel_size=(1, 3), strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.9))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))

        img = tf.keras.layers.Input(shape=self.input_shape)
        validity = model(img)

        return tf.keras.layers.Model(img, validity)

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

    def train(self, epochs, X_train, y_train, X_valid, y_valid, restore_best=True, batch_size=128):
        # Change the labels from categorical to one-hot encoding
        y_train_onehot = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
        y_valid_onehot = np.asarray(pd.get_dummies(y_valid), dtype=np.int8)
        if restore_best:
            metric = ModelAnalyser(segment_size=self.segments_size,
                                   segment_overlap=self.segments_overlap,
                                   decision_size=self.decision_size,
                                   decision_overlap=self.decision_overlap,
                                   X=X_valid,
                                   y=y_valid)
            restoring_best = RestoringBest(metric=metric, monitor='mv_loss')
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
