import glob
import os
from datetime import timedelta
import pandas as pd
import numpy as np
from sklearn import preprocessing
from transformer import Transformer


class Dataset:

    def __init__(self, db_path, sample_rate, features, window_time, window_overlap_percentage, add_noise, noise_rate,
                 train_blocks: list, valid_blocks: list, test_blocks: list, data_length_time=-1):
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
        # method : "1d", "2d", "3d", "4d"
        segments_path = self.db_path + \
                        "segments/" + \
                        "method: " + str(method) + os.sep + \
                        "ws: " + str(self.window_size) + os.sep + \
                        "wo: " + str(self.window_overlap_size) + os.sep + \
                        "train: " + str(self.train_blocks) + os.sep + \
                        "valid: " + str(self.valid_blocks) + os.sep + \
                        "test: " + str(self.test_blocks) + os.sep
        if os.path.exists(segments_path + 'X_train.npy') \
                and os.path.exists(segments_path + 'y_train.npy') \
                and os.path.exists(segments_path + 'X_valid.npy') \
                and os.path.exists(segments_path + 'y_valid.npy') \
                and os.path.exists(segments_path + 'X_test.npy') \
                and os.path.exists(segments_path + 'y_test.npy'):
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
            label = int(os.path.basename(csv_path).split('.')[0])
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
        transformer = Transformer(segments_size=self.window_size, segments_overlap=self.window_overlap_size)
        self.X_train, self.y_train = transformer.transfer(self.n_train_dataset, self.features, method=method)
        self.X_valid, self.y_valid = transformer.transfer(self.n_valid_dataset, self.features, method=method)
        self.X_test, self.y_test = transformer.transfer(self.n_test_dataset, self.features, method=method)
