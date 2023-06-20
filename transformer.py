import numpy as np


class Transformer:

    def __init__(self, segments_size=90, segments_overlap=45, sampling=2):
        self.segments_size = segments_size
        self.overlap = segments_overlap
        self.sampling = sampling

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
        return segments_dataset, labels

    @staticmethod
    def data_shape(method, n_features, segments_size):
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
        return ()

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
            start += (self.segments_size - self.overlap)

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
