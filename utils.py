import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def get_segments_a_decision_window(segment_size, segment_overlap, decision_size):
    segment_overlap_size = segment_size * segment_overlap
    return int((decision_size - segment_size) / (segment_size - segment_overlap_size) + 1)


def MaximumScore(y_real, y_prediction, segments_a_decision_window, decision_overlap):
    df = []
    y_dw_real = []
    predict_segments = []
    for label_i in np.unique(y_real):
        subset = y_prediction[np.where(y_real == label_i)]
        for i in range(0, subset.shape[0], max(int(segments_a_decision_window * (1 - decision_overlap)), 1)):
            if i + segments_a_decision_window <= subset.shape[0]:
                row = np.zeros(y_prediction.shape[1])
                predict_segments.append(subset[i:i + segments_a_decision_window])
                for j in range(segments_a_decision_window):
                    row += subset[i + j]
                df.append(row)
                y_dw_real.append(label_i)
    df = np.array(df)
    y_pred_labels = np.zeros_like(df)
    y_pred_labels[np.arange(len(df)), df.argmax(1)] = 1
    return y_pred_labels, np.asarray(pd.get_dummies(y_dw_real), dtype=np.int8)


def MajorityVote(y_real, y_prediction, segments_a_decision_window, decision_overlap):
    df = []
    y_dw_real = []

    y_categorical_pred = np.zeros_like(y_prediction)
    y_categorical_pred[np.arange(len(y_prediction)), y_prediction.argmax(1)] = 1
    # y_categorical_pred = np.asarray(pd.get_dummies(np.argmax(y_prediction, axis=1)), dtype=np.int8)
    for class_i in np.unique(y_real):
        subset = y_categorical_pred[np.where(y_real == class_i)]
        for i in range(0, subset.shape[0], max(int(segments_a_decision_window * (1 - decision_overlap)), 1)):
            if i + segments_a_decision_window <= subset.shape[0]:
                row = np.zeros(y_categorical_pred.shape[1])
                for j in range(segments_a_decision_window):
                    row += subset[i + j]
                df.append(row)
                y_dw_real.append(class_i)
    df = np.array(df)
    y_pred_labels = np.zeros_like(df)
    y_pred_labels[np.arange(len(df)), df.argmax(1)] = 1
    return y_pred_labels, np.asarray(pd.get_dummies(y_dw_real), dtype=np.int8)


# p = MaximumScore(np.array([1, 1, 1, 1, 2, 2, 2, 2]), np.array(
#     [[0.1, 0.8], [1.1, 0.2], [0.1, 0.5], [0.1, 0.7], [0.8, 0.2], [0.9, 0.2], [0.3, 0.6], [0.2, 0.8]]), 2, 0.5)
# print(p)
# p = MajorityVote(np.array([1, 1, 1, 1, 2, 2, 2, 2]), np.array(
#     [[0.1, 0.8], [1.1, 0.2], [0.1, 0.5], [0.1, 0.7], [0.8, 0.2], [0.9, 0.2], [0.3, 0.6], [0.2, 0.8]]), 2, 0.5)
# print(p)

def analysis_model(y_pred, y_real_raw, segment_size, segment_overlap, decision_size, decision_overlap):
    result = {'Core': {}, 'MV': {}, 'MS': {}}

    y_pred_arg = np.argmax(y_pred, axis=1)
    result['Core']['accuracy'] = accuracy_score(y_real_raw, y_pred_arg)
    result['Core']['precision'] = precision_score(y_real_raw, y_pred_arg, average='macro')
    result['Core']['recall'] = recall_score(y_real_raw, y_pred_arg, average='macro')
    result['Core']['f1'] = f1_score(y_real_raw, y_pred_arg, average='macro')

    segments_a_decision_window = get_segments_a_decision_window(segment_size, segment_overlap, decision_size)

    # Maximum Score
    y_pred_labels, y_real = MaximumScore(y_real=y_real_raw, y_prediction=y_pred,
                                         segments_a_decision_window=segments_a_decision_window,
                                         decision_overlap=decision_overlap)

    result['MS']['accuracy'] = accuracy_score(y_real, y_pred_labels)
    result['MS']['precision'] = precision_score(y_real, y_pred_labels, average='macro')
    result['MS']['recall'] = recall_score(y_real, y_pred_labels, average='macro')
    result['MS']['f1'] = f1_score(y_real, y_pred_labels, average='macro')

    # Majority Voting
    y_pred_labels, y_real = MajorityVote(y_real=y_real_raw, y_prediction=y_pred,
                                         segments_a_decision_window=segments_a_decision_window,
                                         decision_overlap=decision_overlap)

    result['MV']['accuracy'] = accuracy_score(y_real, y_pred_labels)
    result['MV']['precision'] = precision_score(y_real, y_pred_labels, average='macro')
    result['MV']['recall'] = recall_score(y_real, y_pred_labels, average='macro')
    result['MV']['f1'] = f1_score(y_real, y_pred_labels, average='macro')

    return result


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


def cross_entropy3(p, q, ets=1e-15):
    a = [p[i] * np.log(q[i] + ets) for i in range(len(p))]
    return (-sum(a)) + np.std(np.array(a)[np.where(np.array(q) != 1)])
