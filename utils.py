import os
import csv
import numpy as np
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def get_segments_a_decision_window(segment_size, segment_overlap, decision_size):
    segment_overlap_size = segment_size * segment_overlap
    return int((decision_size - segment_size) / (segment_size - segment_overlap_size) + 1)


def AveragingProbabilities(y_truth, y_prediction, s, r):
    df = []
    y_dw_truth = []
    c = y_prediction.shape[1]
    r = int(np.floor(s * r))
    for _id in np.unique(y_truth):
        subset = y_prediction[np.where(y_truth == _id)]
        n = subset.shape[0]
        o = int(np.floor((n - r) / (s - r)))
        for i in range(o):
            row = np.zeros(c)
            for j in range(s):
                row += subset[(i*s) + j]
            df.append(row / s)
            y_dw_truth.append(_id)
    df = np.array(df)
    y_dw_truth = np.asarray(pd.get_dummies(y_dw_truth), dtype=np.int8)
    return df, y_dw_truth


def MajorityVote(y_truth, y_prediction, s, r):
    df = []
    y_dw_truth = []
    c = y_prediction.shape[1]
    r = int(np.floor(s * r))
    # Make prior prediction to one-hot
    y_categorical_pred = np.zeros_like(y_prediction)
    y_categorical_pred[np.arange(len(y_prediction)), y_prediction.argmax(1)] = 1

    for _id in np.unique(y_truth):
        subset = y_categorical_pred[np.where(y_truth == _id)]
        n = subset.shape[0]
        o = int(np.floor((n - r) / (s - r)))
        for i in range(o):
            row = np.zeros(c)
            for j in range(s):
                row += subset[(i*s) + j]
            df.append(row / s)
            y_dw_truth.append(_id)
    df = np.array(df)
    y_dw_truth = np.asarray(pd.get_dummies(y_dw_truth), dtype=np.int8)
    return df, y_dw_truth


# p = MaximumScore(np.array([1, 1, 1, 1, 2, 2, 2, 2]), np.array(
#     [[0.1, 0.8], [1.1, 0.2], [0.1, 0.5], [0.1, 0.7], [0.8, 0.2], [0.9, 0.2], [0.3, 0.6], [0.2, 0.8]]), 2, 0.5)
# print(p)
# p = MajorityVote(np.array([1, 1, 1, 1, 2, 2, 2, 2]), np.array(
#     [[0.1, 0.8], [1.1, 0.2], [0.1, 0.5], [0.1, 0.7], [0.8, 0.2], [0.9, 0.2], [0.3, 0.6], [0.2, 0.8]]), 2, 0.5)
# print(p)

def analysis_model(y_pred, y_real_raw, segment_size, segment_overlap, decision_size, decision_overlap):
    result = {'Core': {}, 'MV': {}, 'MS': {}}
    loss_fn = MeanSquaredError()

    result['Core']['mse_loss'] = loss_fn(np.asarray(pd.get_dummies(y_real_raw), dtype=np.int8), y_pred).numpy()
    y_pred_arg = np.argmax(y_pred, axis=1)
    result['Core']['accuracy'] = accuracy_score(y_real_raw, y_pred_arg)
    result['Core']['precision'] = precision_score(y_real_raw, y_pred_arg, average='macro')
    result['Core']['recall'] = recall_score(y_real_raw, y_pred_arg, average='macro')
    result['Core']['f1'] = f1_score(y_real_raw, y_pred_arg, average='macro')

    segments_a_decision_window = get_segments_a_decision_window(segment_size, segment_overlap, decision_size)

    # Maximum Score
    y_pred_labels, y_real = AveragingProbabilities(y_truth=y_real_raw, y_prediction=y_pred,
                                                   s=segments_a_decision_window,
                                                   r=decision_overlap)

    result['MS']['mse_loss'] = loss_fn(y_real, y_pred_labels).numpy()
    temp = y_pred_labels.copy()
    y_pred_labels = np.zeros_like(temp)
    y_pred_labels[np.arange(len(temp)), temp.argmax(1)] = 1
    result['MS']['accuracy'] = accuracy_score(y_real, y_pred_labels)
    result['MS']['precision'] = precision_score(y_real, y_pred_labels, average='macro')
    result['MS']['recall'] = recall_score(y_real, y_pred_labels, average='macro')
    result['MS']['f1'] = f1_score(y_real, y_pred_labels, average='macro')

    # Majority Voting
    y_pred_labels, y_real = MajorityVote(y_truth=y_real_raw, y_prediction=y_pred,
                                         s=segments_a_decision_window,
                                         r=decision_overlap)

    result['MV']['mse_loss'] = loss_fn(y_real, y_pred_labels).numpy()
    temp = y_pred_labels.copy()
    y_pred_labels = np.zeros_like(temp)
    y_pred_labels[np.arange(len(temp)), temp.argmax(1)] = 1
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
