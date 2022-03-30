import pandas
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np

from utils import MajorityVote, AveragingProbabilities, get_segments_a_decision_window


class ModelAnalyser:

    def __init__(self, segment_size, segment_overlap, decision_size, decision_overlap, X, y, X_t, y_t):
        """
        :param segment_size: for calculating number of segments in a decision window.
        :param segment_overlap: for calculating number of segments in a decision window.
        :param decision_size:
        :param decision_overlap:
        :param X:
        :param y:
        """
        self.segments_a_decision_window = get_segments_a_decision_window(segment_size, segment_overlap, decision_size)
        self.decision_overlap = decision_overlap
        self.X = X
        self.y_real = y
        self.X_t = X_t
        self.y_t_real = y_t

    def measurement(self, X, y_real, model, monitor):
        """
        :param y_real: expected labels
        :param y_prediction: the outcomes of core
        :param monitor: '#METHOD_#MEASURE', #METHOD={'mv','ms'}, #MEASURE={'loss','accuracy','precision','recall','f1'}
        """

        monitor_method = monitor.split('_')[0]
        monitor_measure = monitor.split('_')[1]

        y_prediction = model.predict(X)

        if monitor_method == 'ms':
            y_pred_labels, y_dw_real = AveragingProbabilities(y_real, y_prediction,
                                                              self.segments_a_decision_window,
                                                              self.decision_overlap)

            y_pred_one_hot = np.zeros_like(y_pred_labels)
            y_pred_one_hot[np.arange(len(y_pred_one_hot)), y_pred_labels.argmax(1)] = 1
            if monitor_measure == 'loss':
                # loss_fn = tf.keras.losses.get(model.loss)
                # return np.mean(loss_fn(y_dw_real, y_pred_labels))
                loss_fn = tf.keras.losses.MeanSquaredError()
                return loss_fn(y_dw_real, y_pred_labels).numpy()
            elif monitor_measure == 'accuracy':
                return accuracy_score(y_dw_real, y_pred_one_hot)
            elif monitor_measure == 'precision':
                return precision_score(y_dw_real, y_pred_one_hot, average='macro')
            elif monitor_measure == 'recall':
                return recall_score(y_dw_real, y_pred_one_hot, average='macro')
            elif monitor_measure == 'f1':
                return f1_score(y_dw_real, y_pred_one_hot, average='macro')

        if monitor_method == 'mv':
            y_pred_labels, y_dw_real = MajorityVote(y_real, y_prediction, self.segments_a_decision_window,
                                                    self.decision_overlap)

            y_pred_one_hot = np.zeros_like(y_pred_labels)
            y_pred_one_hot[np.arange(len(y_pred_one_hot)), y_pred_labels.argmax(1)] = 1
            if monitor_measure == 'loss':
                # loss_fn = tf.keras.losses.get(model.loss)
                # return np.mean(loss_fn(y_dw_real, y_pred_labels))
                loss_fn = tf.keras.losses.MeanSquaredError()
                return loss_fn(y_dw_real, y_pred_labels).numpy()
            elif monitor_measure == 'accuracy':
                return accuracy_score(y_dw_real, y_pred_one_hot)
            elif monitor_measure == 'precision':
                return precision_score(y_dw_real, y_pred_one_hot, average='macro')
            elif monitor_measure == 'recall':
                return recall_score(y_dw_real, y_pred_one_hot, average='macro')
            elif monitor_measure == 'f1':
                return f1_score(y_dw_real, y_pred_one_hot, average='macro')

        return 0


class RestoringBest(keras.callbacks.Callback):
    """
    :param monitor: '#METHOD_#MEASURE', #METHOD={'mv','ms'}, #MEASURE={'loss','accuracy','precision','recall','f1'}
    """

    def __init__(self, metric: ModelAnalyser, monitor):
        super(keras.callbacks.Callback, self).__init__()
        self.metric = metric
        self.monitor = monitor
        if monitor.split('_')[1] == 'loss':
            self.best = np.Inf
        else:
            self.best = -np.Inf
        self.best_weights = None
        self.best_epoch = -1
        self.history = []

    def on_epoch_end(self, epoch, logs={}):

        current = self.metric.measurement(X=self.metric.X, y_real=self.metric.y_real, model=self.model,
                                          monitor=self.monitor)
        accuracy = self.metric.measurement(X=self.metric.X, y_real=self.metric.y_real, model=self.model,
                                           monitor="ms_accuracy")
        f1 = self.metric.measurement(X=self.metric.X, y_real=self.metric.y_real, model=self.model, monitor="ms_f1")

        loss_t = self.metric.measurement(X=self.metric.X_t, y_real=self.metric.y_t_real, model=self.model,
                                         monitor=self.monitor)
        accuracy_t = self.metric.measurement(X=self.metric.X_t, y_real=self.metric.y_t_real, model=self.model,
                                             monitor="ms_accuracy")
        f1_t = self.metric.measurement(X=self.metric.X_t, y_real=self.metric.y_t_real, model=self.model,
                                       monitor="ms_f1")

        self.history.append([loss_t, accuracy_t, f1_t, current, accuracy, f1])
        print(
            'epoch %d: \t %s: %f \t %s: %f \t %s: %f' % (epoch, self.monitor, current, "accuracy", accuracy, "f1", f1))

        monitor_measure = self.monitor.split('_')[1]

        if (monitor_measure != 'loss' and current > self.best) or (monitor_measure == 'loss' and current < self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        h = pandas.read_csv("history/history.csv")
        h = pandas.concat([h, pandas.DataFrame(self.history,
                                  columns=['train_loss', 'train_acc', 'train_f1', 'valid_loss', 'valid_acc',
                                           'valid_f1'])])
        h.to_csv("history/history.csv", index=False)
        if self.best_epoch > -1:
            print("Restoring model weights from the end of the %d epoch." % (self.best_epoch + 1))
            self.model.set_weights(self.best_weights)
