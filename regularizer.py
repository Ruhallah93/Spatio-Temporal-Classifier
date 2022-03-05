import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np

from utils import MajorityVote, MaximumScore, get_segments_a_decision_window


class ModelAnalyser:

    def __init__(self, segment_size, segment_overlap, decision_size, decision_overlap, X, y):
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

    def measurement(self, model, monitor):
        """
        :param y_real: expected labels
        :param y_prediction: the outcomes of core
        :param monitor: '#METHOD_#MEASURE', #METHOD={'mv','ms'}, #MEASURE={'loss','accuracy','precision','recall','f1'}
        """

        monitor_method = monitor.split('_')[0]
        monitor_measure = monitor.split('_')[1]

        y_prediction = model.predict(self.X)

        if monitor_method == 'ms':
            y_pred_labels, y_dw_real = MaximumScore(self.y_real, y_prediction, self.segments_a_decision_window,
                                                    self.decision_overlap)

            if monitor_measure == 'loss':
                loss_fn = tf.keras.losses.get(model.loss)
                return loss_fn(y_dw_real, y_pred_labels)
            elif monitor_measure == 'accuracy':
                return accuracy_score(y_dw_real, y_pred_labels)
            elif monitor_measure == 'precision':
                return precision_score(y_dw_real, y_pred_labels, average='macro')
            elif monitor_measure == 'recall':
                return recall_score(y_dw_real, y_pred_labels, average='macro')
            elif monitor_measure == 'f1':
                return f1_score(y_dw_real, y_pred_labels, average='macro')

        if monitor_method == 'mv':
            y_pred_labels, y_dw_real = MajorityVote(self.y_real, y_prediction, self.segments_a_decision_window,
                                                    self.decision_overlap)
            if monitor_measure == 'loss':
                loss_fn = tf.keras.losses.get(model.loss)
                return np.mean(loss_fn(y_dw_real, y_pred_labels))
            elif monitor_measure == 'accuracy':
                return accuracy_score(y_dw_real, y_pred_labels)
            elif monitor_measure == 'precision':
                return precision_score(y_dw_real, y_pred_labels, average='macro')
            elif monitor_measure == 'recall':
                return recall_score(y_dw_real, y_pred_labels, average='macro')
            elif monitor_measure == 'f1':
                return f1_score(y_dw_real, y_pred_labels, average='macro')

        return 0


class RestoringBest(keras.callbacks.Callback):
    """
    :param monitor: '#METHOD_#MEASURE', #METHOD={'mv','ms'}, #MEASURE={'loss','accuracy','precision','recall','f1'}
    """

    def __init__(self, metric: ModelAnalyser, monitor):
        super(keras.callbacks.Callback, self).__init__()
        self.metric = metric
        self.monitor = monitor
        self.best = -np.Inf
        self.best_weights = None
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs={}):

        current = self.metric.measurement(model=self.model, monitor=self.monitor)
        print('epoch %d: \t %s: %f' % (epoch, self.monitor, current))

        monitor_measure = self.monitor.split('_')[1]

        if (monitor_measure != 'loss' and current > self.best) or (monitor_measure == 'loss' and current < self.best):
            self.best = current
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        if self.best_epoch > -1:
            print("Restoring model weights from the end of the %d epoch." % (self.best_epoch + 1))
            self.model.set_weights(self.best_weights)
