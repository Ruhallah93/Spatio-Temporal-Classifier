from datetime import datetime
import numpy as np
import argparse
from dataset import Dataset
from models import ConvNet
from transformer import Transformer
from utils import analysis_model, save_result
from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')


def train_model(dataset: Dataset, conv_net: ConvNet, restore_best, epochs, batch_size):
    conv_net.train(epochs=epochs,
                   X_train=dataset.X_train,
                   y_train=dataset.y_train,
                   X_valid=dataset.X_valid,
                   y_valid=dataset.y_valid,
                   restore_best=restore_best,
                   batch_size=batch_size)

    y_test_prediction = conv_net.model.predict(dataset.X_test)
    result_test = analysis_model(y_pred=y_test_prediction,
                                 y_real_raw=dataset.y_test,
                                 segment_size=conv_net.segments_size,
                                 segment_overlap=conv_net.segments_overlap,
                                 decision_size=conv_net.decision_size,
                                 decision_overlap=conv_net.decision_overlap)

    print('Test(CNN):%5.2f Test(MV):%5.2f ' % (result_test['Core']['accuracy'], result_test['MV']['accuracy']))

    return result_test


def hv_block_analyzer(db_path, sample_rate, features, n_classes, data_arrangement, noise_rate, segments_time,
                      segments_overlap, convNet: ConvNet, epochs, batch_size, restore_best, data_length_time,
                      n_hv_block, n_train_hv_block, n_valid_hv_block, n_test_hv_block, hv_moving_step=1):
    """
    :param db_path: the address of dataset directory
    :param sample_rate: the sampling rate of signals
    :param features: the signals of original data
    :param n_classes: the number of classes
    :param data_arrangement: the shape of data {'1d','2d','3d','4d'}
    :param noise_rate: the rate of noises injected to test data
    :param segments_time: the length of each segment in seconds.
    :param segments_overlap: the overlap of each segment
    :param convNet: the neural network
    :param epochs: the number of training epochs
    :param batch_size: the number of segments in each batch
    :param restore_best: for using regularization {'True', 'False}
    :param n_hv_block: the number of all hv blocks
    :param n_train_hv_block: the number of hv blocks to train network
    :param n_valid_hv_block: the number of hv blocks to validate network
    :param n_test_hv_block: the number of hv blocks to test network
    :param hv_moving_step: the number of movement of test and validation blocks in each iteration
    :return:
    """

    statistics = {}
    add_noise = noise_rate < 100

    # Create hv blocks
    data_blocks = [i for i in range(n_hv_block)]
    n_vt = (n_valid_hv_block + n_test_hv_block)
    n_iteration = int((n_hv_block - n_vt) / hv_moving_step)
    for i in range(n_iteration + 1):
        print('iteration: %d/%d' % (i + 1, n_iteration + 1))

        training_container = data_blocks[0:i] + data_blocks[i + n_vt:n_hv_block]
        train_blocks = training_container[:n_train_hv_block]
        valid_blocks = data_blocks[i: i + n_valid_hv_block]
        test_blocks = data_blocks[i + n_valid_hv_block: i + n_vt]

        dataset = Dataset(db_path,
                          sample_rate,
                          features=features,
                          window_time=segments_time,
                          window_overlap_percentage=segments_overlap,
                          add_noise=add_noise,
                          noise_rate=noise_rate,
                          train_blocks=train_blocks,
                          valid_blocks=valid_blocks,
                          test_blocks=test_blocks,
                          data_length_time=data_length_time)

        dataset.load_data(n_classes=n_classes, method=data_arrangement)
        result = train_model(dataset=dataset, conv_net=convNet, restore_best=restore_best, epochs=epochs,
                             batch_size=batch_size)

        for key in result.keys():
            if not key in statistics:
                statistics[key] = {}
            for inner_key in result[key].keys():
                if not inner_key in statistics[key]:
                    statistics[key][inner_key] = []
                statistics[key][inner_key].append(result[key][inner_key])
        print(statistics)
    return statistics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ['x-accelerometer', 'y-accelerometer', 'z-accelerometer', 'x-gyroscope', 'y-gyroscope', 'z-gyroscope']
    parser.add_argument('--dataset', type=str, default='datasets/Motor_Failure_Time/',
                        help='the address of dataset directory')
    parser.add_argument('--n_classes', type=int, default=3, help='the number of classes')
    parser.add_argument('--features', nargs='+', type=str,
                        default=['pctid', 'x', 'y', 'z'], help='the signals of original data')
    parser.add_argument('--sample_rate', type=int, default=18, help='the sampling rate of signals')
    parser.add_argument('--noise_rate', type=int, default=100,
                        help='the rate of noises injected to test data, over 100 means false')
    parser.add_argument('--epochs', type=int, default=20, help='the number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='the number of segments in each batch')
    parser.add_argument('--restore_best', type=int, default=1, help='for using regularization, 0 is False other True')
    parser.add_argument('--data_arrangement', type=str, default='4d', help='the shape of data {1d,2d,3d,4d}')
    parser.add_argument('--model_size', type=str, default='large', help='large|medium|small')
    parser.add_argument('--data_length_time', type=int, default=-1, help='the data length for each class,-1 means all')
    parser.add_argument('--n_hv_block', type=int, default=15, help='the number of all hv blocks')
    parser.add_argument('--n_train_hv_block', type=int, default=9, help='the number of hv blocks to train network')
    parser.add_argument('--n_valid_hv_block', type=int, default=2, help='the number of hv blocks to validate network')
    parser.add_argument('--n_test_hv_block', type=int, default=4, help='the number of hv blocks to test network')
    parser.add_argument('--hv_moving_step', type=int, default=1, help='moving test blocks rate in each iteration')
    parser.add_argument('--segments_times', nargs='+', type=int, default=[4,5], help='in seconds')
    parser.add_argument('--segments_overlaps', nargs='+', type=float, default=[0.75], help='percentage in [0,1]')
    parser.add_argument('--decision_times', nargs='+', type=int, default=[60 * 1], help='in seconds')
    parser.add_argument('--decision_overlaps', nargs='+', type=float, default=[0.75], help='percentage in [0,1]')
    opt = parser.parse_args()

    log_dir = "logs/"

    for segments_time in opt.segments_times:
        for segments_overlap in opt.segments_overlaps:
            for decision_time in opt.decision_times:
                for decision_overlap in opt.decision_overlaps:
                    data_shape = Transformer.data_shape(method=opt.data_arrangement,
                                                        n_features=len(opt.features),
                                                        segments_size=segments_time * opt.sample_rate)
                    convNet = ConvNet(data_arrangement=opt.data_arrangement,
                                      model_size=opt.model_size,
                                      input_shape=(data_shape[-3], data_shape[-2], data_shape[-1]),
                                      classes=opt.n_classes,
                                      n_features=len(opt.features),
                                      segments_size=segments_time * opt.sample_rate,
                                      segments_overlap=segments_overlap,
                                      decision_size=decision_time * opt.sample_rate,
                                      decision_overlap=decision_overlap)

                    # cross-validation
                    start = datetime.now()
                    statistics = hv_block_analyzer(db_path=opt.dataset,
                                                   sample_rate=opt.sample_rate,
                                                   features=opt.features,
                                                   n_classes=opt.n_classes,
                                                   data_arrangement=opt.data_arrangement,
                                                   noise_rate=opt.noise_rate,
                                                   segments_time=segments_time,
                                                   segments_overlap=segments_overlap,
                                                   convNet=convNet,
                                                   epochs=opt.epochs,
                                                   batch_size=opt.batch_size,
                                                   restore_best=opt.restore_best != 0,
                                                   data_length_time=opt.data_length_time,
                                                   n_hv_block=opt.n_hv_block,
                                                   n_train_hv_block=opt.n_train_hv_block,
                                                   n_valid_hv_block=opt.n_valid_hv_block,
                                                   n_test_hv_block=opt.n_test_hv_block,
                                                   hv_moving_step=opt.hv_moving_step)
                    end = datetime.now()
                    running_time = end - start

                    # Summarizing the results of cross-validation
                    data = opt.__dict__
                    data['datetime'] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
                    data['running_time'] = str(running_time.seconds) + " seconds"
                    data['n_params'] = convNet.model.count_params()
                    data['segments_times'] = timedelta(seconds=int(segments_time))
                    data['segments_overlaps'] = segments_overlap
                    data['decision_times'] = timedelta(seconds=int(decision_time))
                    data['decision_overlaps'] = decision_overlap
                    statistics_summary = {}
                    for key in statistics.keys():
                        for inner_key in statistics[key].keys():
                            statistics_summary[key + '_' + inner_key + '_mean'] = np.average(statistics[key][inner_key])
                            statistics_summary[key + '_' + inner_key + '_std'] = np.std(statistics[key][inner_key])
                            statistics_summary[key + '_' + inner_key + '_max'] = np.max(statistics[key][inner_key])
                            statistics_summary[key + '_' + inner_key + '_min'] = np.min(statistics[key][inner_key])
                    data.update(statistics_summary)

                    # Save information
                    save_result(log_dir=log_dir, data=data)
