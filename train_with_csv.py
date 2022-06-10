import pandas as pd
import numpy as np
from models import models
from datetime import datetime
from datetime import timedelta
from utils import analysis_model, save_result
from train import h_block_analyzer
import json

if __name__ == '__main__':
    args = pd.read_csv("datasets/args2.csv")

    for dataset in args.dataset:
        log_dir = "logs/"
        opt = args[args.dataset == dataset].iloc[0]

        opt.features = eval(opt.features)
        opt.segments_times = eval(opt.segments_times)
        opt.segments_overlaps = eval(opt.segments_overlaps)
        opt.decision_times = eval(opt.decision_times)
        opt.decision_overlaps = eval(opt.decision_overlaps)
        opt.model = eval(opt.model)

        segments_times = opt.segments_times
        segments_overlaps = opt.segments_overlaps
        decision_times = opt.decision_times
        decision_overlaps = opt.decision_overlaps

        for model in opt.model:
            for segments_time, decision_time in zip(segments_times, decision_times):
                for segments_overlap in segments_overlaps:
                    for decision_overlap in decision_overlaps:
                        classifier = getattr(models, model)(classes=opt.n_classes,
                                                            n_features=len(opt.features),
                                                            segments_size=segments_time * opt.sample_rate,
                                                            segments_overlap=segments_overlap,
                                                            decision_size=decision_time * opt.sample_rate,
                                                            decision_overlap=decision_overlap)

                        # cross-validation
                        start = datetime.now()
                        statistics = h_block_analyzer(db_path=opt.dataset,
                                                      sample_rate=opt.sample_rate,
                                                      features=opt.features,
                                                      n_classes=opt.n_classes,
                                                      noise_rate=opt.noise_rate,
                                                      segments_time=segments_time,
                                                      segments_overlap=segments_overlap,
                                                      classifier=classifier,
                                                      epochs=opt.epochs,
                                                      batch_size=opt.batch_size,
                                                      restore_best=opt.restore_best != 0,
                                                      data_length_time=opt.data_length_time,
                                                      n_h_block=opt.n_h_block,
                                                      n_train_h_block=opt.n_train_h_block,
                                                      n_valid_h_block=opt.n_valid_h_block,
                                                      n_test_h_block=opt.n_test_h_block,
                                                      h_moving_step=opt.h_moving_step)
                        end = datetime.now()
                        running_time = end - start

                        # Summarizing the results of cross-validation
                        data = opt.to_dict()
                        data['inner_classifier'] = str(model)
                        data['datetime'] = datetime.now().strftime("%Y:%m:%d %H:%M:%S")
                        data['running_time'] = str(running_time.seconds) + " seconds"
                        data['n_params'] = classifier.count_params()
                        data['segments_times'] = timedelta(seconds=int(segments_time))
                        data['segments_overlaps'] = segments_overlap
                        data['decision_times'] = timedelta(seconds=int(decision_time))
                        data['decision_overlaps'] = decision_overlap
                        statistics_summary = {}
                        for key in statistics.keys():
                            for inner_key in statistics[key].keys():
                                statistics_summary[key + '_' + inner_key + '_mean'] = np.average(
                                    statistics[key][inner_key])
                                statistics_summary[key + '_' + inner_key + '_std'] = np.std(
                                    statistics[key][inner_key])
                                statistics_summary[key + '_' + inner_key + '_max'] = np.max(
                                    statistics[key][inner_key])
                                statistics_summary[key + '_' + inner_key + '_min'] = np.min(
                                    statistics[key][inner_key])
                        data.update(statistics_summary)

                        # Save information
                        save_result(log_dir=log_dir, data=data)
