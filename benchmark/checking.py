import os
from datetime import datetime
from datetime import timedelta
import pandas as pd

from benchmark.dataset_information import problems

models = ['GRU', 'LSTM', 'BiLSTM']

counter = 0
for problem in ['Power_consumption', 'RSSI', 'User_Identification_From_Walking', 'WISDM', 'PRSA2017',
                'Motor_Failure_Time', 'DriverIdentification', 'ConfLongDemo_JSI', 'Healthy_Older_People']:
    dataset = problems[problem]['dataset']
    n_classes = problems[problem]['n_classes']
    features = problems[problem]['features']
    sample_rate = problems[problem]['sample_rate']
    data_length_time = problems[problem]['data_length_time']
    n_h_block = problems[problem]['n_h_block']
    n_train_h_block = problems[problem]['n_train_h_block']
    n_valid_h_block = problems[problem]['n_valid_h_block']
    n_test_h_block = problems[problem]['n_test_h_block']
    h_moving_step = problems[problem]['h_moving_step']
    decision_times = problems[problem]['decision_times']
    segments_times = problems[problem]['segments_times']

    #     log_dir = f"./comparisons/log/{problem}/recurrent/"
    log_dir = ""

    for model in models:
        for decision_time in decision_times:
            for decision_overlap in [0.0]:
                for segments_time in segments_times:
                    for segments_overlap in [0.75]:
                        if float(decision_time * 0.75) < float(segments_time):
                            continue

                        if os.path.exists(log_dir + "non_dup_statistics.csv"):
                            is_investigated = False
                            saved_statistics = pd.read_csv(log_dir + "non_dup_statistics.csv")
                            for index, row in saved_statistics.iterrows():
                                try:
                                    if row['dataset'] == str(problem) and row['inner_classifier'] == str(model):
                                        t1 = datetime.strptime(row['segments_time'], "%H:%M:%S")
                                        t2 = datetime.strptime(row['decision_time'], "%H:%M:%S")
                                        td1 = timedelta(hours=t1.hour, minutes=t1.minute, seconds=t1.second)
                                        td2 = timedelta(hours=t2.hour, minutes=t2.minute, seconds=t2.second)
                                        if (td1.total_seconds() == int(segments_time)
                                                and td2.total_seconds() == int(decision_time)):
                                            is_investigated = True
                                            break
                                except:
                                    print(row)
                            if not is_investigated:
                                counter += 1
                                print(counter, "dataset: " + problem + " " +
                                      "model: " + model + " " +
                                      "dl: " + str(str(timedelta(seconds=int(decision_time))) +
                                                   " and wl:"
                                                   + str(timedelta(seconds=int(segments_time)))
                                                   + " is not investigated!"))
