import os
from datetime import datetime
from datetime import timedelta
import pandas as pd

from benchmark.dataset_information import problems

log_dir = ""

drop_list = []
if os.path.exists(log_dir + "non_nan_statistics.csv"):
    is_investigated = False
    saved_statistics = pd.read_csv(log_dir + "non_nan_statistics.csv")
    for i, row1 in saved_statistics.iterrows():
        for j, row2 in saved_statistics.iterrows():
            if (i < j and
                    row1['dataset'] == row2['dataset'] and
                    row1['inner_classifier'] == row2['inner_classifier'] and
                    row1['decision_time'] == row2['decision_time'] and
                    row1['segments_time'] == row2['segments_time']):
                print(row1['dataset'],
                      row1['inner_classifier'],
                      row1['decision_time'],
                      row1['segments_time'])
                drop_list.append(i)

saved_statistics = saved_statistics.drop(drop_list)
saved_statistics.to_csv("non_dup_statistics.csv", index=False)
