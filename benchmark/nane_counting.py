import os
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

from benchmark.dataset_information import problems

log_dir = ""
counter = 0

drop_list = []
if os.path.exists(log_dir + "statistics.csv"):
    saved_statistics = pd.read_csv(log_dir + "statistics.csv")
    for index, row in saved_statistics.iterrows():
        if np.isnan(row['mse_loss_mean']):
            counter += 1
            print(counter,
                  "dataset: " + row["dataset"],
                  "model: " + row["inner_classifier"],
                  "dl: " + row["decision_time"],
                  "wl:" + row["segments_time"],
                  "run time:" + row["running_time"],
                  "is null!")
            drop_list.append(index)

saved_statistics = saved_statistics.drop(drop_list)
saved_statistics.to_csv("non_nan_statistics.csv", index=False)
