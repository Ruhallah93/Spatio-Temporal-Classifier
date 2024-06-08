import time
import datetime
import pandas
import plotly.express as px
import numpy as np
from benchmark.dataset_information import problems
import os


def string_seconds(x):
    x = time.strptime(x, '%H:%M:%S')
    return datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()


data = pandas.read_csv("non_dup_statistics.csv")
datasets = np.unique(data.dataset)

classifiers = ["KNN", "MLP", "LR", "RF", "SVM", "CNN_L"]
# classifiers = ["MLP", "CNN_L"]
classifiers_title = ["KNN", "MLP", "Linear Regression", "Random Forest", "SVM", "CNN"]
# classifiers_title = ["MLP", "CNN"]
for dataset in datasets:
    for classifier, ctitle in zip(classifiers, classifiers_title):
        data_ds = data[data.dataset == dataset]
        data_ds = data_ds[data_ds.inner_classifier == classifier]

        sr = problems[dataset]['sample_rate']
        segments_times = data_ds.segments_time.to_numpy()
        decision_times = data_ds.decision_time.to_numpy()
        data_ds["w"] = np.array([string_seconds(segments_times[j]) for j in range(len(segments_times))])

        max_id = data_ds.f1_mean.argmax()
        # max_id = data_ds.accuracy_mean.argmax()
        print(dataset, ctitle,
              int(data_ds.w.iloc[max_id]),
              str(round(data_ds.f1_mean.iloc[max_id] * 100, 2)) +
              " ($\pm$" +
              str(round(data_ds.f1_std.iloc[max_id], 2)) + ")")

        fig = px.scatter(data_ds, x="w", y="f1_mean", color="f1_mean", title=ctitle,
                         range_color=(0, 1))
        fig.update_traces(marker={'size': 8, 'opacity': 0.7})

        labels = dict(
            xaxis_title='w',
            yaxis_title='Mean f1-score')
        fig.update_layout(height=500, width=500, title_x=0.5,
                          showlegend=False,
                          xaxis_title='w',
                          yaxis_title='Mean f1-score',
                          yaxis=dict(nticks=20, range=[0, 1]))
        fig.update_coloraxes(showscale=False)

        if not os.path.exists("images/" + dataset + "/"):
            os.mkdir("images/" + dataset + "/")
        # fig.show()
        # fig.write_image("images/" + dataset + "/" + ctitle + ".png")
