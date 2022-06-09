import pandas
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

data = pandas.read_csv("benchmark.csv")

datasets = ["\cite{kaluvza2010agent}",
            "\cite{eftekhari2018hybrid}",
            "\cite{torres2013sensor}",
            "\cite{scalabrini2019prediction}",
            "\cite{salam2018comparison}",
            "\cite{zhang2017cautionary}",
            "\cite{Dua:2019}",
            "\cite{casale2012personalization}",
            "\cite{weiss2019smartphone}"]
names = ["Kaluža et al. (2010)",
         "Eftekhari and Ghatee (2018)",
         "Torres et al. (2013)",
         "Scalabrini Sampaio et al. (2019)",
         "Salam and El Hibaoui (2018)",
         "Zhang et al. (2017)",
         "Dua and Graﬀ (2017)",
         "Casale et al. (2012)",
         "Weiss et al. (2019)"]

fig = make_subplots(rows=3, cols=3, subplot_titles=names)
for i, dataset in enumerate(datasets):
    r = int(i / 3) + 1
    c = i % 3 + 1
    showlegend = i < 1

    df = data[data.Datasets == dataset]
    ours = df[df.restore_best == 1]
    ours = ours.sort_values(by=['Datasets'])
    other = df[df.restore_best == 0]
    other = other.sort_values(by=['Datasets'])

    fig.add_trace(
        go.Bar(name='Loss \n(w = s)', x=other.Classifiers, y=other.loss, marker_color='indianred', showlegend=showlegend),
        row=r, col=c)
    fig.add_trace(go.Bar(name='Loss (w < s)', x=ours.Classifiers, y=ours.loss, marker_color='green',
                         showlegend=showlegend), row=r, col=c)

    fig.add_trace(
        go.Bar(name='F-1 Score (w = s)', x=other.Classifiers, y=other.f1, marker_color='lightsalmon', showlegend=showlegend),
        row=r, col=c)
    fig.add_trace(
        go.Bar(name='F-1 Score (w < s)', x=ours.Classifiers, y=ours.f1, marker_color='yellowgreen',
               showlegend=showlegend),
        row=r, col=c)

    fig.update_xaxes(dict(  # attribures for x axis
        showline=True,
        showgrid=True,
        linecolor='black',
        tickfont=dict(
            family='Calibri'
        )
    ), title_text='Algorithms', row=r, col=c)
    fig.update_yaxes(dict(  # attribures for y axis
        showline=True,
        showgrid=True,
        linecolor='black',
        tickfont=dict(
            family='Times New Roman'
        )
    ), range=[0, 1], nticks=10, title_text='', row=r, col=c)

fig.update_layout(
    height=900, width=1200,
    plot_bgcolor='white'  # background color for the graph
)

fig.show()
