import pandas
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

## Multiple plots
datasets = ["CLD",
            "HOP",
            "UIFW",
            ]
names = ["CLD",
         "HOP",
         "UIFW",
         ]

fig = make_subplots(rows=1, cols=3, subplot_titles=names)
for i, dataset in enumerate(datasets):
    print(dataset)
    r = int(i / 3) + 1
    c = i % 3 + 1
    df = pandas.read_csv("user_identification_files/" + dataset + ".csv")
    showlegend = i < 1

    # model = np.polyfit(np.arange(df.valid_loss.shape[0]), df.valid_loss, 1)
    # y0 = model[0] * 0 + model[1]
    # y1 = model[0] * df.valid_loss.shape[0] + model[1]
    # fig.add_shape(go.layout.Shape(type='line', xref='x', yref='y', x0=0, y0=y0, x1=df.valid_loss.shape[0], y1=y1,
    #                               line=dict(color='gray', width=1))
    #               , row=r, col=c)
    #
    # model = np.polyfit(np.arange(df.org_valid_loss.shape[0]), df.org_valid_loss, 1)
    # y0 = model[0] * 0 + model[1]
    # y1 = model[0] * df.valid_loss.shape[0] + model[1]
    # fig.add_shape(go.layout.Shape(type='line', xref='x', yref='y', x0=0, y0=y0, x1=df.valid_loss.shape[0], y1=y1,
    #                               line=dict(color='gray', width=1))
    #               , row=r, col=c)

    fig.add_trace(
        go.Line(y=df.org_valid_loss, name="Inner Classifier", line=dict(color='blue'),
                showlegend=showlegend),
        row=r, col=c)
    fig.add_trace(go.Line(y=df.valid_loss, name="Proposed Fusion", line=dict(color='firebrick'), showlegend=showlegend),
                  row=r,
                  col=c)

    fig.add_trace(go.Scatter(y=[np.min(df.org_valid_loss)], x=[np.argmin(df.org_valid_loss)], name="Minimum",
                             showlegend=showlegend, mode='markers',
                             marker=dict(color='blue'), marker_line_color="cyan", marker_line_width=1)
                  , row=r, col=c)
    fig.add_trace(
        go.Scatter(y=[np.min(df.valid_loss)], x=[np.argmin(df.valid_loss)], name="Minimum", showlegend=showlegend,
                   mode='markers',
                   marker=dict(color='firebrick'), marker_line_color="red", marker_line_width=1)
        , row=r, col=c)

    for j, diff in enumerate(df.diff_PR):
        if diff < 0:
            fig.add_shape(go.layout.Shape(type='line', xref='x', yref='y', x0=j, y0=df.valid_loss.min(), x1=j,
                                          y1=df.org_valid_loss.max(), line=dict(color='gray', width=0.8))
                          , row=r, col=c)

    fig.update_xaxes(dict(  # attribures for x axis
        showline=True,
        showgrid=True,
        linecolor='black',
        tickfont=dict(
            family='Calibri'
        )
    ), title_text='Epoch', row=r, col=c)

    if i < 1:
        fig.update_yaxes(dict(  # attribures for y axis
            showline=True,
            showgrid=True,
            linecolor='black',
            # tickfont=dict(
            #     family='Times New Roman'
            # )
        ), title_text='$MS_{PBS}$', row=r, col=c)
    else:
        fig.update_yaxes(dict(  # attribures for y axis
            showline=True,
            showgrid=True,
            linecolor='black',
            tickfont=dict(
                family='Times New Roman'
            )
        ), row=r, col=c)

fig.update_layout(
    height=360, width=840,
    # xaxis=dict(  # attribures for x axis
    #     showline=True,
    #     showgrid=True,
    #     linecolor='black',
    #     tickfont=dict(
    #         family='Calibri'
    #     )
    # ),
    # yaxis=dict(  # attribures for y axis
    #     showline=True,
    #     showgrid=True,
    #     linecolor='black',
    #     tickfont=dict(
    #         family='Times New Roman'
    #     )
    # ),
    plot_bgcolor='white',  # background color for the graph

    legend=dict({'orientation': 'h', 'y': -0.5, 'xanchor': 'center', 'x': 0.5})
)

fig.show()
