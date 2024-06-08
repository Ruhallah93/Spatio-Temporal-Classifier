import pandas
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

## Multiple plots
datasets = ["ConfLongDemo_JSI_2022_04_01 23_36_21",
            "DriverIdentification_2022_04_03 19_06_33",
            "Healthy_Older_People_2022_04_03 02_35_53",
            "Motor_Failure_Time_2022_04_03 04_25_44",
            "Power_consumption_2022_04_02 01_05_58",
            "PRSA2017_2022_04_02 18_06_21",
            "RSSI_2022_04_02 18_37_24",
            "User_Identification_From_Walking_2022_04_02 18_44_15",
            "WISDM_2022_04_03 07_42_02"]
names = ["Kaluža et al. (2010)",
         "Eftekhari and Ghatee (2018)",
         "Torres et al. (2013)",
         "Scalabrini Sampaio et al. (2019)",
         "Salam and El Hibaoui (2018)",
         "Zhang et al. (2017)",
         "Dua and Graﬀ (2017)",
         "Casale et al. (2012)",
         "Weiss et al. (2019)"]
names = ["dataset [62]",
         "dataset [67]",
         "dataset [61]",
         "dataset [63]",
         "dataset [64]",
         "dataset [65]",
         "dataset [66]",
         "dataset [59]",
         "dataset [60]"]

fig = make_subplots(rows=3, cols=3, subplot_titles=names)
for i, dataset in enumerate(datasets):
    print(dataset)
    r = int(i / 3) + 1
    c = i % 3 + 1
    df = pandas.read_csv("files/" + dataset + ".csv")
    showlegend = i < 1

    fig.add_trace(
        go.Line(y=df.org_valid_loss, name="Validation IC", line=dict(color='firebrick', dash='dot'), showlegend=showlegend),
        row=r, col=c)
    fig.add_trace(go.Line(y=df.valid_loss, name="Validation OC", line=dict(color='red'), showlegend=showlegend), row=r,
                  col=c)
    fig.add_trace(go.Scatter(y=df.minp_org_valid_loss, name="Min Valid. IC", showlegend=showlegend, mode='markers',
                             marker=dict(color='firebrick'), marker_line_color="black", marker_line_width=1)
                  , row=r, col=c)
    fig.add_trace(go.Scatter(y=df.minp_valid_loss, name="Min Valid. OC", showlegend=showlegend, mode='markers',
                             marker=dict(color='red'), marker_line_color="black", marker_line_width=1)
                  , row=r, col=c)

    for j, diff in enumerate(df.diff_PR):
        if diff == 1:
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
    fig.update_yaxes(dict(  # attribures for y axis
        showline=True,
        showgrid=True,
        linecolor='black',
        tickfont=dict(
            family='Times New Roman'
        )
    ), title_text='Loss', row=r, col=c)

fig.update_layout(
    height=900, width=1200,
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
    plot_bgcolor='white'  # background color for the graph
)

fig.show()
