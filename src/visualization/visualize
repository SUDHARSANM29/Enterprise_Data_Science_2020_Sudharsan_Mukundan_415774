import pandas as pd
import numpy as np

import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State

import plotly.graph_objects as go

import os
print(os.getcwd())
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data

    This responsive dash board is created by automated data gathering, data transformations, filtering and machine learning
    to approximating the doubling time. The dashboard enables user to vsualize the confirmed cases and doubling rate of
    COVID-19 spread. It also visualization of filtered confirmed cases and doubling rate. The user can select the
    country/countires and the type of data to be visualized.
        '''),
    dcc.Markdown('''
    For filtering, the dashboard uses Savgol-filter with a window of 5 and polynomial of first degree.
    '''),


    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['Germany','US','Russia','Brazil','India'],                     #df_input_large['country'].unique(), # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        '''),


    dcc.Dropdown(
    id='doubling_time',
    options=[
        {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
        {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
        {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
        {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
    ],
    value='confirmed',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time', 'value')])
def update_figure(country_list,show_doubling):


    if 'DR' in show_doubling:
        my_yaxis='Approximated doubling rate over 3 days (larger numbers are better #stayathome, log-scale)'

    else:
        my_yaxis='Confirmed infected people (source johns hopkins csse, log-scale)'



    traces = []
    for each in country_list:

        df_plot=df_input_large[df_input_large['country']==each]

        if show_doubling=='doubling_rate_filtered':
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
       #print(show_doubling)


        traces.append(dict(x=df_plot.date,
                                y=df_plot[show_doubling],

                                mode='markers+lines',
                                opacity=0.9,
                                name=each,

                        )
                )

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },
                yaxis={'title':my_yaxis,
                        'type':'log'},
                updatemenus=[
                dict(

                    direction='right',
                    xanchor='left',
                    yanchor='top',
                    y=1.1,
                    x=0.015,
                    buttons=[
                        dict(label="Log",
                          method="relayout",
                          args=[{"yaxis.type": "log","yaxis.title":my_yaxis}]),
                        dict(label="Linear",
                          method="relayout",
                          args=[{"yaxis.type": "linear","yaxis.title":my_yaxis[0:-10]+" linear-scale)"}]),

                                  ])],
        )
    }

if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)
