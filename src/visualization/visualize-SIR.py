import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go

import os
print(os.getcwd())
df_input_large=pd.read_csv('data/processed/COVID_final_set.csv',sep=';')


fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([

    dcc.Markdown('''
    #  Dynamic Dashboard for SIR curve fitting

    This is a dynamic dashboard that enables users to adjust the Infection rate, Recovery rate, Initial Time Period,
    Introduction of measures time period, hold period, and relax period. By adjusting these parameters, the SIR curve
    can be fit to the chosen country's data. For easy adjustment, the cumulative infection rate of the country is also
    displayed. Further, different timelines are displayed in the background with different shades..

    '''),
    dcc.Markdown('''
    Assumptions:     '''),
    dcc.Markdown('''
    1) The data is taken from John Hopkin's repository. The data does not include the population size of each country.
    Therefore, an assumption is made regarding the total population size of a country. Since the global infected
    percentage is 3%* it is used to calculate the population size of the country. Though it is not a good measure,
    to visualize the dashboard working, it will be helpful.
    '''),
    dcc.Markdown('''
    2) The minimum starting number of infected people is set to 50 to have smooth curve.
    '''),
    dcc.Markdown('''

    Note: The data from JohnHopkin's repository is cumulative infected people data. In SIR, the number of active
    infected people is calculated.
       '''),
    dcc.Markdown('''

    *- World population data from worldometer (https://www.worldometers.info/world-population/) &
    Total infected people data from John Hopkin's dashboard are taken for (https://coronavirus.jhu.edu/map.html)
    calculation.
       '''),
    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value=['Germany'], # which are pre-selected
        multi=True
    ),

    dcc.Markdown('''
        ## Select the parameter to be visualized (Susceptible, Infected, Recovered, or SIR)
        '''),

    dcc.Dropdown(
    id='SIR',
    options=[
        {'label': 'Susceptible ', 'value': 'susceptible'},
        {'label': 'Infected', 'value': 'infected'},
        {'label': 'Recovered', 'value': 'recovered'},
        {'label': 'S-I-R', 'value': 's-i-r'},
    ],
    value='infected',
    multi=False
    ),


    dcc.Markdown('''
        ## Select following values for dynamically adjusting the SIR curve
        '''),
     dcc.Markdown('''
         ---Infection_Rate_Max - - - - - Infection_Rate_Min - - - - - - -Recovery_Rate - - - - - - - - Initial_Period - - - - - - - - - - - Intro_measures - - - - - - - - - - Hold - - - - - - - - - -  - - - - - - Relax
        '''),
    dcc.Input(id="infect_rate_max", type="text", placeholder="Infection Rate Max",
                 min=0, max=1, step=1000,value='0.4',name='Infect rate',debounce=True),
    dcc.Input(id="infect_rate_min", type="text", placeholder="Infection Rate Min",
                 min=0, max=1, step=1000,value='0.4',debounce=True),
    dcc.Input(id="recover_rate", type="text", placeholder="Recovery Rate",
                 min=0, max=1, step=1000,value='0.4',debounce=True),
    dcc.Input(id="init_per", type="text", placeholder="Initial Period",
                 min=0, max=1, step=1000,value='20',debounce=True),
    dcc.Input(id="intro_meas", type="text", placeholder="Introduction of measure",
                 min=0, max=1, step=1000,value='20',debounce=True),
    dcc.Input(id="hold", type="text", placeholder="Hold",
                 min=0, max=1, step=1000,value='20',debounce=True),
    dcc.Input(id="relax", type="text", placeholder="Relax",
                 min=0, max=1, step=1000,value='20',debounce=True),


    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('infect_rate_max','value'),
    Input('infect_rate_min','value'),
    Input('recover_rate','value'),
    Input('init_per','value'),
    Input('intro_meas','value'),
    Input('hold','value'),
    Input('relax','value'),
    Input('SIR','value')])
def update_figure(country_list,infect_rate_max,infect_rate_min,recover_rate,init_per,intro_meas,hold,relax,sir,):

    t_initial=1
    t_intro_measures=1
    t_hold=1
    t_relax=1
    N0=20
    df_plot=df_input_large
    traces=[]
    for each in country_list:
        df_plot=df_input_large[df_input_large['country']==each]
        df_plot=df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()
        df_plot=df_plot['confirmed'][df_plot['confirmed']>50].reset_index(drop=True)
        traces.append(dict(x=np.arange(len(df_plot)),
                               y=df_plot,
                               type='bar',
                               visible=True,
                               opacity=0.9,
                               name=each,

                       )
               )
        N0=np.array(df_plot)[-1]/0.03
    if ((len(country_list)>0)&(len(infect_rate_max)>0)&(len(infect_rate_min)>0)&(len(recover_rate)>0)&(len(init_per)>0)&(len(intro_meas)>0)&(len(hold)>0)&(len(relax)>0)):
        infect_rate_max=float(infect_rate_max)
        infect_rate_min=float(infect_rate_min)
        recover_rate=float(recover_rate)
        t_initial=int(init_per)
        t_intro_measures=int(intro_meas)
        t_hold=int(hold)
        t_relax=int(relax)




        beta_max=infect_rate_max
        beta_min=infect_rate_min
        gamma=recover_rate




        N0=np.array(df_plot)[-1]/0.03
        I0=df_plot[0]
        S0=N0-I0
        R0=0

        pd_beta=np.concatenate((np.array(t_initial*[beta_max]),
                               np.linspace(beta_max,beta_min,t_intro_measures),
                               np.array(t_hold*[beta_min]),
                               np.linspace(beta_min,beta_max,t_relax),
                               ))
        SIR=np.array([S0,I0,R0])
        propagation_rates=pd.DataFrame(columns={'susceptible':S0,
                                                'infected':I0,
                                                'recovered':R0})

        def SIR_model(SIR,beta,gamma):
            #gamma=0
            S,I,R=SIR
            dS_dt=-beta*S*I/N0          #S*I is the
            dI_dt=beta*S*I/N0-gamma*I
            dR_dt=gamma*I
            return([dS_dt,dI_dt,dR_dt])

        for each_beta in pd_beta:
            new_delta_vec=SIR_model(SIR,each_beta,gamma)
            SIR=SIR+new_delta_vec
            propagation_rates=propagation_rates.append({'susceptible':SIR[0],
                                                            'infected':SIR[1],
                                                            'recovered':SIR[2]}, ignore_index=True)
        if sir == 's-i-r':
            sir=['susceptible','infected','recovered']
            for each in sir:
                traces.append(dict(x=propagation_rates.index,
                                  y=propagation_rates[each],
                                  mode='markers+lines',
                                  opacity=0.9,
                                  name=each),
                              )
        else:
            traces.append(dict(x=propagation_rates.index,
                                  y=propagation_rates[sir],
                                  mode='markers+lines',
                                  opacity=0.9,
                                  name=sir),
                              )



    my_yaxis='Number of People (log-scale)'
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
                          args=[{"yaxis.type": "linear","yaxis.title":my_yaxis[0:-11]+" (linear-scale)"}]),

                                  ]),
                     ],
                shapes= [
            {  # Unbounded line at x = 4
                'type': 'rect',
                # x-reference is assigned to the x-values
                'xref': 'x',
                # y-reference is assigned to the plot paper [0,1]
                'yref': 'paper','layer':'below','opacity':0.05,'x0': 0,'y0': 0,'x1': t_initial,'y1': N0,'fillcolor':'green',
                'line': {
                    'color': 'green','width': 3,'opacity':0.05
                }
            },{ 'type': 'rect',
                'xref': 'x',
                'yref': 'paper','layer':'below','opacity':0.10,'x0': t_initial+1,'y0': 0,'x1': t_initial+t_intro_measures,'y1': N0,'fillcolor':'green',
                'line': {
                    'color': 'green','width': 3,'opacity':0.10
                }
            },{ 'type': 'rect',
                'xref': 'x',
                'yref': 'paper','layer':'below','opacity':0.15,'x0': t_initial+t_intro_measures+1,'y0': 0,'x1': t_initial+t_intro_measures+t_hold,'y1': N0,'fillcolor':'green',
                'line': {
                    'color': 'green','width': 3,'opacity':0.15
                }
              },{ 'type': 'rect',
                'xref': 'x', 'Showlegend':True,
                'yref': 'paper','layer':'below','opacity':0.2,'x0': t_initial+t_intro_measures+t_hold+1,'y0': 0,'x1': t_initial+t_intro_measures+t_hold+t_relax,'y1': N0,'fillcolor':'green',
                'line': {
                    'color': 'green','width': 3,'opacity':0.2
                }
              }
                ],
                label=[{'color':'blue','name':'hi'}]
                )
    }

if __name__ == '__main__':

    app.run_server(debug=True, use_reloader=False)
