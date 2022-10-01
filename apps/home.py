from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app import app

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to my Stock Analysis dashboard", className="text-center"),
                    className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='This app visualizes stock data and allows for a Free Cash Flow forecast! ',
                            className='')
                    , className="mb-2")
        ]),

        dbc.Row([
            dbc.Col(html.P(
                children='It consists of four pages:')
                , className="mb-1"),
            html.Ol([
                html.Li('Stats, which gives a summary of Financial Information of a given Stock'),
                html.Li('FCF-Forecast, which tries to estimate a stocks fair value given its Free Cash Flow Forecast'),
                html.Li('Technical Analysis, which shows Bollinger Bands and Ichimoku Cloud plots'),
                html.Li('ML Model, which trains a Gaussian Process and tries to predict the stock price movement.')
            ])
        ],className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Go to the Stock Analysis Dashboard',
                                               className="text-center"),
                                       dbc.Button("Stock Analysis",
                                                  href="/stockanalysis",
                                                  class_name= "mt-3 d-grid col-6 mx-auto")
                                       ],
                             body=True, color="dark", outline=True)
                    , width={"size": 4, "offset": 2}, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='Access the code used to build this dashboard',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/likai97",
                                                  #color="Dark",
                                                  className="mt-3 d-grid col-6 mx-auto"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=4, className="mb-4"),
        ], className="mb-5"),
    ])
])
