from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

technical_analysis = dbc.Container([
    html.H2("Bollinger Bands", className='mb-1'),
    dbc.Row([
        dcc.Graph(id="bollinger_bands",
                  figure={},
                  style={'height': '70vh'})
    ], className='mb-3'),
    html.H2("Ichimoku Cloud", className='mb-1'),
    dbc.Row([
        dcc.Graph(id="ichimoku_cloud",
                  figure={},
                  style={'height': '70vh'})

    ]),
])
