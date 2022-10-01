from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

overview = dbc.Container([
    html.H2("Financial Statistics", className=''),
    dbc.Row([
        dbc.Col([
            html.Div(id='stock-key-figures')
        ], width=6
        ),
        dbc.Col([
            dcc.Graph(id="price_time_series",
                      figure={})], width=6
        )]
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="bar_revenue",
                      figure={})
        ], width=6
        ),
        dbc.Col([
            dcc.Graph(id="bar_margins",
                      figure={})], width=6
        )]
    ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="bar_roe",
                      figure={})
        ], width=6
        ),
        dbc.Col([
            dcc.Graph(id="bar_netdebt",
                      figure={})], width=6
        )]
    )
])