from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

ml_model = dbc.Container([
    html.H2("ML Model Forecasts", className='mb-1'),
    dbc.Row([
       dbc.Col([
           html.P(
                'Guassian Process Forecast using the GPytorch Package. The GP will be trained on daily Closing data '
                'for the last 5 years using a the combination of a Matern Kernel with a linear Kernel. Although the '
                'prediction will not be perfect, it is usually within the GPs confidence bounds.'
           ),
           html.P(
               'Please click the button only once and wait as model training takes a while.'
           ),

       ])
    ], className='mb-1'),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='drpdwn-ml',
                options=[
                    # {'label':i, 'value':j} for i,j in zip(sp500['label'],sp500['value'])
                    {'label': 'Gaussian Process', 'value': 'GP'},
                ],
                value='GP'
            )
        ], width=3),
        dbc.Col([
            dbc.Button(
                "Train Model", id="btn-train", className=''
            )
        ], width=2),
        dbc.Col([
            dbc.Spinner([html.Div(), html.Div(id="loading-output-ml")], color="primary")
        ], width=1, className="mt-4")
    ]),
    dbc.Row([
        dcc.Graph(id="ml-forecast",
                  figure={},
                  style={'height': '70vh'})

    ], className="mt-4"),
    html.H2("Model Validation", className='mt-1'),
    dbc.Row([
        dcc.Graph(id="ml-validation",
                  figure={},
                  style={'height': '70vh'})

    ], className="mt-4"),
    ])
