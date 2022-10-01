from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, Output, State

fcf_forecast = html.Div([
    dbc.Container([
        dbc.Col([
            dbc.Row(
                html.H3(children='Free Cash Flow Forecasts')
            ),
            dbc.Row(
                html.P(
                    children='Please enter assumptions. Each entry should be between 0 and 1. Once finished hit calculate and wait shortly for the model to process the inputs.')
            )
        ], className='mb-2'),
        dbc.Row([
            dbc.Col(width=11),
            dbc.Col([
                daq.BooleanSwitch(id='fcf-switch',
                                  on=True,
                                  className='mb-1 right')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Collapse(
                    html.Div(id='FCF_Stats'),
                    id="collapse",
                    is_open=True,
                )

            ])
        ], className='mb-4'),
        dbc.Card([
            dbc.CardHeader("FCF Assumptions", style={"font-weight": "bold"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Required Rate of Return"),
                        dbc.Input(id="npt-rrr", type="number", min=0, max=1, value=0.1, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Forecast t1"),
                        dbc.Input(id="npt-gr1", type="number", min=0, max=1, value=0.15, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Forecast t2"),
                        dbc.Input(id="npt-gr2", type="number", min=0, max=1, value=0.1, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Forecast t3"),
                        dbc.Input(id="npt-gr3", type="number", min=0, max=1, value=0.08, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Forecast t4"),
                        dbc.Input(id="npt-gr4", type="number", min=0, max=1, value=0.07, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Forecast t5"),
                        dbc.Input(id="npt-gr5", type="number", min=0, max=1, value=0.06, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Label("Growth Rate Perpetual"),
                        dbc.Input(id="npt-grp", type="number", min=0, max=.2, value=0.05, step=0.01)
                    ]),
                    dbc.Col([
                        dbc.Button("Calculate", id="btn-calculate", color="primary")
                    ])
                ]),
            ])
        ], class_name="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H3("DCF")
            ])
        ], class_name="mb-2"),
        dbc.Row([
            dbc.Col(
                html.Div(id="output")
            )
        ], class_name="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("FCF Forecast Results", style={"font-weight": "bold"}),
                    dbc.CardBody([
                        html.P(id='total_pv_fcf', className="mb-2", style={"font-weight": "bold"}),
                        html.P(id='total_value_firm', className="mb-2", style={"font-weight": "bold"}),
                        html.P(id='estimated_share_value', className="mb-2", style={"font-weight": "bold"}),
                        html.P(id='current_stock_price', className="mb-2", style={"font-weight": "bold"}),
                        html.P(id='upside_fair_value', className="mb-2", style={"font-weight": "bold"}),
                    ])
                ])
            ], width=6)
        ], class_name="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H3("Sensitivity Analysis")
            ])
        ], class_name="mb-2"),
        # dbc.Row([
        #     dbc.Col(
        #         html.Div(id="sensitivity-analysis")
        #     )
        # ], class_name="mb-3"),
    ])
])
