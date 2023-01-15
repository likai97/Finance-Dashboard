import pandas as pd
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go
import numpy as np
import pickle

from app import app
from apps.overview import overview
from apps.forecast import fcf_forecast
from apps.technical_analysis import technical_analysis
from apps.ml_model import ml_model
from utils import get_data_alphavantage, format_number, plot_bollinger_band, plot_ichimoku_cloud, train_ml_model


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([html.H2(children='Welcome to my Stock Picker', className='')]
                    , className="mb-2 mt-5")
        ]),
        dbc.Row([
            dbc.Col([html.P(
                children=['Please Search for a Stock via its Ticker and hit search once! Please note that only a limited'
                         ' APIs are possible and that fetching the data can take a while.'
                         'If an error occurs or the page is not loading properly, please report a bug at ',
                html.A(
                    'https://github.com/likai97/Finance-Dashboard',
                    href="https://github.com/likai97/Finance-Dashboard",
                    target="_blank"
                ), '.'])
            ]
                , className="")
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='drpdwn-stock',
                    options=[
                        # {'label':i, 'value':j} for i,j in zip(sp500['label'],sp500['value'])
                        {'label': 'AbbVie', 'value': 'ABBV'},
                        {'label': 'Adobe', 'value': 'ADBE'},
                        {'label': 'AMD', 'value': 'AMD'},
                        {'label': 'Amazon', 'value': 'AMZN'},
                        {'label': 'Apple', 'value': 'AAPL'},
                        {'label': 'BlackRock', 'value': 'BLK'},
                        {'label': 'Coca Cola', 'value': 'KO'},
                        {'label': 'Facebook', 'value': 'FB'},
                        {'label': 'Intel', 'value': 'INTC'},
                        {'label': 'Mastercard', 'value': 'MA'},
                        {'label': 'Mc Donalds', 'value': 'MCD'},
                        {'label': 'Microsoft', 'value': 'MSFT'},
                        {'label': 'Nvidia', 'value': 'NVDA'},
                        {'label': 'PayPal', 'value': 'PYPL'},
                        {'label': 'Tesla', 'value': 'TSLA'},
                        {'label': 'Visa', 'value': 'V'},
                        {'label': 'Walt Disney', 'value': 'DIS'},
                    ],
                    value='AAPL'
                )
            ], width=3),
            dbc.Col([
                dbc.Button(
                    "Search", id="btn-search", className=''
                )
            ], width=1),
            dbc.Col([
                dbc.Spinner([html.Div(),html.Div(id="loading-output")], color="primary")
            ], width=1, className="mt-4")
        ]),
    ], class_name="mb-4"),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(overview, label="Overview", className=''),
                    dbc.Tab(fcf_forecast, label="FCF Forecast", className=''),
                    dbc.Tab(technical_analysis, label="Technical Analysis", className=''),
                    dbc.Tab(ml_model, label="ML Forecast", className='')
                ], class_name="mb-4")
            ])
        ])
    ]),
    html.Div([
        dcc.Store(id="memory", data=[])
    ])
])


@app.callback(
    Output('price_time_series', 'figure'),
    Output('bar_revenue', 'figure'),
    Output('bar_margins', 'figure'),
    Output('bar_roe', 'figure'),
    Output('bar_netdebt', 'figure'),
    Output('stock-key-figures', 'children'),
    Output('memory', 'data'),
    Output("loading-output", "children"),
    Output('bollinger_bands', 'figure'),
    Output('ichimoku_cloud', 'figure'),
    Input("btn-search", "n_clicks"),
    State('drpdwn-stock', 'value'))
def display_page(n_clicks, ticker):
    # if n_clicks is None:
    #     raise PreventUpdate
    df_plot, df_inc, df_overview, df_store = get_data_alphavantage(n_clicks, ticker)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close']))
    fig.update_layout(title=ticker + " Stockprice",
                      template="plotly_white")
    fig.update_yaxes(title_text="Stock Price in $")
    fig.update_xaxes(showgrid=False)

    rev = go.Figure(
        data=[
            go.Bar(name='Total Revenue', x=df_inc['date'].dt.year, y=df_inc['revenue'],
                   marker_color="palevioletred"),
            go.Bar(name='EBIT', x=df_inc['date'].dt.year, y=df_inc['ebit'], marker_color="midnightblue"),
            go.Bar(name='Net Income', x=df_inc['date'].dt.year, y=df_inc['netIncome'],
                   marker_color="lightsteelblue"),
        ]
    )
    rev.update_layout(title="Revenue, EBIT, Net Income in $",
                      template="plotly_white")

    margin = go.Figure(
        data=[
            go.Bar(name='Gross Margin', x=df_inc['date'].dt.year, y=df_inc['Gross Margin'],
                   marker_color="palevioletred"),
            go.Bar(name='EBIT Margin', x=df_inc['date'].dt.year, y=df_inc["EBIT Margin"],
                   marker_color="midnightblue"),
            go.Bar(name='Net Profit Margin', x=df_inc['date'].dt.year, y=df_inc["Net Profit Margin"],
                   marker_color="lightsteelblue"),
        ]
    )
    margin.update_layout(title="Margin Analysis in %",
                         template="plotly_white")

    roe = go.Figure(
        data=[
            go.Scatter(name='ROE', x=df_inc['date'].dt.year, y=df_inc['ROE'],
                       marker_color="palevioletred")
        ]
    )
    roe.update_layout(title="Return on Equity in %",
                      template="plotly_white")

    netdebt = go.Figure(
        data=[
            go.Bar(name='Net Debt', x=df_inc['date'].dt.year, y=df_inc['netDebt'],
                   marker_color="palevioletred")
        ]
    )
    netdebt.update_layout(title="Net Debt in $",
                          template="plotly_white",
                          bargap=0.2)

    bollinger = plot_bollinger_band(df_plot, ticker)
    ichimoku = plot_ichimoku_cloud(df_plot, ticker)
    # fig.show(config={"displayModeBar": False, "showTips": False})  # Remove floating menu and unnecesary dialog box
    print('done')
    return fig, rev, margin, roe, netdebt, \
           dbc.Table.from_dataframe(df_overview, bordered=True,
                                                                    striped=True), [df_store.to_dict(),
                                                                                    df_plot["Close"][
                                                                                        df_plot.index == df_plot.index.max()],
                                                                                    df_plot.index.max()], \
           "", bollinger, ichimoku


@app.callback(
    Output('FCF_Stats', 'children'),
    Input('memory', 'data')
)
def display_stats_for_forecast(data):
    if data is None:
        raise PreventUpdate
    df_table = pd.DataFrame(data[0])
    for x in ['revenue', 'OperatingCashFlow', 'Free Cash Flow', 'netDebt']:
        df_table[x] = df_table[x].apply(format_number)
    df_table.rename(columns={"revenue": 'Total Revenue', 'OperatingCashFlow': 'Operating Cash Flow',
                             'sharesOutstanding': 'Common Stock Shares Outstanding', 'netDebt': 'Net Debt'},
                    inplace=True)
    df_table = df_table.T
    df_table.reset_index(level=0, inplace=True)
    df_table.rename(columns={"index": 'Fiscal Year'}, inplace=True)
    return dbc.Table.from_dataframe(df_table, bordered=True)

@app.callback(
    Output("collapse", "is_open"),
    [Input("fcf-switch", "on")]
)
def toggle_collapse(on):
    print(on)
    if on:
        return True
    else:
        return False


@app.callback(
    Output('output', 'children'),
    Output('total_pv_fcf', 'children'),
    Output('total_value_firm', 'children'),
    Output('estimated_share_value', 'children'),
    Output('current_stock_price', 'children'),
    Output('upside_fair_value', 'children'),
    Input("btn-calculate", "n_clicks"),
    Input('memory', 'data'),
    State('npt-rrr', 'value'),
    State('npt-gr1', 'value'),
    State('npt-gr2', 'value'),
    State('npt-gr3', 'value'),
    State('npt-gr4', 'value'),
    State('npt-gr5', 'value'),
    State('npt-grp', 'value'),
)
def fcf_forecast(btn, data, rrr, gr1, gr2, gr3, gr4, gr5, grp):
    if btn is None:
        raise PreventUpdate
    df_table = pd.DataFrame(data[0])
    year = df_table.index.max()
    df_table = df_table.loc[year]
    years = [int(year) + x for x in range(6)]
    years.append("Terminal Value")
    # calculate fcf
    fcf = []
    fcf.append(df_table['Free Cash Flow'].item())
    for i in [gr1, gr2, gr3, gr4, gr5]:
        fcf.append(fcf[-1] * (1 + i))
    fcf.append(fcf[-1] * (1 + grp) / (rrr - grp))
    pv = []
    pv.append(fcf[0])
    for i in range(1, 6):
        pv.append(fcf[i] / (1 + rrr) ** i)
    pv.append(fcf[6] / (1 + rrr) ** 5)
    df = pd.DataFrame(
        {'Year': years,
         'Growth Assumption': ["/", gr1, gr2, gr3, gr4, gr5, grp],
         'Free Cash Flow(FCF)': [round(num, 2) for num in fcf],
         'PV of FCF': [round(num, 2) for num in pv]
         }
    )
    # forecast = rrr + gr1 + gr2 + gr3 + gr4 + gr5 + grp
    forecast = df.T.reset_index()
    forecast.columns = forecast.iloc[0]
    forecast = forecast[1:]
    total_pv_fcf = sum(pv)
    total_value_firm = total_pv_fcf - df_table['netDebt'].item()
    estimated_share_value = total_value_firm / df_table['sharesOutstanding'].item()
    return dbc.Table.from_dataframe(forecast, bordered=True), "Total Present Value of FCFs: {}".format(
        format_number(total_pv_fcf)), \
           "Total Value of Firm (+ Cash - Debt): {}".format(format_number(total_value_firm)), \
           "Estimated share value (original currency): {}".format(round(estimated_share_value, 2)), \
           "Current stock price ({}): {}".format(data[2], round(data[1][0],2)), \
           "Upside to fair value in %: {}".format(round(((estimated_share_value / data[1][0]) - 1) * 100, 2))


@app.callback(
    Output('ml-forecast', 'figure'),
    Output('ml-validation', 'figure'),
    Output("loading-output-ml", "children"),
    Input("btn-train", "n_clicks"),
    State('drpdwn-stock', 'value'))
def fcf_forecast(n_clicks, ticker):
    if n_clicks is None:
        raise PreventUpdate
    forecast_plot, validation_plot = train_ml_model(ticker)
    return forecast_plot, validation_plot, ""

