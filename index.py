from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from app import server
from app import app
from apps import home, dashboard

navbar = dbc.Navbar(
    dbc.Container([
        html.Div(
            dbc.Row(
                [
                    dbc.Col(html.Img(src="/assets/stonks.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("Stock Analysis", className="ml-2")),
                ],
                align="center",
                className="g-2",
            ),
        ),
        dbc.Col(width=6),
        dbc.Row([
            dbc.Col([
                html.A(
                    dbc.Col(html.Img(src="/assets/github.png", height="30px")),
                    href="https://github.com/likai97/Finance-Dashboard",
                    target="_blank"
                )
            ]),
            dbc.Col([
                html.A(
                    dbc.Col(html.Img(src="/assets/home.png", height="30px")),
                    href="/stockanalysis",
                )
            ]),

        ])
    ])
)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/stockanalysis':
        return dashboard.layout
    else:
        return dashboard.layout


if __name__ == '__main__':
    app.run_server(debug=False)
