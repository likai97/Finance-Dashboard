import requests
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

import torch
import gpytorch
from datetime import timedelta

av_keys = ["", ""]  # enter your alphavantage key here
fmp_key = ""  # financialmodelingprep api key
pg_key = ""  # polygon api key


def get_data_alphavantage(n_clicks, stock):
    """Returns the alpha vantage data

    Args:
      n_clicks: Depending on the number of clicks cycle through the api keys
      stock: Ticker of which stock to analyze

    Returns:
      4 dataframes: stock price data, balance/income/cash flow data, company statistics and summary statistics on balance/income/cash flow data
    """
    print(n_clicks)
    print(stock)
    # default data for display
    if n_clicks is None:
        df_time = pd.read_csv('./data/default_price_time_series.csv', index_col=0)
        df_all = pd.read_csv('./data/default_financial_data.csv', index_col=0)
        df_all['date'] = pd.to_datetime(df_all['date'])
        key_stats = pd.read_csv('./data/default_company_statistics.csv')
        store_data = pd.read_csv('./data/default_fcf_data.csv', index_col=0)
        print("finished fetching data")
        return df_time, df_all, key_stats, store_data

    if n_clicks % 2 == 0:
        av_key = av_keys[0]
    else:
        av_key = av_keys[1]
    try:
    # TimeSeries data
        df_time = get_stock_price_data(stock)
        # add daily and cumulative return
        df_time = add_daily_return(df_time)
        df_time = add_cumulative_return(df_time)
        #add bollinger bands
        df_time = add_bollinger_band(df_time)
        #add ichimoku cloud data
        df_time = add_ichimoku_cloud(df_time)
        # Income Statement
        df_inc = get_income_data(fmp_key, stock)
        # Balance
        df_balance = get_balance_sheet_data(fmp_key, stock)
        # Cash Flow
        df_cash = get_cashflow_data(av_key, stock)
        # Overview data
        key_stats, shares = get_company_statistics(av_key, stock)
        # Join the dataset
        df_all = process_financials(df_inc, df_balance, df_cash, shares)
        store_data = df_all[["Fiscal Year", 'revenue', 'Revenue % Change', 'Average Revenue Growth',
                             'OperatingCashFlow', 'OCF % Change', 'Average OCF Growth',
                             'Free Cash Flow', 'FCF % Change', 'Average FCF Growth', 'netDebt',
                             'sharesOutstanding']].set_index('Fiscal Year')
    # error occured because api provider probably changed their dataformat
    except:
        df_time = pd.read_csv('./data/default_price_time_series.csv', index_col=0)
        df_all = pd.read_csv('./data/default_financial_data.csv', index_col=0)
        df_all['date'] = pd.to_datetime(df_all['date'])
        key_stats = pd.read_csv('./data/default_company_statistics.csv')
        store_data = pd.read_csv('./data/default_fcf_data.csv', index_col=0)

    print("finished fetching data")
    return df_time, df_all, key_stats, store_data


def get_stock_price_data(ticker, period='5y'):
    """Returns the stock price data from yahoo finance

    Args:
      ticker: Ticker of which stock to analyze
      period: period of price data

    Returns:
      A dataframe with stock price movement data
    """
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval='1d')


def get_balance_sheet_data(key, stock):
    """Returns the balance sheet data

    Args:
      key: api key
      stock: Ticker of which stock to analyze

    Returns:
      A dataframe with balance sheet data
    """
    url = "https://financialmodelingprep.com/api/v3/balance-sheet-statement/" + stock + "?apikey=" + key
    r = requests.get(url)
    temp = r.json()
    df = pd.DataFrame.from_dict(temp)
    df = df[['date', 'totalAssets', 'totalCurrentAssets', 'cashAndCashEquivalents', 'shortTermDebt',
             'totalCurrentLiabilities', 'longTermDebt', 'totalNonCurrentLiabilities', 'totalLiabilities', 'commonStock',
             'totalStockholdersEquity', 'totalDebt', 'netDebt']]
    df['date'] = pd.to_datetime(df['date'])
    df['Fiscal Year'] = df['date'].dt.year
    return df


def get_income_data(key, stock):
    """Returns the income statement data

    Args:
      key: api key
      stock: Ticker of which stock to analyze

    Returns:
      A dataframe with income statement data
    """
    pass
    url = "https://financialmodelingprep.com/api/v3/income-statement/" + stock + "?apikey=" + key
    r = requests.get(url)
    temp = r.json()
    df = pd.DataFrame.from_dict(temp)
    df = df[['date', 'reportedCurrency', 'grossProfit', 'revenue', 'costOfRevenue', 'depreciationAndAmortization',
             'ebitda', 'operatingIncome', 'incomeBeforeTax', 'netIncome', 'eps']]
    df['ebit'] = df['ebitda'] - df['depreciationAndAmortization']
    df['date'] = pd.to_datetime(df['date'])
    df['Fiscal Year'] = df['date'].dt.year
    return df


def get_cashflow_data(key, stock):
    """Returns the cashflow data

    Args:
      key: api key
      stock: Ticker of which stock to analyze

    Returns:
      A dataframe with cashflow data
    """
    try:
        url = 'https://www.alphavantage.co/query?function=CASH_FLOW&symbol=' + stock + '&apikey=' + key
        r = requests.get(url)
        df = pd.DataFrame(r.json()['annualReports'])
        df.rename(columns={'fiscalDateEnding': 'date',
                           'operatingCashflow': 'OperatingCashFlow',
                           'capitalExpenditures': 'CapitalExpenditure'}, inplace=True)
        df.date = pd.to_datetime(df.date, format="%Y/%m/%d %H:%M")
        df['Fiscal Year'] = df.date.dt.year
        # Change datatype
        df['OperatingCashFlow'] = pd.to_numeric(df['OperatingCashFlow'])
        df['CapitalExpenditure'] = pd.to_numeric(df['CapitalExpenditure'])
        return df[['date', 'Fiscal Year', 'OperatingCashFlow', 'CapitalExpenditure']]
    except KeyError:
        stock = yf.Ticker(stock)
        df = stock.get_cashflow().T
        df.reset_index(level=0, inplace=True)
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['Fiscal Year'] = df.date.dt.year
        return df[['date', 'Fiscal Year', 'OperatingCashFlow', 'CapitalExpenditure']]


def get_company_statistics(key, stock):
    """Returns the company statistcs

    Args:
      key: api key
      stock: Ticker of which stock to analyze

    Returns:
      A dataframe with company statistcs
    """
    url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + stock + '&apikey=' + key
    r = requests.get(url)
    temp = r.json()
    temp = pd.DataFrame.from_dict(temp, orient='index').T
    shares = int(temp['SharesOutstanding'])
    for x in ['MarketCapitalization', 'SharesOutstanding']:
        temp[x] = format_number(temp[x].item())
    temp['DividendYield'] = temp['DividendYield'].astype("float64")
    temp['DividendYield'] = round(temp['DividendYield'] * 100, 1)
    temp.rename(columns={"MarketCapitalization": 'Market Capitalization', 'PERatio': 'PE Ratio',
                         'PEGRatio': 'PEG Ratio', 'PriceToSalesRatioTTM': 'Price to Sales Ratio TTM',
                         'DividendYield': 'Dividend Yield in %', 'SharesOutstanding': 'Shares Outstanding'},
                inplace=True)

    df = temp.T.loc[
                ["Name", "Country", "Sector", "Market Capitalization", "PE Ratio", "PEG Ratio",
                 "Price to Sales Ratio TTM",
                 "EPS",
                 'Dividend Yield in %', "Shares Outstanding"], :].reset_index()
    df.columns = df.iloc[0]
    return df[1:], shares


def process_financials(inc, balance, cash, shares):
    """Returns the cashflow data

    Args:
      key: api key
      stock: Ticker of which stock to analyze

    Returns:
      A dataframe with cashflow data
    """
    # Join the dataset
    df_all = pd.merge(inc, balance, how="left", on=["Fiscal Year"]).merge(cash, how="left", on=["Fiscal Year"])

    # calculate Stats
    df_all["Gross Margin"] = df_all["grossProfit"]/df_all["revenue"] * 100
    df_all["Net Profit Margin"] = df_all["netIncome"] / df_all["revenue"] * 100
    df_all["EBIT Margin"] = df_all["ebit"] / df_all["revenue"] * 100
    df_all['Free Cash Flow'] = df_all['OperatingCashFlow'] + df_all['CapitalExpenditure']
    df_all["ROE"] = df_all["netIncome"] / df_all['totalStockholdersEquity'] * 100

    df_all['Revenue % Change'] = round(df_all['revenue'].pct_change(-1) * 100, 2)
    df_all['OCF % Change'] = round(df_all['OperatingCashFlow'].pct_change(-1) * 100, 2)
    df_all['FCF % Change'] = round(df_all['Free Cash Flow'].pct_change(-1) * 100, 2)

    df_all['Average Revenue Growth'] = round(df_all['Revenue % Change'].mean(), 2)
    df_all['Average OCF Growth'] = round(df_all['OCF % Change'].mean(), 2)
    df_all['Average FCF Growth'] = round(df_all['FCF % Change'].mean(), 2)

    # As alpha vantage no longer provides historical information, just take the most recent info
    df_all['sharesOutstanding'] = None
    df_all.loc[df_all['Fiscal Year']==max(df_all['Fiscal Year']), 'sharesOutstanding'] = shares
    return df_all


def convert_dict_to_pd(dict_data):
    dff = pd.DataFrame(dict_data)

    dff['Revenue % Change'] = dff['revenue'].pct_change(-1) * 100
    dff['OCF % Change'] = dff['OperatingCashFlow'].pct_change(-1) * 100
    dff['FCF % Change'] = dff['Free Cash Flow'].pct_change(-1) * 100

    dff['Average Revenue Growth'] = round(dff['Revenue % Change'].mean(), 2)
    dff['Average OCF Growth'] = round(dff['OCF % Change'].mean(), 2)
    dff['Average FCF Growth'] = round(dff['FCF % Change'].mean(), 2)

    return dff.loc[
        0, ["Fiscal Year", 'revenue', 'Average Revenue Growth', 'OperatingCashFlow', 'Average OCF Growth',
            'Free Cash Flow', 'Average FCF Growth', 'Net Debt', 'commonStock']]


def safe_num(num):
    if isinstance(num, str):
        num = float(num)
    return float('{:.3g}'.format(abs(num)))


def format_number(num):
    num = safe_num(num)
    sign = ''

    metric = {' Trillion': 1000000000000, ' Billion': 1000000000, ' Million': 1000000, ' Thousand': 1000, '': 1}

    for index in metric:
        num_check = num / metric[index]
        if (num_check >= 1):
            num = num_check
            sign = index
            break
    return f"{str(num).rstrip('0').rstrip('.')}{sign}"


def add_daily_return(df):
    df['daily_return'] = (df['Close'] / df['Close'].shift(1)) - 1
    return df


def add_cumulative_return(df):
    df['cum_return'] = (1 + df['daily_return']).cumprod() - 1
    return df


def add_bollinger_band(df):
    df['middle_band'] = df['Close'].rolling(window = 20).mean()
    df['upper_band'] = df['middle_band'] + 1.96 * df['Close'].rolling(window = 20).std()
    df['lower_band'] = df['middle_band'] - 1.96 * df['Close'].rolling(window = 20).std()
    return df


def add_ichimoku_cloud(df):
    # Conversion Line
    high_val = df['High'].rolling(window=9).max()
    low_val = df['High'].rolling(window=9).min()
    df['conversion_line'] = (high_val + low_val) * 0.5

    # Baseline
    high_val2 = df['High'].rolling(window=26).max()
    low_val2 = df['High'].rolling(window=26).min()
    df['base_line'] = (high_val2 + low_val2) * 0.5

    # Spans
    df['leading_span_A'] = ((df['conversion_line'] + df['base_line']) * 0.5).shift(26)

    high_val3 = df['High'].rolling(window=52).max()
    low_val3 = df['High'].rolling(window=52).min()
    df['leading_span_B'] = ((high_val3 + low_val3) * 0.5).shift(26)

    df['lagging_span'] = df['Close'].shift(-26)
    return df


def plot_bollinger_band(df, ticker):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df['Low'],
                            close=df['Close'], name="Candlestick")

    upper_line = go.Scatter(x=df.index, y=df['upper_band'],
                            line=dict(color='rgba(250, 0, 0, 0.75)',
                                      width=1), name="Upper Band")

    mid_line = go.Scatter(x=df.index, y=df['middle_band'],
                          line=dict(color='rgba(0, 0, 250, 0.75)',
                                    width=0.7), name="Middle Band")

    lower_line = go.Scatter(x=df.index, y=df['lower_band'],
                            line=dict(color='rgba(0, 250, 0, 0.75)',
                                      width=1), name="Lower Band")

    fig.add_trace(candle)
    fig.add_trace(upper_line)
    fig.add_trace(mid_line)
    fig.add_trace(lower_line)

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    fig.update_layout(title=ticker + " Bollinger Bands",
                      # height=1200, width=1800,
                      showlegend=True, template="plotly_white")
    return fig


# Used to generate the red and green fill for the Ichimoku cloud
def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


def plot_ichimoku_cloud(df, ticker):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

    conversion = go.Scatter(x=df.index, y=df['conversion_line'],
                            line=dict(color='blue', width=1),
                            name='Conversion')

    baseline = go.Scatter(x=df.index, y=df['base_line'],
                          line=dict(color='red', width=1),
                          name='Baseline')

    spanA = go.Scatter(x=df.index, y=df['leading_span_A'],
                       line=dict(color='rgba(250,0,0,.8)', width=1, dash='dot'),
                       name='Span A')

    spanB = go.Scatter(x=df.index, y=df['leading_span_B'],
                       line=dict(color='rgba(0,250,0,0.8)', width=1, dash='dot'),
                       name='Span B')

    lagging = go.Scatter(x=df.index, y=df['lagging_span'],
                         line=dict(color='grey', width=1),
                         name='Lagging Span')

    df1 = df.copy()
    df['label'] = np.where(df['leading_span_A'] > df['leading_span_B'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()

    df = df.groupby('group')

    dfs = []
    for name, data in df:
        dfs.append(data)

    for df in dfs:
        # noinspection PyTypeChecker
        fig.add_traces(go.Scatter(x=df.index, y=df.leading_span_A,
                                  line=dict(color='rgba(0,0,0,0)'), showlegend=False))

        # noinspection PyTypeChecker
        fig.add_traces(go.Scatter(x=df.index, y=df.leading_span_B,
                                  line=dict(color='rgba(0,0,0,0)'),
                                  fill='tonexty',
                                  fillcolor=get_fill_color(df['label'].iloc[0]), showlegend=False))

    fig.add_trace(candle)
    fig.add_trace(conversion)
    fig.add_trace(baseline)
    fig.add_trace(spanA)
    fig.add_trace(spanB)
    fig.add_trace(lagging)

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    fig.update_layout(title=ticker + " Ichimoku Cloud",
                      # height=1200, width=1800,
                      showlegend=True, template="plotly_white")

    return fig


def train_ml_model(ticker, n_shifts=30, train_size=0.9, epochs=30):
    """
    Trains a GP Model and returns a forecast depending on the closing price

    Args:
        ticker: Stock Ticker
        n_shifts: Number of days to base the forecast on
        train_size: Train size
        epochs: Train epochs

    Returns:
        Train Features and train labels
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period='5y', interval='1d')
    data_close_price = df['Close']
    # normalize
    scaler = StandardScaler()
    scaled_data_close_price = scaler.fit_transform(data_close_price)
    scaled_data_close_price = scaled_data_close_price.to_frame()

    X, y = prepare_ml_forecast_data(scaled_data_close_price, n_shifts)

    # Divide into train and test dataset
    split_index = int(X.shape[0] * train_size)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    # Turn to Tensors
    X_train = torch.Tensor(X_train.values)
    X_test = torch.Tensor(X_test.values)
    y_train = torch.Tensor(y_train.values)
    y_test = torch.Tensor(y_test.values)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.15)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        if i % 10 ==0:
            print('Iter %d/%d - Loss: %.3f ' % (
                i + 1, epochs, loss.item()
            ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test))
        train_pred = likelihood(model(X_train))

    # Validation plot
    # prepare data for plotting
    num_data_points = X.shape[0]
    plot_data_y_train_pred = np.zeros(num_data_points)
    plot_data_y_test_pred = np.zeros(num_data_points)

    plot_data_y_train_pred[:split_index] = scaler.inverse_transform(train_pred.mean)
    plot_data_y_test_pred[split_index:] = scaler.inverse_transform(observed_pred.mean)

    plot_data_y_train_pred = np.where(plot_data_y_train_pred == 0, None, plot_data_y_train_pred)
    plot_data_y_test_pred = np.where(plot_data_y_test_pred == 0, None, plot_data_y_test_pred)

    validation_plot = go.Figure()

    validation_plot.add_trace(go.Scatter(x=X.index, y=plot_data_y_train_pred,
                                         mode='lines',
                                         line=dict(color='rgb(44, 160, 44)'),
                                         name='Predicted prices (train)'))

    validation_plot.add_trace(go.Scatter(x=X.index, y=scaler.inverse_transform(y),
                                         mode='lines',
                                         line=dict(color='rgb(31, 119, 180)'),
                                         name='Actual prices'))

    validation_plot.add_trace(go.Scatter(x=X.index, y=plot_data_y_test_pred,
                                         mode='lines',
                                         line=dict(color='rgb(214, 39, 40)'),
                                         name='Predicted prices (validation)'))

    validation_plot.update_xaxes(title="Date", rangeslider_visible=True)
    validation_plot.update_yaxes(title="Price")

    validation_plot.update_layout(title=ticker + " Forecast",
                                  # height=1200, width=1800,
                                  showlegend=True, template="plotly_white")


    # Prediction for unseen test point
    test_point = torch.Tensor(X.tail(1).values)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(test_point))
    predictions = prediction.mean.item()
    lower, upper = prediction.confidence_region()

    #prepare plot data
    dates = X.index[split_index:-1].append(pd.date_range(start=X.index[-1], end=X.index[-1] + timedelta(days=1)))
    n = len(dates) - 1

    plot_data_true_values = np.zeros(len(dates))
    plot_data_past_predictions = np.zeros(len(dates))
    plot_data_upper = np.zeros(len(dates))
    plot_data_lower = np.zeros(len(dates))
    plot_data_predictions = np.zeros(len(dates))

    plot_data_true_values[:n] = scaler.inverse_transform(y[-n:])
    plot_data_past_predictions[:n] = scaler.inverse_transform(observed_pred.mean)[-n:]
    past_lower, past_upper = observed_pred.confidence_region()
    plot_data_upper[:n] = scaler.inverse_transform(past_upper.detach().numpy())[-n:]
    plot_data_upper[n] = scaler.inverse_transform(np.array(upper.item()))
    plot_data_lower[:n] = scaler.inverse_transform(past_lower.detach().numpy())[-n:]
    plot_data_lower[n] = scaler.inverse_transform(np.array(lower.item()))
    plot_data_predictions[n:] = scaler.inverse_transform(np.array(predictions))

    plot_data_true_values = np.where(plot_data_true_values == 0, None, plot_data_true_values)
    plot_data_past_predictions = np.where(plot_data_past_predictions == 0, None, plot_data_past_predictions)
    plot_data_upper = np.where(plot_data_upper == 0, None, plot_data_upper)
    plot_data_lower = np.where(plot_data_lower == 0, None, plot_data_lower)
    plot_data_predictions = np.where(plot_data_predictions == 0, None, plot_data_predictions)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=plot_data_past_predictions,
                             mode='lines+markers',
                             line=dict(color='rgba(44, 160, 44,1)'),
                             name='Past Predictions'))

    fig.add_trace(go.Scatter(x=dates, y=plot_data_upper,
                             mode='lines',
                             line=dict(width=0),
                             showlegend=False))

    fig.add_trace(go.Scatter(x=dates, y=plot_data_lower,
                             mode='lines',
                             line=dict(width=0),
                             fillcolor='rgba(44, 160, 44, .1)',
                             fill='tonexty',
                             showlegend=False))

    fig.add_trace(go.Scatter(x=dates, y=plot_data_true_values,
                             mode='lines+markers', line=dict(color='rgb(31, 119, 180)'),
                             name='Actual Price'))

    fig.add_trace(go.Scatter(x=dates, y=plot_data_predictions,
                             mode='markers',
                             marker=dict(color='rgb(214, 39, 40)'),
                             name='Prediction for the next day'))

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    fig.update_layout(title=ticker + " Forecast",
                      # height=1200, width=1800,
                      showlegend=True, template="plotly_white")

    return fig, validation_plot


class StandardScaler():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.sd = np.std(x, axis=0)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu


def prepare_ml_forecast_data(df, n_shifts):
    """
    Creates a dataframe with the past n_shifts closing prices for eaach day

    Args:
        df:
        n_shifts:

    Returns:
        Train Features and train labels
    """
    for shift_amount in range(1, n_shifts + 1):
        df[f"Close.lag{shift_amount}"] = df['Close'].shift(shift_amount)

    train = df.iloc[n_shifts:, 1:]
    test = df.iloc[n_shifts:, 0]

    return train, test

class ExactGPModel(gpytorch.models.ExactGP):
    """
    Simple GPytorch GP Class with Matern Kernel

    Args:
        train_x: Train Features
        train_y: Train Labels
        likelihood: Likelihood (Gaussian in this case)

    Returns:
        Train Features and train labels
    """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=train_x.shape[1]) +
            gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)