# Finance-Dashboard
This app visualizes stock data and allows for a Free Cash Flow forecast!

Visit it on: https://fcf-dashboard.herokuapp.com/

Note: As the App is in Sleep Mode per Default to reduce costs, the initial loading time about a minute. 


## Requirements
Python: 3.9 or 3.10

Requirements: 

* dash==2.6.1
* dash_bootstrap_components==1.2.1
* dash_daq==0.5.0
* numpy==1.23.3
* pandas==1.4.0
* plotly==5.9.0
* requests==2.27.1
* yfinance==0.1.74
* gpytorch==1.6.0
* torch==1.11.0
* gunicorn==20.1.0

It is recommended to create a separate virtual environment and install [requirements.txt](https://github.com/likai97/Finance-Dashboard/blob/main/requirements.txt).

### Usage

To run the app locally, you'll need to get the following API keys and add them to the [utils.py](https://github.com/likai97/Finance-Dashboard/blob/main/utils.py):

* [Alphavantage](www.alphavantage.co)
* [Financial Modelling Prep](https://site.financialmodelingprep.com/)
* [Polygon](https://polygon.io/)
