import pandas as pd
import numpy as np
import scipy.stats as si
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import time
import bs4 as bs
import datetime
from requests_html import HTMLSession

session = HTMLSession()

def sp500_tickers() -> list:
    """
    Get's a list of the S&P 500 tickers from wikipedia.

    Returns: 
        List of tickers
    """
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        if "." in ticker:
            ticker = ticker.replace('.', '-')
        tickers.append(ticker)

    return tickers

def google_news(ticker: str):
    google_ticker_list = []
    all_google_news_headlines = []
    all_google_news_dates = []

    url = f"https://news.google.com/search?q={ticker.upper()}%20stock&hl=en-US&gl=US&ceid=US%3Aen"

    while True:
        r = session.get(url)
        r.html.render(sleep = 1, scrolldown = 0, timeout = 20)
        articles = r.html.find('article')

        for item in articles:
            try:
                news_item = item.find('h3', first = True)
                title_google = news_item.text
                news_date_element = item.find('time')[0]
                news_date_str = str(news_date_element.attrs['datetime']).replace('T', ' ').replace('Z', '')
                news_date_google = datetime.datetime.strptime(news_date_str, '%Y-%m-%d %H:%M:%S').date()

                google_ticker_list.append(ticker)
                all_google_news_headlines.append(title_google)
                all_google_news_dates.append(news_date_google)
                    
            except:
                pass
        
        else:
            break

    google_articles = pd.DataFrame({'ticker': google_ticker_list, 'date': all_google_news_dates, 'headline': all_google_news_headlines})
    return google_articles

def yahoo_news(ticker: str):
    yahoo_ticker_list = []
    all_yahoo_news_headlines = []
    all_yahoo_news_dates = []

    for news in yf.Ticker(ticker).news:
        title_yahoo = news['title'].rstrip('\\').replace('$', '\$')
        news_date_yahoo = str(datetime.datetime.fromtimestamp(news['providerPublishTime']).date())

        yahoo_ticker_list.append(ticker)
        all_yahoo_news_headlines.append(title_yahoo)
        all_yahoo_news_dates.append(news_date_yahoo)

        time.sleep(0.1)

    yahoo_articles = pd.DataFrame({'ticker': yahoo_ticker_list, 'date': all_yahoo_news_dates, 'headline': all_yahoo_news_headlines})
    return yahoo_articles

def options_data(ticker: str):
    expiration_date = '2023-06-16'

    try:
        ticker_data = yf.download(f'{ticker.upper()}', period = '1d', progress = False)
        underlying_asset_price = ticker_data['Adj Close'].iloc[-1]

        option_chain = yf.Ticker(ticker.upper()).option_chain(expiration_date)
        options_data = pd.DataFrame(option_chain.calls)
        options_data['ticker'] = ticker.upper()
        options_data['underlying_price'] = underlying_asset_price
        options_data = options_data[['ticker', 'underlying_price', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]
    
        time.sleep(0.1)

    except ValueError:
        pass
    
    return options_data

def black_scholes_model(S: float, K: int, T: float, r: float, sigma: float) -> float:
    """
    S: Current price of the underlying asset
    K: Strike price of the option
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset

    Returns:
        Call Price
    """
    if sigma == 0:
        return 0
    
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = si.norm.cdf(d1)
    N_d2 = si.norm.cdf(d2)
    call = S * N_d1 - K * np.exp(-r * T) * N_d2
    
    return call

def time_exp(input_date: str, input_expiration_date: str) -> float:
    """
    input_date: Input date provided
    input_expiration_date: Expiration date of the option

    Returns: 
        Time to Expiration
    """
    trade_date = datetime.datetime.strptime(input_date, '%Y-%m-%d %H:%M:%S%z').date()
    expiration_date = datetime.datetime.strptime(input_expiration_date, '%Y-%m-%d').date()
    time_to_expiration = (expiration_date - trade_date).days / 365.25
    
    return time_to_expiration

def assign_weight(score: int) -> float:
    """
    Assigns a weight to a value.

    Returns: 
        The weight
    """
    if score < 0:
        return 1 + abs(score)
    elif score > 0:
        return 1 + score
    else:
        return 1

def backtesting_bsp_model(ticker: str, expiration_date: str, risk_free_rate: float) -> list:
    try:
        ticker_data = yf.download(f'{ticker.upper()}', period = '1d', progress = False)
        underlying_asset_price = ticker_data['Adj Close'].iloc[-1]

        option_chain = yf.Ticker(ticker.upper()).option_chain(expiration_date)
        ticker_options_data = pd.DataFrame(option_chain.calls)
        ticker_options_data['ticker'] = ticker.upper()
        ticker_options_data['underlyingPrice'] = underlying_asset_price
    
    except:
        print('Options data not available.')
        return 0

    ticker_bs_call_list = []
    for _, row in ticker_options_data.iterrows():
        S = row['underlyingPrice']
        K = row['strike']
        T = time_exp(str(row['lastTradeDate']), expiration_date)
        r = risk_free_rate
        sigma = row['impliedVolatility']
        ticker_bs_call_price = black_scholes_model(S, K, T, r, sigma)
        ticker_bs_call_list.append(ticker_bs_call_price)

    ticker_options_data['BSP'] = ticker_bs_call_list
    ticker_options_data = ticker_options_data[['ticker', 'underlyingPrice', 'lastTradeDate', 'strike', 'lastPrice', 'BSP', 
                                                'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']]

    google_articles = google_news(f'{ticker}')
    yahoo_articles = yahoo_news(f'{ticker}')
    clean_news = pd.concat([google_articles, yahoo_articles]).drop_duplicates(subset = 'headline').reset_index(drop = True)

    vader = SentimentIntensityAnalyzer()

    ticker_scores = clean_news['headline'].apply(vader.polarity_scores).tolist()
    ticker_sentiment_scores_df = pd.concat([clean_news, pd.DataFrame(ticker_scores)], axis = 1)

    ticker_sentiment_scores_df['normalized_sentiment_score'] = (ticker_sentiment_scores_df['compound'] - ticker_sentiment_scores_df['compound'].mean()) / \
        (ticker_sentiment_scores_df['compound'].std())
    
    ticker_sentiment_scores_df['weight'] = ticker_sentiment_scores_df['compound'].apply(assign_weight)
    
    ticker_options_data['adjustedVolatility'] = ticker_options_data['impliedVolatility'] \
        * (1 + ticker_sentiment_scores_df['weight'] * ticker_sentiment_scores_df['normalized_sentiment_score'].median())

    adj_bs_call_list = []
    for _, row in ticker_options_data.iterrows():
        S = row['underlyingPrice']
        K = row['strike']
        T = time_exp(str(row['lastTradeDate']), expiration_date)
        r = risk_free_rate
        sigma = row['adjustedVolatility']
        adj_bs_call_price = black_scholes_model(S, K, T, r, sigma)
        adj_bs_call_list.append(adj_bs_call_price)

    ticker_options_data['adjBSP'] = adj_bs_call_list

    final_options_data = ticker_options_data[['ticker', 'underlyingPrice', 'strike' , 'lastPrice', 'BSP', 'adjBSP', 
                                                'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'adjustedVolatility']]
    final_options_data = final_options_data.copy()
    final_options_data = final_options_data.rename({'BSP': 'orgBSP'}, axis = 1)
    
    wilcoxon_stat_adj, p_value_adj_wil = si.wilcoxon(final_options_data['adjBSP'], final_options_data['lastPrice'])
    wilcoxon_stat_org, p_value_org_wil = si.wilcoxon(final_options_data['orgBSP'], final_options_data['lastPrice'])

    final_list = [wilcoxon_stat_adj, p_value_adj_wil, wilcoxon_stat_org, p_value_org_wil]

    return final_list

expiration_date = '2023-06-16'
risk_free_interest_rate = 0.0512
ticker_list = sp500_tickers()
backtest_data = {}

for ticker in ticker_list:
    ticker_test = backtesting_bsp_model(ticker, expiration_date, risk_free_interest_rate)
    backtest_data[ticker] = ticker_test
    print(f"{ticker} Done")

df = pd.DataFrame(backtest_data).T.reset_index()
df.columns = ['ticker', 'wilcoxon_stat_adj', 'p_value_adj_wil', 'wilcoxon_stat_org', 'p_value_org_wil']

df.to_csv('wilcoxen_test.csv', index = False)
