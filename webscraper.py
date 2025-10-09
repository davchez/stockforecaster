
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pandas as pd

def scrape_finviz_news(ticker):
    """
    Parameters:
    - ticker (str): Name of specific stock to be analyzed
    """
    url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    page = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(page.text, 'lxml')
    news_table = soup.find('table', {'class': 'fullview-news-outer'})
    news_rows = news_table.find_all('tr', {'class': 'cursor-pointer'})
    return news_rows

def analyze_sentiment(headline):
    scores = SentimentIntensityAnalyzer().polarity_scores(headline)
    return scores['compound']

def clean_data(news_rows: list, ticker: str):
    """
    Parameters:
    - news_rows (list): List of news data meant to be inserted from scrape_finviz_news
    - ticker (str): Name of specific stock to extract from news_rows
    """
    news_data = []
    current_day = 'Today'

    for row in news_rows:
        headline_link = row.find('a', {'class': 'tab-link-news'})

        if headline_link:
            headline = headline_link.text
            link = headline_link.get('href')

            if ticker.upper() not in headline.upper(): 
                continue

            score = analyze_sentiment(headline)

            date_cell = row.find('td', {'align': 'right'})
            date_time = date_cell.text.strip() if date_cell else 'N/A'

            date_list = date_time.split(" ")

            if len(date_list) == 2:
                current_day = date_list[0]
                full_datetime = date_time
            else:
                full_datetime = f"{current_day} {date_time}" if current_day else date_time

            news_data.append({
                'datetime': full_datetime,
                'headline': headline,
                'link': link,
                'sentiment': score
        })
    
    return pd.DataFrame(news_data)

def calculate_sentiment(df: pd.DataFrame, adjust: bool):
    """
    Parameters:
    - df (pd.DataFrame): DataFrame meant to be inserted after clean_data DataFrame produced
    - adjust (bool): Boolean flag whether or not sentiment scores include scores of 0 or not
    """
    if adjust:
        return df[df['sentiment'] != 0]['sentiment'].mean()
    return df['sentiment'].mean()