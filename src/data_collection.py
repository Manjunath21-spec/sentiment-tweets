import snscrape.modules.twitter as sntwitter
import pandas as pd
from pathlib import Path

def scrape(query: str, limit: int = 100, out_csv: str = 'data/scraped_tweets.csv'):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append([tweet.date, tweet.user.username, tweet.content])
    df = pd.DataFrame(tweets, columns=['date', 'user', 'text'])
    Path('data').mkdir(exist_ok=True)
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df)} tweets to {out_csv}")

if __name__ == "__main__":
    scrape("python -filter:retweets", limit=50)
