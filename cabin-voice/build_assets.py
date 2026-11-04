"""Pre-computes the data the Cabin Voice app needs (run once: python build_assets.py).

Uses the same cleaning as the project notebook, then:
  * saves cleaned tweets with their complaint topic  -> data/tweets_clean.csv
  * saves the complaint-topic top words               -> data/topics.json
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_tweet, remove_unwanted, lemmatize_text, remove_numbers, remove_extra_spaces, PUNCT

DOMAIN_WORDS = ['flight', 'united', 'usairways', 'americanair', 'southwestair', 'jetblue', 'virginamerica',
                'get', 'would', 'fly', 'u']

df = pd.read_csv('Tweets.csv')
df = df.drop_duplicates(subset='tweet_id').drop_duplicates(subset='text').reset_index(drop=True)
df['airline'] = df['airline'].replace({'Delta': 'JetBlue'})
df['tweet_created'] = pd.to_datetime(df['tweet_created'].str[:19])
df['date'] = df['tweet_created'].dt.date.astype(str)
df['hour'] = df['tweet_created'].dt.hour
df['negativereason'] = df['negativereason'].fillna('No complaint')

clean = df['text'].fillna('').apply(clean_tweet).str.translate(str.maketrans(PUNCT, ' ' * len(PUNCT)))
df['text_clean_3'] = clean.apply(remove_unwanted).apply(lemmatize_text).apply(remove_numbers).apply(remove_extra_spaces)
df = df[df['text_clean_3'] != ''].reset_index(drop=True)

# ---- complaint topics: NMF on negative tweets (same settings as notebook Section 7.5)
neg = df['airline_sentiment'] == 'negative'
vec = TfidfVectorizer(max_df=0.5, min_df=5,
                      stop_words=DOMAIN_WORDS + ['not', 'no', 'never', 'thanks', 'still', 'need', 'go', 'time', 'one'])
X = vec.fit_transform(df.loc[neg, 'text_clean_3'])
nmf = NMF(n_components=7, random_state=42, max_iter=500, init='nndsvda')
W = nmf.fit_transform(X)
df['topic'] = -1
df.loc[neg, 'topic'] = W.argmax(axis=1)
names = vec.get_feature_names_out()
topics = {int(k): [names[j] for j in comp.argsort()[::-1][:10]] for k, comp in enumerate(nmf.components_)}

os.makedirs('data', exist_ok=True)
cols = ['tweet_id', 'airline', 'airline_sentiment', 'airline_sentiment_confidence', 'negativereason',
        'text', 'text_clean_3', 'date', 'hour', 'topic']
df[cols].to_csv('data/tweets_clean.csv', index=False)
json.dump(topics, open('data/topics.json', 'w'), indent=1)
for k, words in topics.items():
    print(k, int((df['topic'] == k).sum()), ', '.join(words))
