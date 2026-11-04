"""Text preprocessing for the Cabin Voice app.

Same cleaning steps as the DATA-549 project notebook (Section 3.4 and Section 12),
so a new tweet is transformed exactly like the tweets the model was trained on.
"""
import re
import html
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

for pkg in ['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng', 'vader_lexicon']:
    nltk.download(pkg, quiet=True)

stop_words = set(stopwords.words('english'))

# Pronouns (personal, possessive, reflexive, demonstrative, interrogative, relative, indefinite)
pronouns = {
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whomever', 'whichever', 'whatever', 'anyone', 'anybody', 'anything', 'everyone', 'everybody', 'everything', 'someone', 'somebody', 'something', 'each', 'either', 'one', 'both', 'few', 'many', 'several', 'all', 'any', 'some', 'such', 'another', 'other', 'others', 'oneself',
}
extra_words = {'u', 'ur', 'im', 'amp', 'dont', 'ill', 'rt'}
negations = {'no', 'not', 'nor', 'never', 'nothing', 'nobody', 'none', 'neither', 'cannot'}
remove_words = (stop_words | pronouns | extra_words) - negations

lemmatizer = WordNetLemmatizer()
vader = SentimentIntensityAnalyzer()
PUNCT = string.punctuation + '“”'


def clean_tweet(text):
    text = html.unescape(text).lower().replace('’', "'")
    text = re.sub(r"can't|cannot", 'can not', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r"n't\b", ' not', text)
    return text


def remove_unwanted(text):
    return ' '.join(w for w in text.lower().split() if w not in remove_words)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    words = word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words)


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_extra_spaces(text):
    return ' '.join(text.split())


def prepare_tweet(text, airline):
    """Turn one raw tweet into the one-row data frame the model expects."""
    clean = clean_tweet(text).translate(str.maketrans(PUNCT, ' ' * len(PUNCT)))
    clean_3 = remove_extra_spaces(remove_numbers(lemmatize_text(remove_unwanted(clean))))
    scores = vader.polarity_scores(html.unescape(text))
    return pd.DataFrame([{
        'text_clean_3': clean_3, 'airline': airline,
        'vader_neg': scores['neg'], 'vader_neu': scores['neu'], 'vader_pos': scores['pos'],
        'vader_compound': scores['compound'],
        'n_words': len(clean.split()), 'n_exclamation': text.count('!'), 'n_question': text.count('?'),
        'pct_upper': sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
    }])
