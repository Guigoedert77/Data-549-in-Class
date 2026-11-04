# Cabin Voice: airline tweet sentiment app (DATA-549)

Interactive stakeholder application for the DATA-549 NLP project (Twitter US Airline Sentiment).
A social-media or customer-experience manager pastes a tweet, picks the airline, and sees:

* the predicted sentiment (negative / neutral / positive) and the model's confidence
* the probability of each class
* **why** the model decided: the words and signals that pushed the answer (model interpretation)
* **Batch**: upload a CSV of tweets, get every tweet scored, download the result
* **Benchmark**: sentiment mix, scorecard, daily complaint spikes and complaint reasons by airline
* **Compare**: two airlines side by side (KPIs, most common and most distinctive words, word clouds, complaint reasons, angriest hours)
* **Topics**: the 7 complaint themes found by NMF topic modeling, by airline, with real example tweets
* **Model**: model comparison, feature comparison, confusion matrix, per-class F1, most influential words

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Keep these files in the same folder:

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `preprocess.py` | Same text cleaning as the project notebook (text_clean -> text_clean_3, VADER, meta features) |
| `style.py` | Liquid-glass styling |
| `sentiment_model.joblib` | Trained pipeline exported by Section 12 of the notebook (TF-IDF + VADER + meta -> tuned Logistic Regression) |
| `Tweets.csv` | Original Kaggle data |
| `data/` | Pre-computed cleaned tweets, complaint topics and notebook results (rebuild with `python build_assets.py`) |
| `build_assets.py` | Rebuilds `data/` from `Tweets.csv` with the notebook's cleaning and NMF settings |
| `static/bg.jpg`, `static/fonts/` | Background photo and Inter font (fallback when SF Pro is not installed) |
| `.streamlit/config.toml` | Theme and static file serving |

To refresh the model, run the notebook again and copy the new `app/sentiment_model.joblib` here.

## Contribution of AI tools

Following the DATA-549 AI policy: the web-app boilerplate, UI layout and styling were generated with the help of
an AI assistant (Claude, by Anthropic) and reviewed by the team. The background photo was generated with
Canva's AI image tool. The model, data cleaning and evaluation come from the team's project notebook.
