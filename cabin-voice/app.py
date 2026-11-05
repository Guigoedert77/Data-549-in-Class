"""Cabin Voice: interactive stakeholder app for the DATA-549 airline tweet sentiment project.

Run locally:  streamlit run app.py
(First time only, if data/ is missing: python build_assets.py)

AI disclosure (DATA-549 policy): the web-app boilerplate, UI layout and styling of this app were
generated with the help of an AI assistant (Claude, by Anthropic) and reviewed by the team.
The background photo was generated with Canva's AI image tool.
"""
import html
import io
import json
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# run from this folder so data/, static/ and the model are found (Streamlit Cloud starts at the repo root)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from preprocess import prepare_tweet
from style import css, COLORS, ICONS

st.set_page_config(page_title='Cabin Voice', page_icon='✈️', layout='wide', initial_sidebar_state='collapsed')
st.markdown(f'<style>{css()}</style>', unsafe_allow_html=True)

ORDER = ['negative', 'neutral', 'positive']
AIRLINES = ['United', 'American', 'US Airways', 'Southwest', 'JetBlue', 'Virgin America']
VIEWS = ['Score', 'Batch', 'Benchmark', 'Compare', 'Topics', 'Model']
NAV_ICONS = [':material/chat_bubble:', ':material/upload_file:', ':material/bar_chart:',
             ':material/compare_arrows:', ':material/category:', ':material/insights:']
EXAMPLES = {
    'Lost bag': ('@united my bag is lost AGAIN and nobody at the desk will help!! 3 hours waiting', 'United'),
    'Great crew': ('@SouthwestAir thank you for the amazing crew today, best flight ever!', 'Southwest'),
    'Quick question': ('@JetBlue what time does the first flight to Boston leave tomorrow?', 'JetBlue'),
    'Stuck on hold': ('@AmericanAir been on hold for 2 hours to rebook my cancelled flight. Unacceptable.', 'American'),
}
FRIENDLY = {
    'meta__vader_compound': 'Overall tone (VADER)', 'meta__vader_neg': 'Negative words (VADER)',
    'meta__vader_pos': 'Positive words (VADER)', 'meta__vader_neu': 'Neutral words (VADER)',
    'meta__n_words': 'Tweet length', 'meta__n_exclamation': 'Exclamation marks',
    'meta__n_question': 'Question marks', 'meta__pct_upper': 'Capital letters',
}
# words that only say "this tweet is about an airline/flight"
STOP_EDA = {'flight', 'united', 'usairways', 'americanair', 'southwestair', 'jetblue', 'virginamerica', 'get',
            'would', 'fly', 'u', 'not', 'no', 'thanks', 'thank', 'go', 'one', 'make', 'flightled', 'aa', 'us',
            'airway', 'southwest', 'american', 'virgin', 'america', 'delta', 'need', 'please', 'know', 'still', 'time'}

# ----------------------------------------------------------------------------- state
st.session_state.setdefault('view', VIEWS[0])
st.session_state.setdefault('tweet', EXAMPLES['Lost bag'][0])
st.session_state.setdefault('airline', EXAMPLES['Lost bag'][1])


def go_to(view):
    st.session_state.view = view


def use_example():
    choice = st.session_state.get('example')
    if choice:
        st.session_state.tweet, st.session_state.airline = EXAMPLES[choice]


# ----------------------------------------------------------------------------- data + model
@st.cache_resource
def load_model():
    return joblib.load('sentiment_model.joblib')


@st.cache_data
def load_tweets():
    return pd.read_csv('data/tweets_clean.csv', keep_default_na=False)


@st.cache_data
def load_results():
    return json.load(open('data/model_results.json')), json.load(open('data/topics.json'))


@st.cache_data
def meta_reference():
    return json.load(open('data/meta_means.json'))


def explain(model, row, cls_idx, top_n=6):
    """Contribution of each feature to the predicted class, relative to the average class (log-odds).

    Words count from zero (word absent). The extra features (tone, length, punctuation) are measured
    against an average tweet, so a tweet of normal length does not get credit just for having words."""
    features, lr = model.named_steps['features'], model.named_steps['model']
    to_vec = lambda r: (lambda m: m.toarray().ravel() if hasattr(m, 'toarray') else np.asarray(m).ravel())(features.transform(r))
    x = to_vec(row)
    ref_row = row.copy()
    for col, v in meta_reference().items():
        ref_row[col] = v
    names = features.get_feature_names_out()
    is_meta = np.char.startswith(names.astype(str), 'meta__')
    x = np.where(is_meta, x - to_vec(ref_row), x)
    contrib = x * (lr.coef_[cls_idx] - lr.coef_.mean(axis=0))
    nz = np.nonzero(contrib)[0]
    df = pd.DataFrame({'feature': names[nz], 'value': contrib[nz]})
    df = df[~df['feature'].str.startswith('airline__')]
    df['label'] = df['feature'].map(lambda f: FRIENDLY.get(f, f'“{f.replace("tfidf__", "")}”'))
    return df.reindex(df['value'].abs().sort_values(ascending=False).index).head(top_n)


def score_many(texts, airlines):
    """Score a list of tweets with the deployed pipeline."""
    model = load_model()
    rows = pd.concat([prepare_tweet(t, a) for t, a in zip(texts, airlines)], ignore_index=True)
    proba = model.predict_proba(rows)
    classes = list(model.classes_)
    return [classes[i] for i in proba.argmax(axis=1)], proba.max(axis=1), proba, classes


def glass_plot(fig, height, legend=False):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', showlegend=legend,
                      font=dict(family='Inter, -apple-system, sans-serif', color='#FFFFFF', size=13),
                      legend=dict(orientation='h', y=1.12, x=0, bgcolor='rgba(0,0,0,0)'),
                      hoverlabel=dict(bgcolor='#2a2320', font_color='#fff'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,.12)', zerolinecolor='rgba(255,255,255,.3)', color='rgba(255,255,255,.8)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,.08)', color='#FFFFFF')
    return fig


def plot(fig, height, legend=False):
    st.plotly_chart(glass_plot(fig, height, legend), width='stretch', config={'displayModeBar': False})


def head(title, note=''):
    st.markdown(f'<div class="cv-h">{title}</div>' + (f'<div class="cv-small">{note}</div>' if note else ''),
                unsafe_allow_html=True)


def word_counts(texts):
    words = ' '.join(texts).split()
    return Counter(w for w in words if w not in STOP_EDA and len(w) > 2)


@st.cache_data
def distinctive_words(airline, n=12):
    """Words used much more by this airline's passengers than by the other airlines' (log-odds ratio)."""
    df = load_tweets()
    a, b = word_counts(df.loc[df['airline'] == airline, 'text_clean_3']), word_counts(df.loc[df['airline'] != airline, 'text_clean_3'])
    ta, tb = sum(a.values()), sum(b.values())
    scores = {w: np.log((c + 1) / ta) - np.log((b.get(w, 0) + 1) / tb) for w, c in a.items() if c >= 8}
    return sorted(scores, key=scores.get, reverse=True)[:n]


@st.cache_data
def cloud_png(airline, sentiment):
    df = load_tweets()
    sub = df[(df['airline'] == airline) & (df['airline_sentiment'] == sentiment)]
    freqs = word_counts(sub['text_clean_3'])
    if not freqs:
        return None
    cmap = {'negative': 'Oranges', 'positive': 'Greens', 'neutral': 'Greys'}[sentiment]
    wc = WordCloud(width=900, height=420, mode='RGBA', background_color=None, colormap=cmap, max_words=60,
                   prefer_horizontal=.95, random_state=42, font_path=None).generate_from_frequencies(freqs)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    return buf.getvalue()


# ----------------------------------------------------------------------------- floating nav pill
with st.container(key='navpill'):
    for view, icon in zip(VIEWS, NAV_ICONS):
        st.button(' ', icon=icon, key=f'nav_{view}', help=view, on_click=go_to, args=(view,),
                  type='primary' if st.session_state.view == view else 'secondary')

# ----------------------------------------------------------------------------- header
head_l, head_r = st.columns([5, 6], vertical_alignment='center')
with head_l:
    st.markdown('<div class="cv-title">Cabin Voice</div>'
                '<div class="cv-sub">Read what passengers are telling your airline, one tweet at a time.</div>',
                unsafe_allow_html=True)
with head_r:
    st.segmented_control('View', VIEWS, key='view', label_visibility='collapsed')
view = st.session_state.view or VIEWS[0]

with st.container(key='stats'):
    st.markdown(f"""
<div class="cv-stats">
  <div class="cv-stat"><div class="cv-ring">{ICONS['chat']}</div><div><small>Tweets learned from</small><b>14,405</b></div></div>
  <div class="cv-stat"><div class="cv-ring">{ICONS['check']}</div><div><small>Right on new tweets</small><b>80%</b></div></div>
  <div class="cv-stat"><div class="cv-ring">{ICONS['plane']}</div><div><small>Airlines covered</small><b>6 U.S. carriers</b></div></div>
</div>""", unsafe_allow_html=True)

# ============================================================================= SCORE
if view == 'Score':
    left, right = st.columns([5, 8], gap='medium')
    with left:
        with st.container(key='panel_in'):
            head('Passenger tweet')
            with st.container(key='examples'):
                st.pills('Try an example', list(EXAMPLES), key='example', on_change=use_example,
                         label_visibility='collapsed')
            st.selectbox('Airline', AIRLINES, key='airline')
            st.text_area('Tweet', key='tweet', height=150, max_chars=280, placeholder='Paste a tweet sent to the airline')
            st.button('Analyze tweet', type='primary', width='stretch')
            st.markdown('<div class="cv-small">The model reads the words, punctuation and tone of the tweet. '
                        'Trained on labeled tweets sent to six U.S. airlines.</div>', unsafe_allow_html=True)

    tweet, airline = st.session_state.tweet, st.session_state.airline
    with right:
        if not tweet.strip():
            with st.container(key='panel_result'):
                head('Paste a tweet to start', 'Type or paste a tweet on the left, or pick an example, and the reading appears here.')
        else:
            model = load_model()
            row = prepare_tweet(tweet, airline)
            proba = model.predict_proba(row)[0]
            classes = list(model.classes_)
            k = int(np.argmax(proba))
            verdict, conf = classes[k], float(proba[k])
            color = COLORS[verdict]
            circ = 2 * np.pi * 50
            with st.container(key='panel_result'):
                cards = ''.join(
                    f'<div class="cv-prob"><div class="p">{proba[classes.index(c)]:.0%}</div>'
                    f'<div class="cv-bar"><i style="width:{proba[classes.index(c)] * 100:.1f}%;background:{COLORS[c]}"></i></div>'
                    f'<span>{c.capitalize()}</span></div>' for c in ORDER)
                st.markdown(f"""
<div class="cv-verdict">
  <div><div class="cv-vlabel">This tweet reads as</div>
       <div class="cv-vword" style="color:{color}">{verdict.capitalize()}</div>
       <div class="cv-vlabel" style="margin-top:.3rem">{html.escape(airline)}, {len(tweet.split())} words</div></div>
  <svg width="128" height="128" viewBox="0 0 120 120" role="img" aria-label="Confidence {conf:.0%}">
    <circle cx="60" cy="60" r="50" stroke="rgba(255,255,255,.14)" stroke-width="10" fill="none"/>
    <circle cx="60" cy="60" r="50" stroke="{color}" stroke-width="10" fill="none" stroke-linecap="round"
            stroke-dasharray="{circ:.1f}" stroke-dashoffset="{circ * (1 - conf):.1f}" transform="rotate(-90 60 60)"/>
    <text x="60" y="62" text-anchor="middle" fill="#fff" font-size="27" font-weight="600" font-family="Inter, -apple-system">{conf:.0%}</text>
    <text x="60" y="81" text-anchor="middle" fill="rgba(255,255,255,.7)" font-size="11" font-family="Inter, -apple-system">confidence</text>
  </svg>
</div>
<div class="cv-probs">{cards}</div>""", unsafe_allow_html=True)

            ex = explain(model, row, k)
            with st.container(key='panel_why'):
                push = ex[ex['value'] > 0]['label'].head(3).tolist()
                pull = ex[ex['value'] < 0]['label'].head(2).tolist()
                sentence = f'Mostly {", ".join(push)}.' if push else 'Mostly the overall tone of the tweet.'
                if pull:
                    sentence += f' Pulling the other way: {", ".join(pull)}.'
                top = ex['value'].abs().max() or 1
                rows = ''
                for _, r in ex.iterrows():
                    if r['value'] > 0:
                        bar = f'left:35%;width:{r["value"] / top * 62:.1f}%;background:var(--accent)'
                    else:
                        bar = f'left:{35 - abs(r["value"]) / top * 33:.1f}%;width:{abs(r["value"]) / top * 33:.1f}%;background:rgba(255,255,255,.75)'
                    rows += (f'<div class="cv-row"><span class="t">{html.escape(r["label"])}</span>'
                             f'<div class="cv-track"><span class="mid"></span><i style="{bar}"></i></div></div>')
                st.markdown(f"""
<div class="cv-why"><div class="cv-h">Why this answer</div><p>{html.escape(sentence)}</p>{rows}
<div class="cv-legend"><span>Pulls away from {verdict}</span><span>Pushes toward {verdict}</span></div></div>""",
                            unsafe_allow_html=True)

# ============================================================================= BATCH
elif view == 'Batch':
    with st.container(key='panel_batch'):
        head('Score many tweets at once',
             'Upload a CSV with a <b>text</b> column (and an optional <b>airline</b> column). '
             'Every tweet gets a sentiment and a confidence, and you can download the result.')
        sample = load_tweets().sample(25, random_state=7)[['airline', 'text']]
        sample['text'] = sample['text'].map(html.unescape)
        c1, c2 = st.columns([3, 1], vertical_alignment='bottom')
        with c1:
            up = st.file_uploader('CSV file', type='csv', label_visibility='collapsed')
        with c2:
            st.download_button('Download a sample CSV', sample.to_csv(index=False), 'sample_tweets.csv', 'text/csv',
                               width='stretch')
            use_sample = st.button('Try with the sample', width='stretch')

    batch = None
    if up is not None:
        batch = pd.read_csv(up)
    elif use_sample or st.session_state.get('batch_sample'):
        st.session_state.batch_sample = True
        batch = sample.reset_index(drop=True)

    if batch is not None:
        text_col = next((c for c in batch.columns if c.lower() in ('text', 'tweet', 'tweets')), None)
        if text_col is None:
            st.error('The file needs a column called "text" (or "tweet") with the tweets. Rename the column and upload again.')
        else:
            batch = batch.dropna(subset=[text_col]).head(2000).reset_index(drop=True)
            air_col = next((c for c in batch.columns if c.lower() == 'airline'), None)
            # unknown or missing airline -> 'Unknown' (the model's airline one-hot ignores it, so no airline is assumed)
            airlines = batch[air_col].where(batch[air_col].isin(AIRLINES), 'Unknown') if air_col else ['Unknown'] * len(batch)
            with st.spinner(f'Scoring {len(batch):,} tweets…'):
                pred, conf, proba, classes = score_many(batch[text_col].astype(str).tolist(), list(airlines))
            out = batch.copy()
            out['sentiment'], out['confidence'] = pred, np.round(conf, 3)
            share = out['sentiment'].value_counts(normalize=True).reindex(ORDER, fill_value=0)

            k1, k2 = st.columns([5, 7], gap='medium')
            with k1:
                with st.container(key='panel_bsum'):
                    head('Summary', f'{len(out):,} tweets scored')
                    st.markdown('<div class="cv-probs">' + ''.join(
                        f'<div class="cv-prob"><div class="p">{share[c]:.0%}</div><div class="cv-bar"><i style="width:{share[c] * 100:.1f}%;'
                        f'background:{COLORS[c]}"></i></div><span>{c.capitalize()}</span></div>' for c in ORDER) + '</div>',
                        unsafe_allow_html=True)
                    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
                    st.download_button('Download scored CSV', out.to_csv(index=False), 'scored_tweets.csv', 'text/csv',
                                       width='stretch', type='primary')
                    worst = out[out['sentiment'] == 'negative'].sort_values('confidence', ascending=False).head(3)
                    if len(worst):
                        st.markdown('<div class="cv-h" style="font-size:1.05rem;margin-top:1rem">Most urgent complaints</div>' +
                                    ''.join(f'<div class="cv-quote">{html.escape(str(t))}</div>' for t in worst[text_col]),
                                    unsafe_allow_html=True)
            with k2:
                with st.container(key='panel_btable'):
                    head('Every tweet', 'Sorted with the most confident complaints first.')
                    show = out.sort_values(['sentiment', 'confidence'], ascending=[True, False])
                    st.dataframe(show[[c for c in ['sentiment', 'confidence', air_col, text_col] if c]], height=430,
                                 hide_index=True, width='stretch',
                                 column_config={'confidence': st.column_config.ProgressColumn('confidence', min_value=0, max_value=1, format='%.2f')})

# ============================================================================= BENCHMARK
elif view == 'Benchmark':
    df = load_tweets()
    mix = pd.crosstab(df['airline'], df['airline_sentiment'], normalize='index')[ORDER] * 100
    neg = df[df['airline_sentiment'] == 'negative']
    top_reason = neg.groupby('airline')['negativereason'].agg(lambda s: s.value_counts().idxmax())
    score = pd.DataFrame({'tweets': df['airline'].value_counts(), 'net': mix['positive'] - mix['negative'],
                          'reason': top_reason}).sort_values('net', ascending=False)

    c1, c2 = st.columns([7, 6], gap='medium')
    with c1:
        with st.container(key='panel_b1'):
            head('How passengers talk about each airline', 'Share of tweets by sentiment, February 2015. Best net sentiment on top.')
            m = mix.loc[score.index[::-1]]
            fig = go.Figure()
            for c in ORDER:
                fig.add_bar(y=m.index, x=m[c], name=c.capitalize(), orientation='h', marker_color=COLORS[c], marker_line_width=0,
                            text=[f'{v:.0f}%' for v in m[c]], textposition='inside', insidetextanchor='middle',
                            textfont=dict(color='#1b1512' if c != 'negative' else '#fff'),
                            hovertemplate='%{y}: %{x:.1f}%<extra>' + c + '</extra>')
            fig.update_layout(barmode='stack', bargap=.35)
            plot(fig, 330, legend=True)
    with c2:
        with st.container(key='panel_b2'):
            head('Scorecard', 'Net sentiment = % positive minus % negative.')
            trs = ''.join(
                f'<tr><td>{a}</td><td class="num">{r.tweets:,}</td><td class="num" style="color:{COLORS["positive"] if r.net > -30 else COLORS["negative"]}">'
                f'{r.net:+.0f}</td><td><span class="cv-pill">{html.escape(str(r.reason))}</span></td></tr>' for a, r in score.iterrows())
            st.markdown(f'<table class="cv-table"><tr><th>Airline</th><th>Tweets</th><th>Net</th><th>Top complaint</th></tr>{trs}</table>',
                        unsafe_allow_html=True)

    c3, c4 = st.columns([6, 7], gap='medium')
    with c3:
        with st.container(key='panel_b3'):
            head('Complaints spike on disruption days', 'Tweets per day (bars) and share that is negative (line). Feb 22-23 was a week of winter storms and mass cancellations.')
            daily = pd.crosstab(df['date'], df['airline_sentiment'])[ORDER]
            daily = daily[daily.sum(axis=1) > 50]
            pct = daily['negative'] / daily.sum(axis=1) * 100
            fig = go.Figure()
            for c in ORDER:
                fig.add_bar(x=daily.index, y=daily[c], name=c.capitalize(), marker_color=COLORS[c], marker_line_width=0)
            fig.add_scatter(x=daily.index, y=pct, name='% negative', yaxis='y2', mode='lines+markers',
                            line=dict(color='#fff', width=2.5), marker=dict(size=7))
            fig.update_layout(barmode='stack', yaxis2=dict(overlaying='y', side='right', range=[0, 100], showgrid=False,
                                                           color='rgba(255,255,255,.8)', ticksuffix='%'))
            plot(fig, 330, legend=True)
    with c4:
        with st.container(key='panel_b4'):
            head('What the complaints are about', "Share of each airline's negative tweets by reason (human labels).")
            reasons = pd.crosstab(neg['negativereason'], neg['airline'], normalize='columns') * 100
            reasons = reasons.loc[reasons.mean(axis=1).sort_values(ascending=False).index, score.index]
            heat = go.Figure(go.Heatmap(z=reasons.values, x=reasons.columns, y=reasons.index, text=reasons.round(0).values,
                                        texttemplate='%{text:.0f}%', textfont=dict(color='#fff', size=11),
                                        colorscale=[[0, 'rgba(255,255,255,0.04)'], [1, COLORS['accent']]], showscale=False,
                                        xgap=3, ygap=3, hovertemplate='%{x}, %{y}: %{z:.1f}%<extra></extra>'))
            heat.update_yaxes(autorange='reversed')
            plot(heat, 330)

# ============================================================================= COMPARE
elif view == 'Compare':
    df = load_tweets()
    with st.container(key='panel_pick'):
        p1, p2 = st.columns(2, gap='medium')
        a = p1.selectbox('First airline', AIRLINES, index=0, key='cmp_a')
        b = p2.selectbox('Second airline', AIRLINES, index=4, key='cmp_b')

    def kpis(name):
        sub = df[df['airline'] == name]
        s = sub['airline_sentiment'].value_counts(normalize=True).reindex(ORDER, fill_value=0) * 100
        return len(sub), s['negative'], s['positive'], s['positive'] - s['negative']

    cols = st.columns(2, gap='medium')
    for col, name, key in [(cols[0], a, 'panel_ca'), (cols[1], b, 'panel_cb')]:
        n, pn, pp, net = kpis(name)
        top_words = [w for w, _ in word_counts(df.loc[(df['airline'] == name) & (df['airline_sentiment'] == 'negative'), 'text_clean_3']).most_common(10)]
        distinct = distinctive_words(name)
        with col:
            with st.container(key=key):
                st.markdown(f"""<div class="cv-airline">{name}</div>
<div class="cv-kpis"><div class="cv-kpi"><small>Tweets</small><b>{n:,}</b></div>
<div class="cv-kpi"><small>Negative</small><b style="color:{COLORS['negative']}">{pn:.0f}%</b></div>
<div class="cv-kpi"><small>Positive</small><b style="color:{COLORS['positive']}">{pp:.0f}%</b></div>
<div class="cv-kpi"><small>Net</small><b>{net:+.0f}</b></div></div>
<div class="cv-h" style="font-size:1.05rem;margin-top:1.1rem">Most common words in complaints</div>
<div class="cv-chips">{''.join(f'<span class="cv-chip">{w}</span>' for w in top_words)}</div>
<div class="cv-h" style="font-size:1.05rem;margin-top:1rem">Words that set {name} apart</div>
<div class="cv-small">Used much more by {name} passengers than by the other airlines'.</div>
<div class="cv-chips">{''.join(f'<span class="cv-chip hot">{w}</span>' for w in distinct)}</div>""", unsafe_allow_html=True)
                tab_neg, tab_pos = st.tabs(['Complaints word cloud', 'Praise word cloud'])
                for tab, sent in [(tab_neg, 'negative'), (tab_pos, 'positive')]:
                    img = cloud_png(name, sent)
                    if img:
                        tab.image(img, width='stretch')

    with st.container(key='panel_cr'):
        head(f'{a} vs {b}: what passengers complain about', 'Share of each airline\'s negative tweets by reason.')
        neg = df[(df['airline_sentiment'] == 'negative') & (df['airline'].isin([a, b]))]
        rs = pd.crosstab(neg['negativereason'], neg['airline'], normalize='columns') * 100
        rs = rs.loc[rs.max(axis=1).sort_values().index]
        fig = go.Figure()
        for name, colr in [(a, COLORS['accent']), (b, '#E9E4DC')]:
            if name in rs:
                fig.add_bar(y=rs.index, x=rs[name], name=name, orientation='h', marker_color=colr, marker_line_width=0,
                            hovertemplate='%{y}: %{x:.1f}%<extra>' + name + '</extra>')
        fig.update_layout(barmode='group', bargap=.3)
        fig.update_xaxes(ticksuffix='%')
        plot(fig, 420, legend=True)

    with st.container(key='panel_ch'):
        head('When are passengers angriest?', '% of tweets that are negative, by hour of the day (U.S. Pacific time, as in the data).')
        fig = go.Figure()
        for name, colr in [(a, COLORS['accent']), (b, '#E9E4DC')]:
            sub = df[df['airline'] == name]
            hourly = sub.groupby('hour')['airline_sentiment'].apply(lambda s: (s == 'negative').mean() * 100)
            vol = sub['hour'].value_counts().reindex(hourly.index)
            hourly = hourly[vol >= 15]
            fig.add_scatter(x=hourly.index, y=hourly.values, name=name, mode='lines+markers', line=dict(color=colr, width=3, shape='spline'),
                            hovertemplate='%{x}:00 → %{y:.0f}% negative<extra>' + name + '</extra>')
        fig.update_xaxes(title_text='hour of the day (Pacific time)', dtick=2)
        fig.update_yaxes(ticksuffix='%')
        plot(fig, 300, legend=True)

# ============================================================================= TOPICS
elif view == 'Topics':
    df = load_tweets()
    results, topics = load_results()
    names = {int(k): v for k, v in results['topic_names'].items()}
    neg = df[(df['airline_sentiment'] == 'negative') & (df['topic'].astype(int) >= 0)].copy()
    neg['topic_name'] = neg['topic'].astype(int).map(names)

    with st.container(key='panel_t0'):
        head('What are passengers complaining about?',
             'Seven themes found by NMF topic modeling on the text of negative tweets, without using the human labels. '
             'A new tweet can be routed to the right team even before anyone reads it.')

    t1, t2 = st.columns([6, 7], gap='medium')
    with t1:
        with st.container(key='panel_t1'):
            head('Size of each theme', 'Number of negative tweets in each theme.')
            sizes = neg['topic_name'].value_counts().sort_values()
            fig = go.Figure(go.Bar(y=sizes.index, x=sizes.values, orientation='h', marker_color=COLORS['accent'], marker_line_width=0,
                                   text=sizes.values, textposition='outside', textfont=dict(color='#fff'),
                                   hovertemplate='%{y}: %{x:,} tweets<extra></extra>'))
            fig.update_xaxes(range=[0, sizes.max() * 1.18])
            plot(fig, 340)
    with t2:
        with st.container(key='panel_t2'):
            head('Which airline suffers most from each theme', "Share of each airline's negative tweets.")
            mat = pd.crosstab(neg['topic_name'], neg['airline'], normalize='columns') * 100
            mat = mat.loc[sizes.index[::-1]]
            heat = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, text=mat.round(0).values, texttemplate='%{text:.0f}%',
                                        textfont=dict(color='#fff', size=11), colorscale=[[0, 'rgba(255,255,255,0.04)'], [1, COLORS['accent']]],
                                        showscale=False, xgap=3, ygap=3, hovertemplate='%{x}, %{y}: %{z:.1f}%<extra></extra>'))
            heat.update_yaxes(autorange='reversed')
            plot(heat, 340)

    with st.container(key='panel_t3'):
        head('Look inside a theme')
        pick = st.selectbox('Theme', list(sizes.index[::-1]), label_visibility='collapsed', key='topic_pick')
        k = [i for i, v in names.items() if v == pick][0]
        sub = neg[neg['topic'].astype(int) == k]
        top_reason = sub['negativereason'].value_counts(normalize=True).head(3)
        st.markdown('<div class="cv-small">Top words the model found for this theme</div><div class="cv-chips">' +
                    ''.join(f'<span class="cv-chip hot">{w}</span>' for w in topics[str(k)]) + '</div>'
                    '<div class="cv-small" style="margin-top:.9rem">How humans labeled these tweets</div><div class="cv-chips">' +
                    ''.join(f'<span class="cv-chip">{r}: {p:.0%}</span>' for r, p in top_reason.items()) + '</div>'
                    '<div class="cv-small" style="margin-top:.9rem">Real tweets in this theme</div>' +
                    ''.join(f'<div class="cv-quote">{html.escape(html.unescape(t))}</div>' for t in sub[sub['text'].str.len() > 60].sample(min(4, len(sub)), random_state=3)['text']),
                    unsafe_allow_html=True)

# ============================================================================= MODEL
else:
    results, _ = load_results()
    t = results['test']
    with st.container(key='panel_m0'):
        st.markdown(f"""<div class="cv-h">How good is the model?</div>
<div class="cv-small">{results['source']}. Scores of the notebook's final model (tuned Linear SVM on TF-IDF + VADER + meta + BERT).</div>
<div class="cv-kpis" style="margin-top:1rem">
<div class="cv-kpi"><small>Accuracy on new tweets</small><b>{t['accuracy']:.1%}</b></div>
<div class="cv-kpi"><small>Macro F1 (main metric)</small><b>{t['macro_f1']:.3f}</b></div>
<div class="cv-kpi"><small>Ready-made dictionary (VADER)</small><b>{t['vader_macro_f1']:.3f}</b></div>
<div class="cv-kpi"><small>Gain over VADER</small><b style="color:{COLORS['positive']}">+{t['macro_f1'] - t['vader_macro_f1']:.2f}</b></div></div>""",
                    unsafe_allow_html=True)

    m1, m2 = st.columns(2, gap='medium')
    with m1:
        with st.container(key='panel_m1'):
            head('Which model won?', 'Macro F1 with 5-fold cross-validation. Grey line = VADER with no training.')
            mods = pd.DataFrame(results['models'], index=['f1', 'std']).T.sort_values('f1')
            fig = go.Figure(go.Bar(y=mods.index, x=mods['f1'], orientation='h', error_x=dict(array=mods['std'], color='rgba(255,255,255,.7)'),
                                   marker_color=[COLORS['accent'] if i == len(mods) - 1 else 'rgba(255,255,255,.55)' for i in range(len(mods))],
                                   marker_line_width=0, text=[f'{v:.3f}' for v in mods['f1']], textposition='outside', textfont=dict(color='#fff'),
                                   hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
            fig.add_vline(x=results['vader_cv'], line_dash='dash', line_color='rgba(255,255,255,.6)')
            fig.update_xaxes(range=[0.4, 0.84])
            plot(fig, 400)
    with m2:
        with st.container(key='panel_m2'):
            head('Which text features worked best?', 'Same model (Logistic Regression), different features.')
            fs = pd.Series(results['feature_sets']).sort_values()
            fig = go.Figure(go.Bar(y=fs.index, x=fs.values, orientation='h', marker_line_width=0,
                                   marker_color=[COLORS['accent'] if i == len(fs) - 1 else 'rgba(255,255,255,.55)' for i in range(len(fs))],
                                   text=[f'{v:.3f}' for v in fs.values], textposition='outside', textfont=dict(color='#fff'),
                                   hovertemplate='%{y}: %{x:.4f}<extra></extra>'))
            fig.update_xaxes(range=[0.4, 0.84])
            plot(fig, 400)

    m3, m4 = st.columns([6, 7], gap='medium')
    with m3:
        with st.container(key='panel_m3'):
            head('Where does it get confused?', 'Test set: % of each true class (rows) predicted as each class (columns).')
            cm = np.array(t['confusion'])
            pct = cm / cm.sum(axis=1, keepdims=True) * 100
            heat = go.Figure(go.Heatmap(z=pct, x=[c.capitalize() for c in ORDER], y=[c.capitalize() for c in ORDER],
                                        text=[[f'{p:.0f}%<br>({n:,})' for p, n in zip(pr, nr)] for pr, nr in zip(pct, cm)],
                                        texttemplate='%{text}', textfont=dict(color='#fff', size=13),
                                        colorscale=[[0, 'rgba(255,255,255,0.04)'], [1, COLORS['accent']]], showscale=False, xgap=4, ygap=4,
                                        hovertemplate='true %{y} → predicted %{x}: %{z:.1f}%<extra></extra>'))
            heat.update_yaxes(autorange='reversed', title_text='true')
            heat.update_xaxes(title_text='predicted')
            plot(heat, 330)
            pc = t['per_class']
            st.markdown('<div class="cv-chips">' + ''.join(f'<span class="cv-chip">{c.capitalize()} F1 {pc[c][2]:.2f}</span>' for c in ORDER) + '</div>',
                        unsafe_allow_html=True)
    with m4:
        with st.container(key='panel_m4'):
            head('Most influential words', 'Logistic Regression coefficients used by this app.')
            model = load_model()
            lr, names = model.named_steps['model'], model.named_steps['features'].get_feature_names_out()
            tabs = st.tabs([c.capitalize() for c in lr.classes_])
            for tab, k, c in zip(tabs, range(3), lr.classes_):
                s = pd.Series(lr.coef_[k], index=[FRIENDLY.get(n, n.replace('tfidf__', '')) for n in names])
                s = s[~s.index.str.startswith('airline__')].nlargest(12)[::-1]
                s.index = s.index.str.replace('fuck', 'f*ck')
                f = go.Figure(go.Bar(y=s.index, x=s.values, orientation='h', marker_color=COLORS[c], marker_line_width=0,
                                     hovertemplate='%{y}: %{x:.2f}<extra></extra>'))
                with tab:
                    plot(f, 320)

    with st.container(key='panel_m5'):
        conf = results['labeler_confidence']
        st.markdown(f"""<div class="cv-h">Good to know</div><div class="cv-why">
<p><b>Errors happen where humans disagree too.</b> The model is right on {conf['0.9-1.0'][0]:.0%} of tweets where every labeler agreed,
but only {conf['0.5-0.7'][0]:.0%} where they disagreed. Sarcasm and mixed tweets are hard for people and machines.</p>
<p><b>Which model runs here.</b> The scores above are the notebook's best model, the tuned Linear SVM that also reads BERT sentence embeddings (test macro F1 {t['macro_f1']:.3f}).
This app runs a lighter model without BERT, the tuned Logistic Regression on words, tone and punctuation (test accuracy {results['app_model_test']['accuracy']:.1%}, test macro F1 {results['app_model_test']['macro_f1']:.3f}).
It is less accurate, mostly on neutral tweets, but it answers instantly, gives a probability for each class and its word weights can be explained.
After testing, it was retrained on all 14,405 labeled tweets.</p>
<p><b>Data.</b> 14,405 tweets sent to six U.S. airlines in February 2015 (Kaggle, Twitter US Airline Sentiment). Tweets labeled
"Delta" were actually sent to JetBlue and were corrected. One stormy week of 2015 data: new slang or events may be read less well.</p>
<p class="cv-small">DATA-549 Natural Language Processing, Grand View University. App code and styling written with help from an AI
assistant (Claude, by Anthropic), cited per the course AI policy.</p></div>""", unsafe_allow_html=True)
