import nltk
nltk.download('vader_lexicon')

import streamlit as st
import plotly.express as px
import glob
from nltk.sentiment import SentimentIntensityAnalyzer

filepaths = sorted(glob.glob("diary/*.txt"))
analyzer = SentimentIntensityAnalyzer()

negative = []
positive = []
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    score = analyzer.polarity_scores(content)
    positive.append(score["pos"])
    negative.append(score["neg"])

dates = [name.strip(".txt").strip("diary/") for name in filepaths]

st.title("Diary Tone")

st.subheader("Positivity")
figure_pos = px.line(x=dates, y=positive, labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(figure_pos)

st.subheader("Negativity")
figure_neg = px.line(x=dates, y=negative, labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(figure_neg)


