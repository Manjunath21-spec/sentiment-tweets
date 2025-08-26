import streamlit as st
import joblib
import os

# Load trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Streamlit UI
st.title("💬 Sentiment Analysis on Tweets")
st.write("Enter a tweet below and get its sentiment (Positive / Negative / Neutral).")

# Input box
tweet = st.text_area("✍️ Enter your tweet here:")

# Predict button
if st.button("Analyze Sentiment"):
    if tweet.strip() != "":
        X = vectorizer.transform([tweet])
        prediction = model.predict(X)[0]

        if prediction == "positive":
            st.success("✅ Sentiment: Positive 😊")
        elif prediction == "negative":
            st.error("❌ Sentiment: Negative 😡")
        else:
            st.info("😐 Sentiment: Neutral")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")

