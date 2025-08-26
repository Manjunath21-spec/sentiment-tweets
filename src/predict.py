import joblib
import os

MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

def predict_sentiment(text):
    # Load model + vectorizer
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Transform input
    X = vectorizer.transform([text])

    # Predict
    prediction = clf.predict(X)[0]
    return prediction

if __name__ == "__main__":
    # Test with some examples
    samples = [
        "I love this product! It's amazing 😍",
        "This is the worst thing I’ve ever bought 😡",
        "Not bad, but could be better."
    ]
    
    for s in samples:
        print(f"Tweet: {s}")
        print(f"Predicted Sentiment: {predict_sentiment(s)}\n")
import joblib

# Load trained model + vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

while True:
    tweet = input("Enter a tweet (or 'quit' to exit): ")
    if tweet.lower() == "quit":
        break
    X = vectorizer.transform([tweet])
    prediction = model.predict(X)[0]
    print(f"Predicted Sentiment: {prediction}\n")
