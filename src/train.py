import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

# Path to dataset
DATA_PATH = os.path.join("data", "sample_tweets.csv")
MODEL_PATH = os.path.join("models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def train():
    # 1. Load data
    df = load_data()
    print("Dataset loaded:", df.shape)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"]
    )

    # 3. Convert text to TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    # 6. Save model + vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("✅ Model and vectorizer saved in 'models/'")

if __name__ == "__main__":
    train()



