import re
import nltk
from nltk.corpus import stopwords

# ensure stopwords available
try:
    stopwords.words('english')[:2]
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Basic tweet cleaning."""
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+|rt', '', text)           # remove mentions + RT
    text = re.sub(r'#', '', text)                 # remove hashtags symbol
    text = re.sub(r'[^a-z\s]', '', text)          # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()      # clean spaces
    return text

def remove_stopwords(text: str) -> str:
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(tokens)

def preprocess(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
