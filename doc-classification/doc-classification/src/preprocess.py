"""Text cleaning utilities used by training and by the demo app."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

_ensure_nltk()
try:
    STOP = set(stopwords.words("english"))
except Exception:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP = set(ENGLISH_STOP_WORDS)

try:
    _lem = WordNetLemmatizer()
    _lem.lemmatize("tests")
    def _lemma(w): return _lem.lemmatize(w)
except Exception:
    def _lemma(w): return w  # fallback if WordNet is unavailable

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [_lemma(t) for t in text.split() if t not in STOP and len(t) > 2]
    return " ".join(tokens)
