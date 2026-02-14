import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# cached downloads (safe to call on startup)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

_EN_STOP = set(stopwords.words("english"))

def text_to_gloss(text: str) -> list[str]:
    """
    Baseline English gloss:
    - tokenize
    - keep alphabetic tokens
    - remove stopwords
    - uppercase
    """
    text = (text or "").lower().strip()
    tokens = word_tokenize(text)

    out: list[str] = []
    for t in tokens:
        if t.isalpha() and t not in _EN_STOP:
            out.append(t.upper())

    return out
