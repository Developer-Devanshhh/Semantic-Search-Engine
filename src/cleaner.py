# cleaner.py — text cleaning for 20 Newsgroups Usenet data
#
# sklearn's remove=('headers','footers','quotes') gets ~80% of the noise,
# but leaves residual headers, emails, and paths that bias the embeddings
# toward metadata. This module handles the rest.

import re


# compiled once — these get called ~20K times during preprocessing
_HEADERS = re.compile(
    r'^(From|Subject|Organization|Lines|NNTP-Posting-Host|Reply-To|'
    r'Distribution|Sender|X-[\w-]+|References|Message-ID|Date|Newsgroups|Path):.*$',
    re.MULTILINE | re.IGNORECASE
)
_EMAIL       = re.compile(r'[\w\.\-+]+@[\w\.\-]+\.\w+')
_URL         = re.compile(r'https?://\S+|www\.\S+|ftp://\S+')
_QUOTED      = re.compile(r'^[>|]+\s?.*$', re.MULTILINE)
_ARTICLE_REF = re.compile(r'In article\s*<[^>]+>.*$', re.MULTILINE | re.IGNORECASE)
_SIGNATURE   = re.compile(r'\n--\s*\n.*', re.DOTALL)
_FILEPATH    = re.compile(r'(?:/[\w\.\-]+){2,}')
_MULTI_NL    = re.compile(r'\n{3,}')
_MULTI_SPACE = re.compile(r'[ \t]+')


def clean_document(text):
    """Strip Usenet noise from a single document. Returns empty string if input is junk."""
    if not text or not text.strip():
        return ""

    text = _HEADERS.sub('', text)
    text = _EMAIL.sub('', text)
    text = _URL.sub('', text)
    text = _QUOTED.sub('', text)
    text = _ARTICLE_REF.sub('', text)
    text = _SIGNATURE.sub('', text)
    text = _FILEPATH.sub('', text)

    text = _MULTI_NL.sub('\n\n', text)
    text = _MULTI_SPACE.sub(' ', text)

    # drop tiny residual lines (punctuation, initials)
    text = '\n'.join(line for line in text.split('\n') if len(line.strip()) >= 3)
    return text.strip()


def is_valid(text, min_len=50, max_len=5000):
    """Quality gate — too short means noise-only, too long skews cluster geometry."""
    n = len(text)
    return min_len <= n <= max_len


def clean_corpus(texts):
    """
    Clean and filter a full corpus.
    Returns (cleaned_texts, original_indices) for traceability.
    """
    cleaned, indices = [], []
    for i, raw in enumerate(texts):
        doc = clean_document(raw)
        if is_valid(doc):
            cleaned.append(doc)
            indices.append(i)
    return cleaned, indices
