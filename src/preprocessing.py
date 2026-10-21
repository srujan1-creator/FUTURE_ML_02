"""
src/preprocessing.py
====================
Text preprocessing, entity normalization, tokenization, and morphological lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Quietly download necessary NLTK corpora
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
ENTITY_TAGS = {'<currency>', '<num>', '<url>', '<email>'}


def preprocess_ticket_text(text: str) -> str:
    """
    Cleans, normalizes, and lemmatizes raw ticket text.
    
    Transforms:
    - Lowercasing
    - Entity normalization (URLs -> <URL>, Emails -> <EMAIL>, Currency -> <CURRENCY>, Numbers -> <NUM>)
    - Punctuation removal while preserving tag brackets < >
    - Tokenization, stopword removal, and WordNet dual-pass lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Entity Normalization
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' <EMAIL> ', text)
    text = re.sub(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?', ' <CURRENCY> ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    
    # 3. Strip non-alphanumeric symbols
    text = re.sub(r'[^a-z0-9\s<>]', ' ', text)
    tokens = text.split()
    
    # 4. Stopwords & Lemmatization
    cleaned = []
    for token in tokens:
        if token in STOP_WORDS and token not in ENTITY_TAGS:
            continue
        if len(token) <= 1 and token not in ENTITY_TAGS:
            continue
        lemma = LEMMATIZER.lemmatize(token, pos='v')
        lemma = LEMMATIZER.lemmatize(lemma, pos='n')
        cleaned.append(lemma)
        
    return " ".join(cleaned)
