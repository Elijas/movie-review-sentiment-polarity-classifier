"""
Functions related to preprocessing datasets
"""
from typing import List

import nltk

nltk.download('stopwords', quiet=True, raise_on_error=True)
NLTK_STOP_WORDS = list(set(nltk.corpus.stopwords.words('english')))

nltk.download('punkt', quiet=True, raise_on_error=True)
nltk_porter_stemmer = nltk.stem.PorterStemmer()
TOKENIZED_STOP_WORDS = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))


def normalize_word(word):
    return nltk_porter_stemmer.stem(word)


def process_string_to_tokens(string: str, remove_stop_words=True, normalize_word_fun=normalize_word) -> List[str]:
    tokens = nltk.word_tokenize(string)
    # Ensure lower-case characters
    tokens = (token.lower() for token in tokens)
    # Remove words with non-alphabet characters
    tokens = (token for token in tokens if token.isalpha())
    # Remove stop words
    if remove_stop_words:
        tokens = (token for token in tokens if token not in NLTK_STOP_WORDS)
    # Stem words
    tokens = (normalize_word_fun(token) for token in tokens)
    return list(tokens)


def process_strings_to_token_lists(strings: List[str]) -> List[List[str]]:
    token_lists = (process_string_to_tokens(line) for line in strings)
    return [tokens for tokens in token_lists if tokens]  # Remove empty lists
