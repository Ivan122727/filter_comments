import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    snowball = SnowballStemmer(language="russian")
    russian_stop_words = stopwords.words("russian")

    tokens = word_tokenize(sentence, language="russian")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens
