import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import joblib

from utils import tokenize_sentence
# def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
#     tokens = word_tokenize(sentence, language="russian")
#     tokens = [i for i in tokens if i not in string.punctuation]
#     if remove_stop_words:
#         tokens = [i for i in tokens if i not in russian_stop_words]
#     tokens = [snowball.stem(i) for i in tokens]
#     return tokens

if __name__ == "__main__":
    # nltk.download('punkt')
    # nltk.download('stopwords') 
    
    df = pd.read_csv("./data/labeled.csv", sep=",")
    df["toxic"] = df["toxic"].apply(int)

    train_df, test_df = train_test_split(df, test_size=500)

    snowball = SnowballStemmer(language="russian")
    russian_stop_words = stopwords.words("russian")
    
    vectorizer = TfidfVectorizer(tokenizer=tokenize_sentence)
    model_pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("model", LogisticRegression(random_state=0, C=10.))
    ])
    model_pipeline.fit(train_df["comment"], train_df["toxic"])

    prec_c_10, rec_c_10, thresholds_c_10 = precision_recall_curve(y_true=test_df["toxic"], probas_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1])


    precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[378])

    recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[378])

    joblib.dump(value=model_pipeline, filename="./pipeline.joblib", protocol=4)
