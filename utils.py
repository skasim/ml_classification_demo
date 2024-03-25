import os
import operator
from gensim.models import KeyedVectors
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.

def ident_top_values_to_keep(df_categorical, df, is_train):
    if is_train:
        numerically_large_cols = ["shipper", "consignee", "notify", "vessel_name"]
        values_to_keep = {}
        for col in df_categorical.columns:
            counts = {}
            for value in df[col]:
                counts[value] = int(counts.get(value, 0)) + 1
            sorted_dict = dict( sorted(counts.items(), key=operator.itemgetter(1), reverse=True))
            if col in numerically_large_cols:
                reduced = sorted(sorted_dict.items(), key=lambda x:-x[1])[:15]
            elif col in ["country_of_origin"]:
                reduced = sorted(sorted_dict.items(), key=lambda x:-x[1])[:59]
            else:
                reduced = sorted(sorted_dict.items(), key=lambda x:-x[1])[:30]
            results = []
            for val, _ in reduced:
                results.append(val)
            values_to_keep[col] = results
        with open(os.getenv("VALUES_TO_KEEP_JSON_FILE"), "w", encoding="utf-8") as f:
            json.dump(values_to_keep, f, ensure_ascii=False, indent=4)
        return values_to_keep
    else:
        with open(os.getenv("VALUES_TO_KEEP_JSON_FILE")) as f:
            values_to_keep = json.load(f)
            return values_to_keep



def replace_low_ranking_values(df_categorical, values_to_keep):
    # replace values in columns that are not in the top most frequently occuring with OTHER
    for i, row in df_categorical.iterrows():
        for j, col in enumerate(['shipper', 'consignee', 'us_port', 'notify', 'vessel_name',
           'ship_registered_in', 'carrier_code', 'country_of_origin']):
            if row[col] not in values_to_keep[col]:
                df_categorical.at[i, col] = "OTHER"
    return df_categorical


def impute_missing_values_categorical(df_categorical):
    # impute most frequent values into missing rows in categorical columns
    for col in df_categorical.columns:
        most_frequent = df_categorical[col].mode()
        df_categorical[col] = df_categorical[col].fillna(most_frequent[0])
    return df_categorical

def impute_missing_text_data(df_product_details):
    # imput missing text value, which is the most frequent value
    most_frequent_txt = df_product_details["product_details"].mode()
    df_product_details["product_details"] = df_product_details["product_details"].fillna(most_frequent_txt[0])
    df_product_details["product_details"].isnull().sum()
    return df_product_details

def is_float(element: any) -> bool:
    # return True if value is a float
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
        
def word2vec_generator(str, word2vec, stop_words):
    count = 0
    # split string into word tokens
    tokens = word_tokenize(str)
    # remove stop words and numbers
    pattern_digits = r"[0-9]"
    vectors = []
    for word in tokens:
        if word not in stop_words and not word.isdigit() and not is_float(word):
            try:
                word = re.sub(pattern_digits, "", word)
                vec = word2vec[word.lower()]
                vectors.append(vec)
            except KeyError:
                pass
                # print(f"error with key: {w}")
    if len(vectors) == 0:
        return word2vec[re.sub(pattern_digits, "", "footwear")]
    return sum(vectors)/len(vectors)


def vectorize_product_details(google_vecs_file, df_product_details):
    word2vec = KeyedVectors.load_word2vec_format(google_vecs_file, binary=True)
    stop_words = set(stopwords.words("english"))
    vec_product_details = df_product_details.copy(deep=True)
    vec_product_details["vec_product_details"] = vec_product_details["product_details"].apply(lambda row: word2vec_generator(row, word2vec, stop_words))
    vec_product_details.drop(columns=["product_details"], inplace=True)
    return vec_product_details


def label_encode_target(df_target, label_enc_file, is_train):
    label_enc = LabelEncoder()

    if is_train:
        # fit data and save model to disk
        label_enc.fit(df_target.country_of_origin.unique())
        pickle.dump(label_enc, open(label_enc_file, "wb"))
        y = label_enc.transform(df_target.country_of_origin)
        return y
    else:
        # load model from disk and transform label 
        label_enc = pickle.load(open(label_enc_file, "rb"))
        y = label_enc.transform(df_target.country_of_origin)
        return y
    
    
def minmax_scale_numeric_data(df_numerical, y, minmax_scaler_file, is_train=False):
    minmax_scaler = MinMaxScaler()
    
    if is_train:
        # if training data then fit and save to disk
        minmax_scaler.fit(df_numerical[["weight_lb", "container_count"]], y)
        pickle.dump(minmax_scaler, open(minmax_scaler_file, "wb"))
        scaling_model = pickle.load(open(minmax_scaler_file, "rb"))
    else:
        # load the scaling model from disk
        scaling_model = pickle.load(open(minmax_scaler_file, "rb"))
    return scaling_model.transform(df_numerical[["weight_lb", "container_count"]])