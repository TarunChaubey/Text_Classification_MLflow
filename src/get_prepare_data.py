from src.utils.HelperFunction import process_text
from src.utils.HelperFunction import read_yaml
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
import os
import pickle
import argparse

vocab_size = 10000
embedding_dim = 64
max_length = 256
trunc_type = 'post'
padding_type = 'post'

try:
    logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a")
except:
    if not os.path.isdir('./logs'):
        os.makedirs('./logs',exist_ok=True)
        logging.basicConfig(
        filename=os.path.join("logs", 'running_logs.log'), 
        level=logging.INFO, 
        format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
        filemode="a")    

def get_data():
    try:
        logging.info(f"loading data from ./data/online_data/dataset.csv")
        df = pd.read_csv("./data/online_data/dataset.csv")
    except:
        df = pd.read_csv("https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv",encoding='cp1252')
        df.to_csv("./data/online_data/dataset.csv",index=False)
        logging.info("saved csv file from cloud to local system")
        print("CSV file saved locally")

    return df

def CleanAndLabeling():
    df = get_data()
    df['news'] = df['news'].apply(lambda x:process_text(x))
    label = pd.get_dummies(df['type']).values


    tokenizer = Tokenizer(num_words=vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(df['news'].values)

    # X = tokenizer.texts_to_sequences(df['news'].values)
    # X = pad_sequences(X, maxlen=max_num_words)

    text = tokenizer.texts_to_sequences(df['news'].values)
    text = pad_sequences(text,padding=padding_type, truncating=trunc_type, maxlen=max_length)
    return text,label 
    
def SplitData():
    X,Y = CleanAndLabeling()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 42)
    return X_train, X_test, Y_train, Y_test

def SaveAndLoadData():
    
    X_train, X_test, Y_train, Y_test = SplitData()

    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    print(X_train[:1])

    print("\n")

    print(Y_train[:1])

    with open('./data/clean_data/X_train.pkl', 'wb') as handle:
        pickle.dump(X_train, handle)

    with open('./data/clean_data/X_test.pkl', 'wb') as handle:
        pickle.dump(X_test, handle)

    with open('./data/clean_data/Y_train.pkl', 'wb') as handle:
        pickle.dump(Y_train, handle)

    with open('./data/clean_data/Y_test.pkl', 'wb') as handle:
        pickle.dump(Y_test, handle)

    return X_train, X_test, Y_train, Y_test
if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    # parsed_args = args.parse_args()
    logging.info("get_prepare_data.py started running")
    SaveAndLoadData()
    logging.info("get_prepare_data.py running completed")
