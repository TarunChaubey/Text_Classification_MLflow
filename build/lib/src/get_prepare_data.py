import pandas as pd
import numpy as np
import re
import pickle
from src.utils.HelperFunction import process_text
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

Embedding_Dim = 100
max_seq_length = 250

def get_data():
    try:
        df = pd.read_csv("./data/online_data/dataset.csv")
    except:
        df = pd.read_csv("https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv",encoding='cp1252')
        df.to_csv("./data/online_data/dataset.csv",index=False)
        print("Done")

    return df

def CleanAndLabeling():
    df = get_data()
    df['news'] = df['news'].apply(lambda x:process_text(x))
    label = pd.get_dummies(df['type'])


    max_num_words = df['news'].apply(lambda x:len(x.split())).max()

    tokenizer = Tokenizer(num_words=max_num_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(df['news'].values)
    X = tokenizer.texts_to_sequences(df['news'].values)
    X = pad_sequences(X, maxlen=max_num_words+10)
    X = tokenizer.texts_to_sequences(df['news'].values)
    text = pad_sequences(X, maxlen=max_num_words+10)
    print(max_num_words)
    return text,label 
    
def SplitData():
    X,Y = CleanAndLabeling()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
    # print(X_train.shape,Y_train.shape)
    # print(X_test.shape,Y_test.shape)
    return X_train, X_test, Y_train, Y_test

def SaveAndLoadData():
    
    X_train, X_test, Y_train, Y_test = SplitData()

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
    SaveAndLoadData()
