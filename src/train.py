from itertools import count
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.get_prepare_data import SplitData
from src.utils.HelperFunction import LoadpklData
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import argparse
import pickle
import os
import glob

epochs = 1
batch_size = 64
vocab_size = 10000
embedding_dim = 64
max_length = 256


def train_model():

    CountFile = len(glob.glob1('./data/clean_data/',"*.pkl"))
    
    if CountFile == 4:
        X_train, X_test, Y_train, Y_test = LoadpklData()
        print("Load Data From Local System")
    else:
         X_train, X_test, Y_train, Y_test = SplitData()
         print("Loading Data from Server")


    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=X_train.shape[1]))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Checkpoint used to save best model in between model training and can be reused for futher training
    es = EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience = 10)
    mc = ModelCheckpoint('./data/checkpoints/model_checkpoint.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test, Y_test),verbose = 1)
    model.save('./data/models/best_model.h5')

    print("Model Trained")
    
    # SaveGraph(history)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    print(train_model())
