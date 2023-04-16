from itertools import count
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.get_prepare_data import SplitData
from src.utils.HelperFunction import LoadpklData
from src.utils.model import LSTM
import pickle
import os
import glob

epochs = 1
batch_size = 128

def train_model():

    CountFile = len(glob.glob1('./data/clean_data/',"*.pkl"))
    
    if CountFile == 4:
        X_train, X_test, Y_train, Y_test = LoadpklData()
        print("Load Data From Local System")
    else:
         X_train, X_test, Y_train, Y_test = SplitData()
         print("Loading Data from Server")

    from keras.callbacks import EarlyStopping, ModelCheckpoint

    es = EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience = 10)
    mc = ModelCheckpoint('./data/checkpoints/model_checkpoint.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
    model = LSTM()
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test, Y_test),verbose = 1,callbacks=[es,mc])
    # print(X_train.shape,Y_train.shape)
    # print(X_test.shape,Y_test.shape)

if __name__ == '__main__':
    print(train_model())
