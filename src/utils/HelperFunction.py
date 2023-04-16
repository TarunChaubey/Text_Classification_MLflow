import re
import nltk
import yaml
# import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import logging
from PIL import Image

def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r','').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'[0-9]','',text)
    
    
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    
    text = " ".join(filtered_sentence)
    return text

def LoadpklData():
    with open('./data/clean_data/X_train.pkl', 'rb') as handle:
        X_train = pickle.load(handle)

    with open('./data/clean_data/X_test.pkl', 'rb') as handle:
        X_test = pickle.load(handle)

    with open('./data/clean_data/Y_test.pkl', 'rb') as handle:
        Y_test = pickle.load(handle)

    with open('./data/clean_data/Y_train.pkl', 'rb') as handle:
        Y_train = pickle.load(handle)

    return X_train, X_test, Y_train, Y_test 


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content  

# def SaveGraph(history):
#     history_dict = history.history

#     acc = history_dict['accuracy']
#     val_acc = history_dict['val_accuracy']
#     loss = history_dict['loss']
#     val_loss = history_dict['val_loss']
#     epochs = history.epoch

#     plt.figure(figsize=(10,6))
#     plt.plot(epochs, loss, 'r', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss', size=15)
#     plt.xlabel('Epochs', size=15)
#     plt.ylabel('Loss', size=15)
#     plt.legend(prop={'size': 15})
#     # plt.show()
#     # plt.savefig('./data/plots/Train_Vs_Val_Loss.png')

#     plt.figure(figsize=(10,6))
#     plt.plot(epochs, acc, 'g', label='Training acc')
#     plt.plot(epochs, val_acc, 'b', label='Validation acc')
#     plt.title('Training and validation accuracy', size=15)
#     plt.xlabel('Epochs', size=15)
#     plt.ylabel('Accuracy', size=15)
#     plt.legend(prop={'size': 15})
#     plt.ylim((0.5,1))
#     # plt.show()
#     plt.savefig('./data/plots/Train_Vs_Val_Accuracy.png')
#     # Image.open('./data/plots/Train_Vs_Val_Accuracy.png').save('./data/plots/Train_Vs_Val_Accuracy.jpg','JPEG')

#     return print("Graph Saved Successfully")


# plt.savefig('testplot.png')
# Image.open('testplot.png').save('testplot.jpg','JPEG')