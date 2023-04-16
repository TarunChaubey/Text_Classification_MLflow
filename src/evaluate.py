from tensorflow.keras.models import load_model
from src.utils.HelperFunction import LoadpklData

X_train, X_test, Y_train, Y_test  = LoadpklData()
model = load_model('./data/models/best_model.h5')
print("Evaluate on test data")
results = model.evaluate(X_test, Y_test, batch_size=64)
print(" Test Loss: {} \n Test Acc: {}".format(results[:1],results[1:]))