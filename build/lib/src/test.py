from src.utils.HelperFunction import  LoadpklData

X_train, X_test, Y_train, Y_test = LoadpklData()
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)