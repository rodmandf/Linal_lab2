from DataGenerator import data_generator
from perceptron import Perceptron
import numpy as np

def main():

    X,y = data_generator(1000000,30,seed=40)

    split = 900000
    X_train,y_train = X[:split], y[:split]
    X_test,y_test = X[split:], y[split:]

    model = Perceptron(n_features=30, lr=0.1)
    model.fit(X_train,y_train,epochs=63,verbose=True)

    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()