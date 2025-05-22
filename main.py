import numpy as np
import matplotlib.pyplot as plt
from DataGenerator import data_generator
from perceptron import Perceptron

def main():
    X, y = data_generator(m=1000, n=30, seed=42)

    split = 800
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Perceptron(n_features=30, lr=0.01)
    loss_history, acc_history = model.fit(X_train, y_train, epochs=200, verbose=False)

    plt.figure(figsize=(8,4))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.title('Training Loss and Accuracy')
    plt.plot(acc_history, label='Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
