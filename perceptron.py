import numpy as np
import math

class Perceptron:
    def __init__(self, n_features: int, lr: float = 0.1):
        self.n_features = n_features
        self.lr = lr

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z,dtype=float)

        for i, val in enumerate(z):
            out[i] = 1.0 / (1.0 + math.exp(-val))
        return out
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        z=X.dot(self.w) + self.b
        return self._sigmoid(z)
    
    def fit(self, X:np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = False) ->list:
        m,n = X.shape
        assert n == self.n_features, "Число признаков не совпадает!"

        loss_history = []
        acc_history = []
        for epoch in range(1, epochs + 1):
            y_hat = self.predict(X)
            eps = 1e-7
            y_hat_clipped = np.clip(y_hat, eps, 1 - eps)
            loss = - np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
            loss_history.append(loss)

            y_pred = (y_hat >= 0.6).astype(int)
            accuracy = np.mean(y_pred == y)
            acc_history.append(accuracy)

            errors = y_hat - y
            grad_w = X.T.dot(errors) / m
            grad_b = np.mean(errors)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            if verbose:
                print(f"Epoch {epoch}/{epochs} — loss: {loss:.4f}")

        return loss_history, acc_history

