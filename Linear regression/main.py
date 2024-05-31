import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        self.intercept = self.weights[0]

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X @ self.weights


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y = y.reshape(-1, 1)
        self.weights = np.zeros((X.shape[1], 1))
        self.weights[0] = -30
        losses = []

        for i in range(epochs):
            y_prediction = X @ self.weights
            error = y_prediction - y
            mse = np.mean(np.square(error))
            losses.append(mse)
            self.weights -= learning_rate * (2 * np.transpose(X) @ error / X.shape[0])

        self.intercept = self.weights[0].item()
        self.weights = np.squeeze(self.weights.reshape(1, -1))
        return losses

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X @ self.weights

    def plot_learning_curve(self, losses):
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.mean(np.square(prediction - ground_truth))


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=1.92 * 1e-4, epochs=150000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
