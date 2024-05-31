import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        num_feature = inputs.shape[1]

        self.weights = np.zeros(num_feature)
        self.intercept = 0

        for i in range(self.num_iterations):

            a = inputs @ self.weights + self.intercept
            sigma_a = self.sigmoid(a)
            error = sigma_a - targets
            self.weights -= self.learning_rate * (inputs.T @ error)
            self.intercept -= self.learning_rate * np.sum(error)

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        a = inputs @ self.weights + self.intercept
        pred_prob = self.sigmoid(a)
        pred_class = np.where(pred_prob >= 0.5, 1, 0)

        return pred_prob, pred_class

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        class_0 = inputs[targets == 0]
        class_1 = inputs[targets == 1]

        self.m0 = np.mean(class_0, axis=0)
        self.m1 = np.mean(class_1, axis=0)
        self.m0 = self.m0.reshape(-1, 1)
        self.m1 = self.m1.reshape(-1, 1)

        self.sb = (self.m1 - self.m0) @ np.transpose((self.m1 - self.m0))
        self.m0 = np.squeeze(self.m0)
        self.m1 = np.squeeze(self.m1)
        sw1 = np.transpose(class_0 - self.m0) @ (class_0 - self.m0)
        sw2 = np.transpose(class_1 - self.m1) @ (class_1 - self.m1)
        self.sw = sw1 + sw2
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        self.slope = self.w[1] / self.w[0]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        proj = inputs @ self.w
        pred = (proj > np.median(proj)).astype(int)
        return pred

    def plot_projection(self, inputs: npt.NDArray[float]):
        intercept = -2
        plt.plot([inputs[:, 0].min(), inputs[:, 0].max()],
                 [self.slope * inputs[:, 0].min() + intercept, self.slope * inputs[:, 0].max() + intercept],
                 color='red', linestyle='--',
                 label=f'Projection line: slope={self.slope:.2f},  intercept={intercept:.2f}')

        pred = self.predict(inputs)
        plt.scatter(inputs[:, 0], inputs[:, 1], c=pred, cmap='viridis')
        plt.legend()
        plt.show()


def compute_auc(y_trues, y_preds) -> float:
    auc_score = roc_auc_score(y_trues, y_preds)
    return auc_score


def accuracy_score(y_trues, y_preds) -> float:
    correct = 0
    for i in range(len(y_trues)):
        if y_trues[i] == y_preds[i]:
            correct += 1

    accuracy = correct / len(y_trues)
    return accuracy


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-4,  # You can modify the parameters as you want
        num_iterations=10000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()
