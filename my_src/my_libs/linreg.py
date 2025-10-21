import numpy as np
import pandas as pd

class MyLineReg():
    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

        self.weights = None

    def __str__(self):
        self_dict = self.__dict__
        list_of_strings = [f'{k}={v}' for k, v in self_dict.items()]
        one_string_description = ', '.join(list_of_strings)
        return f"MyLineReg class: {one_string_description}"

    @staticmethod
    def _calc_mse_loss(y_true: np.array, y_hat: np.array):
        return np.sum(np.power(y_true - y_hat, 2))

    @staticmethod
    def _calc_mse_gradient(X: np.array, y_true: np.array, y_hat: np.array):
        n = y_true.shape[0]
        return np.dot((y_hat - y_true), X) * (2 / n)


    def predict(self, X: pd.DataFrame):
        X_with_bias = X.copy()
        X_with_bias['bias'] = 1

        X_np = X_with_bias.values

        return np.dot(X_np, self.weights)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int|bool = False):
        X_with_bias = X.copy()
        X_with_bias['bias'] = 1

        X_np = X_with_bias.values
        y_np = y.values
        weights = np.ones(X_with_bias.shape[1])

        learning_curve = []
        for i in range(self.n_iter):
            y_hat = np.dot(X_np, weights)
            loss = self._calc_mse_loss(y_true=y_np, y_hat=y_hat)
            gradient = self._calc_mse_gradient(X=X_np, y_true=y_np, y_hat=y_hat)

            weights = weights - self.learning_rate * gradient

            learning_curve.append(loss)

            # logs
            if verbose > 0:
                if i == 0:
                    print(f"start | loss: {loss}")
                elif (i % verbose) == 0:
                    print(f"{i} | loss: {loss}")
                else:
                    pass
        
        self.weights = weights
        self._learning_curve = learning_curve
        self._features_names = list(X_with_bias.columns)
        return self

    def get_coef(self):
        return self.weights[:-1]

    def _get_feature_importance(self, bias: bool = True):
        if bias:
            return pd.DataFrame({
                'feature_name': self._features_names,
                'coef': self.weights
            })
        else:
            return pd.DataFrame({
                'feature_name': self._features_names[:-1],
                'coef': self.weights[:-1]
            })

