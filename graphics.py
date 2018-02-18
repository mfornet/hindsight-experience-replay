import matplotlib.pyplot as plt
import numpy as np

from her import evaluate_actor
from model import QModel

TRAINED_MODELS = 14

def success_curve():
    X = np.arange(2, TRAINED_MODELS)
    y_her = np.zeros(X.shape)
    y_nher = np.zeros(X.shape)

    for idx, x in enumerate(X):
        model = QModel(x, True)
        model.load()

        success_rate = evaluate_actor(model)
        y_her[idx] = success_rate

        model = QModel(x, False)
        model.load()

        success_rate = evaluate_actor(model)
        y_nher[idx] = success_rate

    plt.plot(X, y_her)
    plt.plot(X, y_nher)

    plt.ylim(-1e-1, 1+1e-1)

    # plt.legend(["HER", "No HER"])

    plt.show()


if __name__ == '__main__':
    success_curve()
