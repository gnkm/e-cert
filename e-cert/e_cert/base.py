import abc

import numpy as np
from numpy.typing import NDArray


class BaseRegressor(abc.ABC):
    """学習器としての基底クラス。fit, predict メソッドのインタフェースを定義する。"""

    @abc.abstractmethod
    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """モデルのパラメータを学習する抽象メソッド。"""
        pass

    @abc.abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """学習済みモデルで予測する抽象メソッド。"""
        pass
