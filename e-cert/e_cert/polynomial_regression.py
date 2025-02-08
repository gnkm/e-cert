import numpy as np
from numpy.typing import NDArray

from e_cert.base import BaseRegressor


class PolynomialRegression(BaseRegressor):
    """多項式回帰を行うクラス。正規方程式を用いてパラメータを求める。
    Attributes:
        degree: 多項式の次数
        theta: 学習後のパラメータ
    """

    def __init__(self, degree: int = 2) -> None:
        """
        Args:
            degree: 多項式の次数
        """
        self.degree = degree
        self.theta: NDArray[np.float64] | None = None

    def _polynomial_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """入力データを多項式特徴量に変換するプライベートメソッド。
        例：X = [x1, x2, ...] -> [[1, x1, x1^2], [1, x2, x2^2], ...]

        Args:
            X: 入力データ(形状: (サンプル数, )) 1次元を想定。
        Returns:
            X_poly: 多項式特徴量 (形状: (サンプル数, degree+1))
        """
        # 2次元配列にしておく
        X = X.reshape(-1, 1)
        # 1列目を1(バイアス項)にし、それ以降に x^1, x^2, ... を追加
        X_poly = np.ones((X.shape[0], self.degree + 1), dtype=np.float64)
        for d in range(1, self.degree + 1):
            X_poly[:, d] = np.power(X[:, 0], d)
        return X_poly

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """正規方程式を用いてパラメータthetaを求める。
        $$\\theta = (X^T X)^{-1} X^T y$$

        Args:
            X: 入力データ(形状: (サンプル数, ))
            y: ターゲットデータ(形状: (サンプル数, ))
        """
        X_poly = self._polynomial_features(X)
        # 正規方程式(単純実装)
        self.theta = np.linalg.inv(X_poly.T @ X_poly) @ (X_poly.T @ y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """学習結果を用いて予測値を計算する。

        Args:
            X: 入力データ(形状: (サンプル数, ))
        Returns:
            予測値(形状: (サンプル数, ))
        """
        if self.theta is None:
            raise ValueError("Model is not trained. Call fit first.")
        X_poly = self._polynomial_features(X)
        return X_poly @ self.theta
