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
        """勾配降下法を用いてパラメータthetaを求める。

        コスト関数（二乗誤差）:
        $$J(\\theta) = \\frac{1}{2m}\\sum_{i=1}^{m}(h_\\theta(x^{(i)}) - y^{(i)})^2$$

        勾配:
        $$\\frac{\\partial J}{\\partial \\theta} = \\frac{1}{m}X^T(X\\theta - y)$$

        Args:
            X: 入力データ(形状: (サンプル数, ))
            y: ターゲットデータ(形状: (サンプル数, ))
        """
        # ハイパーパラメータ
        self.learning_rate = 0.01  # 学習率
        self.max_iter = 1000  # 最大イテレーション回数
        self.tol = 1e-6  # 収束判定の閾値

        # 多項式特徴量の計算
        X_poly = self._polynomial_features(X)
        m = X_poly.shape[0]  # サンプル数

        # パラメータの初期化
        self.theta = np.zeros(self.degree + 1, dtype=np.float64)

        # コスト履歴（学習過程の確認用）
        self.costs: list[float] = []

        # 勾配降下法
        for i in range(self.max_iter):
            # 予測値の計算
            y_pred = X_poly @ self.theta

            # 誤差の計算
            error = y_pred - y

            # コスト（二乗誤差）の計算
            cost = np.sum(error**2) / (2 * m)
            self.costs.append(float(cost))

            # 勾配の計算
            gradient = (X_poly.T @ error) / m

            # パラメータの更新
            theta_prev = self.theta.copy()
            self.theta -= self.learning_rate * gradient

            # 収束判定
            # 1. パラメータの変化が小さい場合
            if np.all(np.abs(self.theta - theta_prev) < self.tol):
                break

            # 2. コストの変化が小さい場合（イテレーション2回目以降）
            if i > 0 and abs(self.costs[-1] - self.costs[-2]) < self.tol:
                break

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
