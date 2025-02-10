import numpy as np
from numpy.typing import NDArray

from e_cert.base import BaseRegressor


class ElasticNetRegression(BaseRegressor):
    """Elastic Net回帰を行うクラス。
    L1正則化とL2正則化を組み合わせた手法で、多項式特徴量にも対応。

    Attributes:
        degree: 多項式の次数
        alpha: 正則化の強さを制御するパラメータ
        l1_ratio: L1正則化の比率 (0 <= l1_ratio <= 1)
        theta: 学習後のパラメータ
    """

    def __init__(
        self, degree: int = 1, alpha: float = 1.0, l1_ratio: float = 0.5
    ) -> None:
        """
        Args:
            degree: 多項式の次数
            alpha: 正則化の強さを制御するパラメータ
            l1_ratio: L1正則化の比率 (0 <= l1_ratio <= 1)
        """
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")

        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.theta: NDArray[np.float64] | None = None

    def _polynomial_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """入力データを多項式特徴量に変換する。

        Args:
            X: 入力データ(形状: (サンプル数, ))
        Returns:
            多項式特徴量 (形状: (サンプル数, degree+1))
        """
        X = X.reshape(-1, 1)
        X_poly = np.ones((X.shape[0], self.degree + 1), dtype=np.float64)
        for d in range(1, self.degree + 1):
            X_poly[:, d] = np.power(X[:, 0], d)
        return X_poly

    def _soft_threshold(self, x: float, lambda_: float) -> float:
        """ソフト閾値処理を行う。

        Args:
            x: 入力値
            lambda_: 閾値パラメータ
        Returns:
            処理後の値
        """
        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        return 0.0

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """パラメータthetaを学習する。

        コスト関数:
        $$J(\\theta) = \\frac{1}{2m}\\sum_{i=1}^{m}(h_\\theta(x^{(i)}) - y^{(i)})^2
                      + \\alpha \\cdot l1\\_ratio \\sum_{j=1}^{n}|\\theta_j|
                      + \\frac{\\alpha \\cdot (1-l1\\_ratio)}{2} \\sum_{j=1}^{n}\\theta_j^2$$

        Args:
            X: 入力データ(形状: (サンプル数, ))
            y: ターゲットデータ(形状: (サンプル数, ))
        """
        # ハイパーパラメータ
        self.learning_rate = 0.01
        self.max_iter = 1000
        self.tol = 1e-6

        # 多項式特徴量の計算
        X_poly = self._polynomial_features(X)
        m = X_poly.shape[0]

        # パラメータの初期化
        self.theta = np.zeros(self.degree + 1, dtype=np.float64)

        # L1とL2の正則化パラメータ
        lambda1 = self.alpha * self.l1_ratio
        lambda2 = self.alpha * (1 - self.l1_ratio)

        # コスト履歴
        self.costs: list[float] = []

        for i in range(self.max_iter):
            # 予測と誤差の計算
            y_pred = X_poly @ self.theta
            error = y_pred - y

            # コストの計算
            l1_term = lambda1 * np.sum(np.abs(self.theta[1:]))
            l2_term = (lambda2 / 2) * np.sum(self.theta[1:] ** 2)
            cost = (np.sum(error**2) / (2 * m)) + l1_term + l2_term
            self.costs.append(float(cost))

            # 勾配の計算
            gradient = (X_poly.T @ error) / m
            # L2正則化項の勾配を追加（バイアス項以外）
            gradient[1:] += lambda2 * self.theta[1:]

            # パラメータの更新
            theta_prev = self.theta.copy()
            self.theta -= self.learning_rate * gradient

            # L1正則化のソフト閾値処理（バイアス項以外）
            for j in range(1, len(self.theta)):
                self.theta[j] = self._soft_threshold(
                    self.theta[j], self.learning_rate * lambda1
                )

            # 収束判定
            if np.all(np.abs(self.theta - theta_prev) < self.tol):
                break
            if i > 0 and abs(self.costs[-1] - self.costs[-2]) < self.tol:
                break

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """予測を行う。

        Args:
            X: 入力データ(形状: (サンプル数, ))
        Returns:
            予測値(形状: (サンプル数, ))
        """
        if self.theta is None:
            raise ValueError("Model is not trained. Call fit first.")
        X_poly = self._polynomial_features(X)
        return X_poly @ self.theta
