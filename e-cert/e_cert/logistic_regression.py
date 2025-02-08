import numpy as np
from numpy.typing import NDArray

from e_cert.base import BaseRegressor


class LogisticRegression(BaseRegressor):
    """ロジスティック回帰を行うクラス。勾配降下法でパラメータを最適化する。
    Attributes:
        learning_rate: 学習率
        max_iter: 最大イテレーション回数
        theta: 学習後のパラメータ
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000) -> None:
        """
        Args:
            learning_rate: 学習率
            max_iter: 勾配降下法の最大イテレーション回数
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta: NDArray[np.float64] | None = None

    def _sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """ロジスティックシグモイド関数を計算するメソッド。
        $$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$
        オーバーフローを防ぐために clip を使用する。

        Args:
            z: 線形結合 X @ theta
        Returns:
            シグモイド関数の出力
        """
        z = np.clip(z, -500, 500)  # オーバーフロー防止策
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """勾配降下法を用いてパラメータを学習する。
        コスト関数: ログ損失
        $$J(\\theta) = -\\frac{1}{m}\\sum_{i=1}^{m}[ y^{(i)}\\log(\\hat{y}^{(i)}) + (1 - y^{(i)})\\log(1 - \\hat{y}^{(i)}) ]$$

        Args:
            X: 入力データ(形状: (サンプル数, 特徴量数))
            y: ターゲットデータ(形状: (サンプル数, ))
        """
        m, n = X.shape
        # バイアス項を計算に含めるため、Xの先頭列に1を追加
        X_bias = np.hstack([np.ones((m, 1), dtype=np.float64), X])
        # パラメータを初期化
        self.theta = np.zeros(n + 1, dtype=np.float64)

        for _ in range(self.max_iter):
            z = X_bias @ self.theta
            predictions = self._sigmoid(z)
            # 勾配の計算
            gradient = (1.0 / m) * (X_bias.T @ (predictions - y))
            # パラメータの更新
            self.theta -= self.learning_rate * gradient

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """学習結果を用いてクラスを予測する。

        Args:
            X: 入力データ(形状: (サンプル数, 特徴量数))
        Returns:
            0または1の予測ラベル(形状: (サンプル数, ))
        """
        if self.theta is None:
            raise ValueError("Model is not trained. Call fit first.")
        m, _ = X.shape
        X_bias = np.hstack([np.ones((m, 1), dtype=np.float64), X])
        probabilities = self._sigmoid(X_bias @ self.theta)
        return (probabilities >= 0.5).astype(np.float64)
