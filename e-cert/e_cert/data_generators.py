import numpy as np
from numpy.typing import NDArray


def generate_polynomial_data(
    num_samples: int = 20, noise_level: float = 1.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """多項式回帰用のダミーデータを生成する。
    y = 2 + 1.5x - 0.3x^2 + ガウスノイズ などで生成。

    Args:
        num_samples: サンプル数
        noise_level: ノイズの強さ
    Returns:
        (X, y) のタプル
    """
    np.random.seed(42)
    X = np.linspace(-3, 3, num_samples)
    # 真の式: 2 + 1.5x - 0.3x^2
    true_y = 2 + 1.5 * X - 0.3 * (X**2)
    noise = np.random.normal(0, noise_level, size=num_samples)
    y = true_y + noise
    return X, y


def generate_logistic_data(
    num_samples: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """ロジスティック回帰用のダミーデータを生成する。
    1次元のxを適当な範囲で取り、0~1のラベルを付与する。

    Args:
        num_samples: サンプル数
    Returns:
        (X, y) のタプル。Xは shape=(num_samples, 1)
    """
    np.random.seed(42)
    X = np.linspace(-5, 5, num_samples)
    # x=0を境にラベル。ノイズを加えて境界をあいまいにする
    probs = 1.0 / (1.0 + np.exp(-X))
    # ラベル
    y = (np.random.rand(num_samples) < probs).astype(np.float64)
    # 機械学習用の形状 (num_samples, 1)
    return X.reshape(-1, 1), y
