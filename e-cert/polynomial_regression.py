import io
import subprocess
import warnings
from contextlib import contextmanager
from typing import Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


class PolynomialRegression:
    """L2正則化された多項式回帰モデル。

    Parameters
    ----------
    degree : int
        多項式の次数
    alpha : float
        L2正則化の強度を制御するパラメータ
    """

    def __init__(self, degree: int = 1, alpha: float = 1.0):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.degree = degree
        self.alpha = alpha
        self.coefficients: Optional[np.ndarray] = None
        self.bias: Optional[float] = None

    def _validate_input(
        self, X: np.ndarray, t: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """入力データを検証し、適切な形状に変換する。"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be 1D or 2D array")

        if t is not None:
            t = np.asarray(t)
            if t.ndim != 1:
                raise ValueError("t must be 1D array")
            if len(t) != len(X):
                raise ValueError("Length of X and t must match")

        return X

    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """多項式特徴量を生成する。"""
        n_samples = len(X)
        X = X.reshape(n_samples, -1)

        features = [np.ones(n_samples)]
        for i in range(X.shape[1]):
            features.append(np.vander(X[:, i], self.degree + 1, increasing=True)[:, 1:])

        return np.column_stack(features)

    def fit(self, X: np.ndarray, t: np.ndarray) -> "PolynomialRegression":
        """モデルを学習する。"""
        X = self._validate_input(X, t)
        phi = self._create_polynomial_features(X)

        reg_matrix = np.eye(phi.shape[1])
        reg_matrix[0, 0] = 0

        A = phi.T @ phi + self.alpha * reg_matrix
        b = phi.T @ t

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Failed to solve the normal equation. Try increasing alpha."
            )

        self.bias = w[0]
        self.coefficients = w[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """新しいデータに対して予測を行う。"""
        if self.coefficients is None or self.bias is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = self._validate_input(X)
        phi = self._create_polynomial_features(X)
        return phi @ np.concatenate([[self.bias], self.coefficients])

    def evaluate(self, X: np.ndarray, t: np.ndarray) -> float:
        """モデルの性能を決定係数（R²スコア）で評価する。

        決定係数は以下の式で計算されます：
        R² = 1 - Σ(t_i - y_i)² / Σ(t_i - t_mean)²

        ここで：
        - t_i: 実測値
        - y_i: 予測値
        - t_mean: 実測値の平均

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            評価データ
        t : array-like of shape (n_samples,)
            目標値（実測値）

        Returns
        -------
        float
            決定係数（R²スコア）
            - 1.0: 完全な予測（最良）
            - 0.0: 定数モデル（平均値による予測）と同等
            - 負値: 平均値による予測よりも悪い予測

        Raises
        ------
        ValueError
            - モデルが未学習の場合
            - 入力データの形状が不適切な場合
        """
        if self.coefficients is None or self.bias is None:
            raise ValueError("モデルが未学習です。先にfit()を実行してください。")

        # 入力データの検証
        X = self._validate_input(X, t)

        # 予測値の計算
        y_pred = self.predict(X)

        # 平均値の計算
        t_mean = np.mean(t)

        # 全体平方和（Total Sum of Squares）の計算
        ss_total = np.sum((t - t_mean) ** 2)

        # エッジケース: すべての実測値が同じ場合
        if ss_total == 0:
            warnings.warn(
                "すべての実測値が同じ値のため、決定係数の定義が不適切です。"
                "返値は予測が完全な場合は1.0、それ以外は0.0となります。"
            )
            return 1.0 if np.allclose(t, y_pred) else 0.0

        # 残差平方和（Residual Sum of Squares）の計算
        ss_residual = np.sum((t - y_pred) ** 2)

        # 決定係数の計算
        r2_score = 1 - (ss_residual / ss_total)

        return r2_score


@contextmanager
def plot_manager():
    """プロットのリソース管理を行うコンテキストマネージャ。"""
    fig = plt.figure(figsize=(10, 6))
    try:
        yield fig
    finally:
        plt.close(fig)


def setup_japanese_fonts(font_size: int = 12) -> None:
    """日本語フォントの設定を行う関数

    Parameters
    ----------
    font_size : int
        基本フォントサイズ（デフォルト: 12）
    """
    # 優先順位付きフォントリスト
    preferred_fonts = [
        "IPAexGothic",  # Linux/macOS向けIPAフォント
        "Noto Sans CJK JP",  # Google Notoフォント
        "Hiragino Sans",  # macOS標準
        "Yu Gothic",  # Windows標準
        "Meiryo",  # Windows標準
    ]

    # 利用可能なフォントの取得
    available_fonts = set([f.name for f in fm.fontManager.ttflist])

    # 利用可能な優先フォントを検索
    selected_font = None
    for font in preferred_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # フォント設定の適用
    if selected_font:
        plt.rcParams["font.family"] = selected_font
    else:
        warnings.warn(
            "推奨日本語フォントが見つかりませんでした。システムデフォルトを使用します。"
        )

    # フォントサイズの設定
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 2,
        }
    )


def check_imgcat_availability():
    """imgcatの利用可能性をチェックする。"""
    try:
        subprocess.run(["which", "imgcat"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def display_with_imgcat(fig, fallback_to_file: bool = True):
    """matplotlib図をimgcatで表示する。

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        表示する図
    fallback_to_file : bool
        imgcat が利用できない場合にファイルに保存するかどうか
    """
    if not check_imgcat_availability():
        if fallback_to_file:
            filename = "polynomial_regression_plot.png"
            fig.savefig(filename)
            warnings.warn(f"imgcat is not available. Plot saved to {filename}")
        else:
            warnings.warn("imgcat is not available. No output generated.")
        return

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    try:
        subprocess.run(["imgcat"], input=buf.read(), check=True)
    except subprocess.CalledProcessError:
        warnings.warn("Failed to display image using imgcat")
    finally:
        buf.close()


def demonstrate_regression():
    """回帰モデルのデモンストレーションを実行する。"""
    # サンプルデータの生成
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    true_func = lambda x: 0.5 * x**2 - x + 2
    t = true_func(X.ravel()) + np.random.normal(0, 0.2, X.shape[0])

    # モデルの学習と予測
    model = PolynomialRegression(degree=2, alpha=0.1)
    model.fit(X, t)
    y_pred = model.predict(X)
    r2_score = model.evaluate(X, t)

    print(f"R^2 score: {r2_score:.4f}")

    # 結果の可視化
    with plot_manager() as fig:
        setup_japanese_fonts()
        plt.scatter(X, t, alpha=0.5, label="データ")
        plt.plot(X, y_pred, "r-", label="予測")
        plt.plot(X, true_func(X.ravel()), "g--", label="真の関数")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("L2正則化された多項式回帰による予測")
        plt.grid(True)

        display_with_imgcat(fig)


if __name__ == "__main__":
    demonstrate_regression()
