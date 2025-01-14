import sys

import matplotlib.pyplot as plt
import numpy as np
from imgcat import imgcat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LogisticRegression:
    """ロジスティック回帰による二値分類器

    Attributes:
        learning_rate (float): 学習率
        num_iterations (int): 学習の繰り返し回数
        weights (ndarray): 学習された重み
        bias (float): バイアス項
        training_history (list): 学習過程での損失値の履歴
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """初期化メソッド

        Args:
            learning_rate (float): 学習率（デフォルト: 0.01）
            num_iterations (int): 学習の繰り返し回数（デフォルト: 1000）
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.training_history = []

    def sigmoid(self, z):
        """シグモイド関数

        Args:
            z (ndarray): 入力値

        Returns:
            ndarray: シグモイド関数による変換値
        """
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # オーバーフロー対策

    def fit(self, X, y):
        """モデルの学習を行う

        Args:
            X (ndarray): 形状[n_samples, n_features]の訓練データ
            y (ndarray): 形状[n_samples]の教師ラベル（0または1）

        Returns:
            self: 学習済みモデル
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("入力データはnumpy配列である必要があります")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.num_iterations):
            # 予測確率の計算
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # 勾配の計算
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # パラメータの更新
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 損失の記録
            loss = -np.mean(
                y * np.log(y_predicted + 1e-15)
                + (1 - y) * np.log(1 - y_predicted + 1e-15)
            )
            self.training_history.append(loss)

        return self

    def predict_proba(self, X):
        """確率予測を行う

        Args:
            X (ndarray): 形状[n_samples, n_features]の入力データ

        Returns:
            ndarray: クラス1に属する確率
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """クラス予測を行う

        Args:
            X (ndarray): 形状[n_samples, n_features]の入力データ

        Returns:
            ndarray: 予測クラス（0または1）
        """
        probas = self.predict_proba(X)
        return (probas > 0.5).astype(int)

    def evaluate(self, X, y_true):
        """モデルの評価を行う

        Args:
            X (ndarray): 評価データ
            y_true (ndarray): 真のラベル

        Returns:
            dict: 各種評価指標
        """
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred),
        }

    def plot_training_history(self):
        """学習履歴をプロットする"""
        try:
            # グラフの作成と保存
            plt.plot(self.training_history)
            plt.title("Training Loss History")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            file_path = "training_history.png"
            plt.savefig(file_path)
            plt.close()

            # 画像ファイルをバイナリモードで読み込んでimgcatで表示
            try:
                with open(file_path, "rb") as f:
                    imgcat(f.read())
            except Exception as e:
                print(f"画像の表示に失敗しました: {e}")
                print(f"グラフは'{file_path}'として保存されています")

        except Exception as e:
            print(f"グラフの作成中にエラーが発生しました: {e}")


# 使用例
if __name__ == "__main__":
    # サンプルデータの生成
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # データセットの生成
    X, y = make_classification(
        n_samples=100,  # サンプル数
        n_features=2,  # 特徴量の総数
        n_classes=2,  # クラス数
        n_informative=2,  # 情報を持つ特徴量の数
        n_redundant=0,  # 冗長な特徴量の数
        n_repeated=0,  # 繰り返される特徴量の数
        random_state=42,  # 乱数シード
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの学習
    model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)

    # 評価
    evaluation = model.evaluate(X_test, y_test)
    print("評価結果:")
    print(f"正解率: {evaluation['accuracy']:.3f}")
    print("\n混同行列:")
    print(evaluation["confusion_matrix"])
    print("\n分類レポート:")
    print(evaluation["classification_report"])

    # 学習履歴の表示
    model.plot_training_history()
    sys.exit()
