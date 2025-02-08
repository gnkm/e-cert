from sklearn.metrics import r2_score

from e_cert.data_generators import generate_logistic_data, generate_polynomial_data
from e_cert.logistic_regression import LogisticRegression
from e_cert.polynomial_regression import PolynomialRegression


def demonstrate_polynomial_regression() -> float:
    """
    多項式回帰の学習と予測を行い、結果をCLI上に出力するデモ関数。

    Returns:
        Tuple[float, Optional[float]]: (R²スコア, 交差検証の平均R²スコア)
    """
    try:
        # ダミーデータ生成
        X, y = generate_polynomial_data(num_samples=20, noise_level=1.0)

        # モデル作成 (2次の多項式回帰)
        model = PolynomialRegression(degree=2)
        model.fit(X, y)

        # 予測
        predictions = model.predict(X)

        # R²スコアを計算
        r2 = r2_score(y, predictions)

        # 結果を表示
        print("=== Polynomial Regression Demonstration ===")
        print(f"Model Performance:")
        print(f"R² Score: {r2:.4f}")
        print("\nDetailed Predictions:")
        print("Sample | X | Actual Y | Predicted Y")
        for i in range(len(X)):
            print(f"{i:6d} | {X[i]:+.3f} | {y[i]:+.3f} | {predictions[i]:+.3f}")

        print("")
        return r2

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        print("")
        return 0.0


def demonstrate_logistic_regression() -> None:
    """ロジスティック回帰の学習と予測を行い、結果をCLI上に出力するデモ関数。"""
    # ダミーデータの生成
    X, y = generate_logistic_data(num_samples=30)

    # モデル作成
    logreg = LogisticRegression(learning_rate=0.1, max_iter=1000)
    logreg.fit(X, y)

    # 予測
    pred_labels = logreg.predict(X)

    # 正解率を計算
    accuracy = (pred_labels == y).mean()

    # 結果を簡易表示
    print("=== Logistic Regression Demonstration ===")
    print("Sample | X | Actual Label | Predicted Label")
    for i in range(len(X)):
        print(
            f"{i:6d} | {X[i][0]:+.3f} | {int(y[i])}            | {int(pred_labels[i])}"
        )
    print(f"Accuracy: {accuracy:.3f}")
    return accuracy
