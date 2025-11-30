import pandas as pd
import yaml
from pathlib import Path
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn


def load_params(params_path: str = "params.yaml") -> dict:
    """Загрузка параметров из YAML-файла."""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    # 1. Читаем параметры
    params = load_params()
    data_params = params["data"]
    model_params = params["model"]

    processed_dir = Path(data_params["processed_dir"])
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    # 2. Загружаем данные
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Последний столбец — таргет, остальные — признаки
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # 3. Создаём и обучаем модель
    model = LogisticRegression(
        C=model_params["C"],
        max_iter=model_params["max_iter"],
        random_state=model_params["random_state"],
    )

    model.fit(X_train, y_train)

    # 4. Считаем метрику
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")

    # 5. Сохраняем модель в файл
    model_path = Path("model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 6. Логируем всё в MLflow
    # Локальный трекинг в файле mlflow.db
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        # параметры модели
        mlflow.log_param("model_type", model_params["type"])
        mlflow.log_param("C", model_params["C"])
        mlflow.log_param("max_iter", model_params["max_iter"])
        mlflow.log_param("model_random_state", model_params["random_state"])

        # параметры данных
        mlflow.log_param("test_size", data_params["test_size"])
        mlflow.log_param("data_raw_path", data_params["raw_path"])
        mlflow.log_param("data_random_state", data_params["random_state"])

        # метрики
        mlflow.log_metric("accuracy", acc)

        # артефакты: модель
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(model, artifact_path="model_mlflow")

    print("Model and metrics logged to MLflow.")


if __name__ == "__main__":
    main()
