import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path


def load_params(params_path: str = "params.yaml") -> dict:
    """Загрузка параметров из YAML-файла."""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def main():
    # 1. Читаем параметры
    params = load_params()
    data_params = params["data"]

    raw_path = Path(data_params["raw_path"])
    processed_dir = Path(data_params["processed_dir"])
    test_size = data_params["test_size"]
    random_state = data_params["random_state"]

    # 2. Создаём папку для обработанных данных, если её нет
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 3. Загружаем сырой датасет
    df = pd.read_csv(raw_path)

    # Для Iris последний столбец — целевая переменная (species)
    # Стратифицированный сплит, чтобы классы были представлены одинаково
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df.iloc[:, -1],  # стратификация по таргету
    )

    # 4. Сохраняем train и test в data/processed
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved test to {test_path}")


if __name__ == "__main__":
    main()
