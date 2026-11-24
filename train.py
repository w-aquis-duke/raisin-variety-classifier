import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import FEATURE_COLUMNS, download_clean_csv


def split_rows(rows, *, validation_fraction, test_fraction, seed):
    development, test = train_test_split(
        rows, test_size=test_fraction, random_state=seed,
        stratify=[row["variety"] for row in rows],
    )
    train, validation = train_test_split(
        development, test_size=validation_fraction / (1 - test_fraction), random_state=seed,
        stratify=[row["variety"] for row in development],
    )
    return train, validation, test


def model_data(rows):
    features = [[float(row[column]) for column in FEATURE_COLUMNS.values()] for row in rows]
    labels = [row["variety"] for row in rows]
    return features, labels


def main():
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    content = download_clean_csv(config["dataset_url"], config["dataset_sha256"])
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    train, validation, test = split_rows(
        rows, validation_fraction=config["validation_fraction"],
        test_fraction=config["test_fraction"], seed=config["random_seed"],
    )

    model = make_pipeline(StandardScaler(), LogisticRegression(C=config["model_c"], max_iter=1000))
    model.fit(*model_data(train))
    validation_features, validation_labels = model_data(validation)
    predictions = model.predict(validation_features)

    model_path = Path("artifacts/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_path.parent) as directory:
        temporary_model = Path(directory) / model_path.name
        joblib.dump(model, temporary_model)
        temporary_model.replace(model_path)

    print(f"Train / validation / test rows: {len(train)} / {len(validation)} / {len(test)}")
    print(f"Validation macro-F1: {f1_score(validation_labels, predictions, average='macro'):.3f}")
    print(f"Validation accuracy: {accuracy_score(validation_labels, predictions):.3f}")
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
