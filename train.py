import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import yaml
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
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


def scores(labels, predictions, split):
    return {
        f"{split}_macro_f1": f1_score(labels, predictions, average="macro"),
        f"{split}_accuracy": accuracy_score(labels, predictions),
    }


def main():
    config_path = Path("config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    content = download_clean_csv(config["dataset_url"], config["dataset_sha256"])
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    train, validation, test = split_rows(
        rows, validation_fraction=config["validation_fraction"],
        test_fraction=config["test_fraction"], seed=config["random_seed"],
    )

    print(f"Train / validation / test rows: {len(train)} / {len(validation)} / {len(test)}")
    train_features, train_labels = model_data(train)
    validation_features, validation_labels = model_data(validation)

    model_path = Path("artifacts/model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
    experiment_name = "raisin-training"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location="artifacts/mlruns")
    mlflow.set_experiment(experiment_name)

    runs = []
    for c in (config["model_c"], config.get("comparison_c", 0.1)):
        with mlflow.start_run(run_name=f"C={c:g}") as run:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
            model.fit(train_features, train_labels)
            validation_scores = scores(validation_labels, model.predict(validation_features), "validation")
            mlflow.log_params({
                "C": c,
                "random_seed": config["random_seed"],
                "dataset_sha256": config["dataset_sha256"],
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            })
            mlflow.log_metrics(validation_scores)
            mlflow.log_artifact(str(config_path))
            macro_f1 = validation_scores["validation_macro_f1"]
            accuracy = validation_scores["validation_accuracy"]
            runs.append((macro_f1, run.info.run_id, model))
            print(f"C={c:g} validation macro-F1: {macro_f1:.3f}, accuracy: {accuracy:.3f}")

    _, run_id, winner = max(runs, key=lambda result: result[0])
    test_features, test_labels = model_data(test)
    predictions = winner.predict(test_features)
    test_scores = scores(test_labels, predictions, "test")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_scores)
        figure = Figure(figsize=(5, 4), layout="constrained")
        ConfusionMatrixDisplay.from_predictions(test_labels, predictions, ax=figure.subplots())
        mlflow.log_figure(figure, "confusion-matrix.png")
        with TemporaryDirectory(dir=model_path.parent) as directory:
            temporary_model = Path(directory) / model_path.name
            joblib.dump(winner, temporary_model)
            mlflow.log_artifact(str(temporary_model))
            temporary_model.replace(model_path)
        mlflow.set_tag("selected", "true")

    print(f"Test macro-F1: {test_scores['test_macro_f1']:.3f}")
    print(f"Test accuracy: {test_scores['test_accuracy']:.3f}")
    print(f"Selected MLflow run: {run_id}")
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
