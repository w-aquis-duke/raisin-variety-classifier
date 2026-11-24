# Raisin Variety Classifier

AIPI 510 project that predicts Kecimen or Besni from four shape measurements.

## Data

We keep area, perimeter, major-axis length, and minor-axis length from 900 raisins measured by İlkay Çınar, Murat Koklu, and Sakir Tasdemir. [UCI Raisin dataset](https://archive.ics.uci.edu/dataset/850/raisin) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Cleaned CSV](https://storage.googleapis.com/raisin-variety-classifier-data/raisins/clean-v1.csv)

## Run locally

Use Python 3.12.

```bash
python3 -m pip install -r requirements.txt
python3 train.py
```

Training downloads the cleaned CSV, uses a 60/20/20 split, and saves the best logistic regression pipeline to `artifacts/model.joblib`. Run `python3 dataset.py prepare` to rebuild the cleaned CSV.

## Experiments

MLflow records both validation runs and the winner’s confusion matrix. `C=0.1` won on validation and scored 0.866 macro-F1 and 86.7% accuracy on the test set.

Start the dashboard and open `http://127.0.0.1:5001`.

```bash
mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5001
```
