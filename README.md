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

## Test the API

After training, start the API with `python3 -m uvicorn app:app`. Open [the prediction form](http://127.0.0.1:8000/) or [the API docs](http://127.0.0.1:8000/docs). To test the API directly, run this in a second terminal.

```bash
curl -sS http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"area_px":87524,"perimeter_px":1184.04,"major_axis_px":442.25,"minor_axis_px":253.29}'
```

## Docker

```bash
docker build -t raisin-classifier .
mkdir -p artifacts/docker
docker run --rm -v "$PWD/artifacts/docker:/app/artifacts" raisin-classifier python train.py
docker run --rm -p 127.0.0.1:8000:8080 -v "$PWD/artifacts/docker:/app/artifacts:ro" raisin-classifier
```

Use the same `curl` request to test the container.

## Experiments

MLflow records both validation runs and the winner’s confusion matrix. `C=0.1` won on validation and scored 0.866 macro-F1 and 86.7% accuracy on the test set.

Start the dashboard and open `http://127.0.0.1:5001`.

```bash
mlflow server --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5001
```
