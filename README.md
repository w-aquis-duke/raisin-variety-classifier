# Raisin Variety Classifier

AIPI 510 project that predicts Kecimen or Besni from four shape measurements.

## Data

We keep area, perimeter, major-axis length, and minor-axis length from 900 raisins measured by İlkay Çınar, Murat Koklu, and Sakir Tasdemir. [UCI Raisin dataset](https://archive.ics.uci.edu/dataset/850/raisin) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Run locally

```bash
python3 dataset.py prepare
```

Upload `data/clean-v1.csv` as `raisins/clean-v1.csv` in a public-read Google Cloud Storage bucket. Set `RAISIN_DATA_URL` to the public object URL and `RAISIN_DATA_SHA256` to the hash printed during preparation.

```bash
python3 dataset.py download --url "$RAISIN_DATA_URL" --sha256 "$RAISIN_DATA_SHA256"
```
