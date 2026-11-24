from contextlib import asynccontextmanager
from typing import Annotated, Literal

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dataset import FEATURE_COLUMNS


Measurement = Annotated[float, Field(gt=0, strict=True, allow_inf_nan=False)]


class RaisinMeasurements(BaseModel):
    area_px: Measurement
    perimeter_px: Measurement
    major_axis_px: Measurement
    minor_axis_px: Measurement


class RaisinPrediction(BaseModel):
    variety: Literal["Besni", "Kecimen"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load("artifacts/model.joblib")
    except FileNotFoundError as error:
        raise RuntimeError("Run python3 train.py before starting the API.") from error
    yield


app = FastAPI(title="Raisin Variety Classifier", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_measurements(_: Request, error: RequestValidationError):
    details = [{key: item[key] for key in ("loc", "msg", "type")} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


@app.post("/predict", response_model=RaisinPrediction)
def predict(measurements: RaisinMeasurements, request: Request):
    values = measurements.model_dump()
    features = [values[column] for column in FEATURE_COLUMNS.values()]
    variety = request.app.state.model.predict([features])[0]
    return {"variety": str(variety)}
