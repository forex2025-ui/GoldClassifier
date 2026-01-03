from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.model_io import load_model
from src.data_loader import load_data
from src.feature_engineering import add_features
from src.config import FEATURE_COLUMNS, DATA_PATH

app = FastAPI(title="Gold Price Direction Classifier")

templates = Jinja2Templates(directory="templates")

# Load model once
model = load_model()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/signal", response_class=HTMLResponse)
def get_signal(request: Request, threshold: float = 0.55):

    df = load_data(DATA_PATH)
    df = add_features(df)

    latest = df.iloc[-1:]
    X_latest = latest[FEATURE_COLUMNS]

    prob_up = model.predict_proba(X_latest)[0, 1]
    signal = "BUY" if prob_up >= threshold else "NO TRADE"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "probability": round(float(prob_up), 4),
            "signal": signal,
            "threshold": threshold
        }
    )
