import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# load the trained model and preprocessor
BASE_DIR = Path(__file__).resolve().parent
MODEL_BUNDLE_PATH = BASE_DIR / "artifacts" / "tuned_xgb_notebook_bundle.joblib"
model_bundle = joblib.load(MODEL_BUNDLE_PATH)
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]

# instantiate the FastAPI app
app = FastAPI(title="Fraud Detection API")


# define the input data model
class Transaction(BaseModel):
    timestamp: datetime = Field(example="2022-10-03T18:40:59.468549+00:00")
    home_country: str = Field(example="us")
    source_currency: str = Field(example="usd")
    dest_currency: str = Field(example="cad")
    channel: str = Field(example="atm")
    amount_src: float = Field(example=278.19)
    amount_usd: float = Field(example=278.19)
    fee: float = Field(example=4.25)
    exchange_rate_src_to_dest: float = Field(example=1.351351)
    new_device: bool = Field(example=False)
    ip_country: str = Field(example="us")
    location_mismatch: bool = Field(example=False)
    ip_risk_score: float = Field(example=0.123)
    kyc_tier: str = Field(example="standard")
    account_age_days: int = Field(example=263)
    device_trust_score: float = Field(example=0.522)
    chargeback_history_count: int = Field(example=0)
    risk_score_internal: float = Field(example=0.223)
    txn_velocity_1h: int = Field(example=0)
    txn_velocity_24h: int = Field(example=0)
    corridor_risk: float = Field(example=0.0)


def predict_fraud(transaction: Transaction):
    try:
        # Use JSON mode so datetime is serialized consistently with request payload format.
        row = transaction.model_dump(mode="json")

        # Derive the same engineered features EDA.ipynb computes from raw fields —
        # the preprocessor was fit on these, not on a raw timestamp.
        row["transaction_hour"] = transaction.timestamp.hour
        row["log_amount_usd"] = np.log1p(transaction.amount_usd)
        row["log_fee"] = np.log1p(transaction.fee)
        row["log_exchange_rate"] = np.log1p(transaction.exchange_rate_src_to_dest)

        df = pd.DataFrame([row])

        # preprocess the input data
        df_processed = preprocessor.transform(df)

        # make the prediction using the loaded model
        prediction = int(model.predict(df_processed)[0])

        result = {
            "is_fraud": prediction,
            "label": "Fraud" if prediction == 1 else "Legit",
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_processed)[0]
            result["fraud_probability"] = float(probabilities[1])

        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/predict")
def predict(transaction: Transaction):
    return predict_fraud(transaction)