from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== Load Trained Model ==========
# Make sure you have saved your trained pipeline earlier using:
# joblib.dump(xgb_pipeline, "xgb_pipeline.pkl")

try:
    model = joblib.load("xgb_exoplanet_pipeline.joblib")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error("❌ Error loading model: %s", e)
    model = None

# ========== FastAPI App ==========
app = FastAPI(title="NASA Exoplanet Predictor", version="1.0.0")

# Configure CORS for frontend. Use environment variable ALLOWED_ORIGINS to override.
# Example: ALLOWED_ORIGINS="http://localhost:3000,https://myfrontend.app" or "*"
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").strip()
if allowed == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [o.strip() for o in allowed.split(",") if o.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Pydantic Model ==========
class PlanetData(BaseModel):
    # Numeric fields
    sy_snum: Optional[float] = 0
    sy_pnum: Optional[float] = 0
    disc_year: Optional[float] = 0
    pl_orbper: Optional[float] = 0
    pl_orbsmax: Optional[float] = 0
    pl_orbeccen: Optional[float] = 0
    pl_rade: Optional[float] = 0
    pl_radj: Optional[float] = 0
    pl_bmasse: Optional[float] = 0
    pl_bmassj: Optional[float] = 0
    pl_insol: Optional[float] = 0
    pl_eqt: Optional[float] = 0
    st_teff: Optional[float] = 0
    st_rad: Optional[float] = 0
    ttv_flag: Optional[float] = 0
    st_mass: Optional[float] = 0
    st_met: Optional[float] = 0
    st_logg: Optional[float] = 0
    ra: Optional[float] = 0
    dec: Optional[float] = 0
    sy_dist: Optional[float] = 0
    sy_vmag: Optional[float] = 0
    sy_kmag: Optional[float] = 0
    sy_gaiamag: Optional[float] = 0
    planet_star_radius_ratio: Optional[float] = 0
    planet_star_mass_ratio: Optional[float] = 0

    # Text fields
    pl_name: Optional[str] = ""
    hostname: Optional[str] = ""
    discoverymethod: Optional[str] = ""
    disc_facility: Optional[str] = ""
    soltype: Optional[str] = ""
    pl_bmassprov: Optional[str] = ""
    st_spectype: Optional[str] = ""
    st_metratio: Optional[str] = ""

# Response model for clarity
class PredictionResponse(BaseModel):
    disposition: str
    prob_not_confirmed: float
    prob_confirmed: float
    message: str

# ========== Routes ==========

@app.get("/")
def home():
    return {"message": "🚀 NASA Exoplanet Prediction API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/predict", response_model=PredictionResponse)
def predict_exoplanet(data: PlanetData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        logger.info("📥 Input received for prediction: %s", input_df.to_dict(orient="records"))

        # Check if model expects specific columns
        logger.info("🧩 Model expects columns: %s", getattr(model, 'feature_names_in_', 'Unknown'))

        # Predict
        prediction = model.predict(input_df)[0]
        logger.info("✅ Raw prediction output: %s", prediction)

        # Try probability prediction
        prob_not_confirmed, prob_confirmed = 0.0, 0.0
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(input_df)[0]
                prob_not_confirmed = round(float(probs[0]) * 100, 2)
                prob_confirmed = round(float(probs[1]) * 100, 2)
                logger.info("📊 Prediction probabilities: %s", probs)
            except Exception as e:
                logger.exception("⚠️ predict_proba failed: %s", e)

        disposition = "CONFIRMED" if int(prediction) == 1 else "NOT CONFIRMED"

        return PredictionResponse(
            disposition=disposition,
            prob_not_confirmed=prob_not_confirmed,
            prob_confirmed=prob_confirmed,
            message=f"Prediction successful for planet {data.pl_name or 'unknown'}"
        )

    except Exception as e:
        logger.exception("❌ Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ========== Run the server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
