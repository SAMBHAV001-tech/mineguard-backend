# app/main.py
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import os

from app.services.weather import fetch_nasa_power, fetch_openweather
from app.services.dem import get_elevation_at   # only elevation now
from app.services.sensors_mqtt import start_mqtt_client, get_latest_sensor_state
from app.models.predictor import predict as risk_score
from app.services.mine_services import get_mine_data
from app.services.predefined_slopes import get_slope   # predefined slope lookup

# ------------ APP ------------ #
app = FastAPI(
    title="MineGuard AI",
    description="Smart Rockfall Prediction & Safety System",
    version="1.0"
)

# Optional: if you ship a swagger.yaml, use it
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    yaml_path = os.path.join(os.path.dirname(__file__), "swagger.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f)
        app.openapi_schema = spec
        return app.openapi_schema
    else:
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

app.openapi = custom_openapi

# CORS: allow the frontend to call this API. For production lock to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use a narrow list in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ ROUTES ------------ #
@app.get("/")
async def root():
    return {"message": "MineGuard AI Backend running 🚀"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/weather/{lat}/{lon}")
async def weather(lat: float, lon: float):
    """Fetch combined weather data and return normalized fields so frontend keys are stable."""
    nasa_data = fetch_nasa_power(lat, lon)
    owm_data = fetch_openweather(lat, lon)

    # Normalize OpenWeather output into stable keys (fallback to NASA POWER or None)
    temperature = None
    humidity = None
    rainfall = None
    wind_speed = None

    if owm_data:
        temperature = owm_data.get("temp")
        humidity = owm_data.get("humidity")
        rainfall = owm_data.get("rain_1h", 0)  # default 0
        wind_speed = owm_data.get("wind_speed")
    else:
        # try to get some approximate data from NASA POWER if present
        try:
            # naive fallback: take last available T2M / PRECTOTCORR / WS10M
            t2m = nasa_data["properties"]["parameter"].get("T2M", {})
            prect = nasa_data["properties"]["parameter"].get("PRECTOTCORR", {})
            ws = nasa_data["properties"]["parameter"].get("WS10M", {})
            # pick most recent key
            if t2m:
                temperature = list(t2m.values())[-1]
            if prect:
                rainfall = list(prect.values())[-1]
            if ws:
                wind_speed = list(ws.values())[-1]
        except Exception:
            pass

    # Ensure numeric types (frontend expects numbers)
    def as_number(v):
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    temperature = as_number(temperature)
    humidity = as_number(humidity)
    rainfall = as_number(rainfall)
    wind_speed = as_number(wind_speed)

    # Return both raw provider objects + normalized summary for frontend convenience
    return {
        "nasa_power": nasa_data,
        "openweather": owm_data,
        "summary": {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "wind_speed": wind_speed
        }
    }

@app.get("/srtm/{lat}/{lon}")
async def elevation(lat: float, lon: float):
    """
    Return elevation and slope for the given latitude and longitude.
    Elevation from DEM/API, slope from predefined lookup table.
    """
    elev = get_elevation_at(lat, lon)
    slope = get_slope(lat, lon)   # use predefined slope lookup
    return {
        "lat": lat,
        "lon": lon,
        "elevation_m": elev,
        "slope_deg": slope
    }

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting MQTT client from FastAPI...")
    start_mqtt_client()

@app.get("/sensors/vibration")
async def vibration():
    """Fetch latest vibration data from sensor state"""
    state = get_latest_sensor_state()
    vib = state.get("vibration")
    # ensure numeric response (frontend expects a number)
    try:
        vib_n = float(vib) if vib is not None else 0.0
    except Exception:
        vib_n = 0.0
    return {"vibration": vib_n}

@app.post("/predict")
async def predict(request: Request):
    """Run AI prediction for rockfall risk"""
    body = await request.json()

    lat = body.get("lat")
    lon = body.get("lon")

    # Demo override for Jharia coalfield to force a HIGH risk example
    if (lat, lon) == (23.0, 86.5):
        features = {
            "slope": 50,
            "rainfall": 70,
            "temperature": 35,
            "vibration": 3.5,
            "wind_speed": 10.0,
            "displacement_rate": 0.5
        }
    else:
        features = {
            "slope": body.get("slope", 30),
            "rainfall": body.get("rainfall", 10),
            "temperature": body.get("temperature", 25),
            "vibration": body.get("vibration", 0.1),
            "wind_speed": body.get("wind_speed", 2.0),
            "displacement_rate": body.get("displacement_rate", body.get("slope", 30) * 0.01)
        }

    result = risk_score(features)

    alert = None
    # predictor may return different key names; normalize them (backwards-compatible)
    risk_level = result.get("risk") or result.get("risk_level") or result.get("risk_level".lower()) or result.get("risk_level".upper())
    risk_level = (risk_level or result.get("risk_level") or result.get("Risk") or "")
    if isinstance(result.get("risk"), str):
        rl = result["risk"].lower()
    else:
        rl = str(risk_level).lower()

    if "high" in rl:
        alert = "⚠️ HIGH RISK of Rockfall detected! Immediate action required."

    return {"features": features, "prediction": result, "alert": alert}

@app.get("/mine/{name}")
async def mine(name: str):
    data = get_mine_data(name)
    if data is None:
        return {"error": f"No mine named '{name}' found"}
    return data

# ------------ START SERVER (optional local run) ------------ #
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
