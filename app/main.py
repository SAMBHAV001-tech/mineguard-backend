from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import uvicorn
import yaml
import os

# Import services
from app.services.weather import fetch_nasa_power, fetch_openweather
from app.services.dem import get_elevation_at   # only elevation now
from app.services.sensors_mqtt import start_mqtt_client, get_latest_sensor_state
from app.models.predictor import predict as risk_score
from app.services.mine_services import get_mine_data
from app.services.predefined_slopes import get_slope   # ✅ NEW IMPORT
from fastapi.middleware.cors import CORSMiddleware


# ------------ APP ------------ #
app = FastAPI(
    title="MineGuard AI",
    description="Smart Rockfall Prediction & Safety System",
    version="1.0"
)


# ------------ CUSTOM OPENAPI LOADER ------------ #
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
        # fallback: generate automatically
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

app.openapi = custom_openapi


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or list of domains ["https://your-frontend.com"]
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
    """Fetch combined weather data from NASA POWER + OpenWeather"""
    nasa_data = fetch_nasa_power(lat, lon)
    owm_data = fetch_openweather(lat, lon)
    return {"nasa_power": nasa_data, "openweather": owm_data}


@app.get("/srtm/{lat}/{lon}")
async def elevation(lat: float, lon: float):
    """
    Return elevation and slope for the given latitude and longitude.
    Elevation from DEM/API, slope from predefined lookup table.
    """
    elev = get_elevation_at(lat, lon)
    slope = get_slope(lat, lon)   # ✅ use predefined slope
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
    return {"vibration": state.get("vibration")}


@app.post("/predict")
async def predict(request: Request):
    """Run AI prediction for rockfall risk"""
    body = await request.json()

    lat = body.get("lat")
    lon = body.get("lon")

    # 🚨 Demo override for Jharia coalfield
    if (lat, lon) == (23.0, 86.5):
        features = {
            "slope": 50,            # very steep artificially
            "rainfall": 70,         # heavy rainfall
            "temperature": 35,      # hot mining conditions
            "vibration": 3.5,       # strong mining activity
            "wind_speed": 10.0,     # strong winds
            "displacement_rate": 0.5
        }
    else:
        # ✅ Normal logic (no changes here)
        features = {
            "slope": body.get("slope", 30),
            "rainfall": body.get("rainfall", 10),
            "temperature": body.get("temperature", 25),
            "vibration": body.get("vibration", 0.1),
            "wind_speed": body.get("wind_speed", 2.0),
            "displacement_rate": body.get("slope", 30) * 0.01
        }

    result = risk_score(features)

    # Simple alert logic
    alert = None
    if result["risk"].lower() == "high":
        alert = "⚠️ HIGH RISK of Rockfall detected! Immediate action required."

    return {"features": features, "prediction": result, "alert": alert}



@app.get("/mine/{name}")
async def mine(name: str):
    """
    Return elevation and slope for a named mine from local DEM.
    """
    data = get_mine_data(name)
    if data is None:
        return {"error": f"No mine named '{name}' found"}
    return data


# ------------ START SERVER ------------ #
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
