import json
import threading
from typing import Dict
import paho.mqtt.client as mqtt
from app.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

# shared sensor state (in-memory)
SENSOR_STATE: Dict[str, float] = {}

_client = None


def _on_connect(client, userdata, flags, rc):
    print("MQTT connected, subscribing to:", MQTT_TOPIC)
    client.subscribe(MQTT_TOPIC)


def _on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        # expected keys: displacement, vibration, pore_pressure, timestamp etc.
        SENSOR_STATE.update(data)
        print("Received sensor data:", data)  # 👈 Added logging for debugging
    except Exception as e:
        print("MQTT message parse error:", e)


def start_mqtt_client():
    global _client
    if _client is not None:
        return _client
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    # run loop in background thread
    t = threading.Thread(target=client.loop_forever, daemon=True)
    t.start()
    _client = client
    return client


def get_latest_sensor_state():
    return dict(SENSOR_STATE)


if __name__ == "__main__":
    print("Starting MQTT Subscriber...")
    # Ensure client starts when running directly
    start_mqtt_client()

    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Subscriber stopped by user.")

