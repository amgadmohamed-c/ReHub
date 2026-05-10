#!/usr/bin/env python3
"""
ReHub API — usage examples
===========================
Demonstrates:
  1. Register a device
  2. POST sensor readings (simulates 60 readings → triggers analysis twice)
  3. GET analysis history
  4. GET alerts
  5. Acknowledge alerts
  6. WebSocket real-time listener (runs concurrently)

Requirements:
    pip install httpx websockets

Run:
    python scripts/api_example.py
"""

import asyncio
import json
import math
import random
from datetime import datetime

import httpx

BASE_URL = "http://localhost:8000"
API = f"{BASE_URL}/api/v1"
DEVICE_ID = "arduino-demo-001"


def make_payload(exercise: str = "walking") -> dict:
    """Generate a realistic sensor payload for a given exercise type."""
    profiles = {
        "walking":  dict(ax=(0.3,1.5), ay=(-1.5,1.0), az=(9.5,1.5), gx=(0,.5), gy=(0,.4), gz=(0,.3), emg=90,  flex=400, temp=36.8),
        "running":  dict(ax=(0.5,3.0), ay=(-2.0,2.5), az=(9.8,3.5), gx=(0,1.5), gy=(0,1.2), gz=(0,.8), emg=180, flex=600, temp=37.4),
        "resting":  dict(ax=(0,.1),   ay=(0,.1),    az=(9.81,.2), gx=(0,.05), gy=(0,.05), gz=(0,.05), emg=40,  flex=200, temp=36.5),
    }
    p = profiles.get(exercise, profiles["resting"])

    def r(mu_sigma): return round(random.gauss(*mu_sigma), 6)

    ax = r(p["ax"]); ay = r(p["ay"]); az = r(p["az"])
    return {
        "emg": round(p["emg"] + random.gauss(0, 10), 2),
        "flex": round(p["flex"] + random.gauss(0, 30), 2),
        "flow": round(random.gauss(60, 10), 2),
        "ax": ax, "ay": ay, "az": az,
        "gx": r(p["gx"]), "gy": r(p["gy"]), "gz": r(p["gz"]),
        "temp": round(p["temp"] + random.gauss(0, 0.1), 5),
    }


async def ws_listener():
    """Listen for real-time analysis + alerts via WebSocket."""
    import websockets
    uri = f"ws://localhost:8000/ws/{DEVICE_ID}"
    print(f"\n[WS] Connecting to {uri}")
    try:
        async with websockets.connect(uri) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "analysis":
                    a = msg["analysis"]
                    print(
                        f"\n[WS] ✅ Analysis — exercise={a['exercise_type']} "
                        f"conf={a['exercise_confidence']:.0%} "
                        f"fatigue={a['fatigue_score']:.2f} "
                        f"injury_risk={a['injury_risk']:.2f}"
                    )
                    for alert in msg.get("alerts", []):
                        print(f"       ⚠️  ALERT [{alert['severity']}] {alert['message']}")
                elif event == "heartbeat":
                    print("[WS] ♥ heartbeat")
                elif event == "connected":
                    print(f"[WS] Connected: {msg['message']}")
    except Exception as e:
        print(f"[WS] Error: {e}")


async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:

        # 1. Health check
        r = await client.get("/health")
        print("Health:", r.json())

        # 2. Register device
        r = await client.post(f"{API}/devices", json={
            "device_id": DEVICE_ID,
            "name": "Test Wearable",
        })
        print("\nDevice registered:", r.json())

        # 3. Start WS listener in the background
        ws_task = asyncio.create_task(ws_listener())
        await asyncio.sleep(1)  # let WS connect

        # 4. Send 60 readings (window=50 → 2 analyses triggered)
        print(f"\nSending 60 sensor readings for device '{DEVICE_ID}' ...")
        exercises = (
            ["resting"] * 5
            + ["walking"] * 25
            + ["running"] * 20
            + ["resting"] * 10
        )
        for i, ex in enumerate(exercises):
            payload = make_payload(ex)
            r = await client.post(
                f"{API}/sensors/{DEVICE_ID}/readings",
                json=payload,
            )
            resp = r.json()
            print(
                f"  [{i+1:02d}] intensity={resp['movement_intensity']:.2f} "
                f"buffer={resp['buffer_size']} "
                f"analysis={'YES' if resp['analysis_triggered'] else '---'}"
            )
            await asyncio.sleep(0.05)  # 50 ms between readings

        # 5. Get analysis history
        r = await client.get(f"{API}/analysis/{DEVICE_ID}/history?limit=5")
        analyses = r.json()
        print(f"\nAnalysis history ({len(analyses)} records):")
        for a in analyses:
            print(
                f"  id={a['id']} exercise={a['exercise_type']} "
                f"conf={a['exercise_confidence']:.0%} "
                f"fatigue={a['fatigue_score']:.2f}"
            )

        # 6. Get alerts
        r = await client.get(f"{API}/alerts/{DEVICE_ID}?unacknowledged_only=true")
        alerts = r.json()
        print(f"\nUnacknowledged alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  id={a['id']} [{a['severity']}] {a['alert_type']}: {a['message'][:60]}")

        # 7. Acknowledge all
        if alerts:
            ids = [a["id"] for a in alerts]
            r = await client.post(
                f"{API}/alerts/{DEVICE_ID}/acknowledge",
                json={"alert_ids": ids},
            )
            print("\nAcknowledge response:", r.json())

        await asyncio.sleep(2)
        ws_task.cancel()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
