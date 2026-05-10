"""
WebSocket connection manager.

Maintains active connections per device_id so that analysis results
and alerts can be pushed in real-time to the connected mobile app.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Dict, List

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from app.core.logging import get_logger

logger = get_logger(__name__)


def _default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ConnectionManager:
    def __init__(self) -> None:
        # device_id → list of active WebSocket connections
        self._connections: Dict[str, List[WebSocket]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self, device_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(device_id, []).append(ws)
        logger.info("WS connected: device=%s total=%d",
                    device_id, len(self._connections[device_id]))

    def disconnect(self, device_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(device_id, [])
        if ws in conns:
            conns.remove(ws)
        logger.info("WS disconnected: device=%s remaining=%d",
                    device_id, len(conns))

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    async def send_to_device(self, device_id: str, payload: dict) -> None:
        """Push a JSON message to all sockets connected for a device."""
        conns = self._connections.get(device_id, [])
        if not conns:
            return

        data = json.dumps(payload, default=_default)
        dead: list[WebSocket] = []

        for ws in conns:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(data)
            except Exception as exc:
                logger.warning("WS send failed device=%s: %s", device_id, exc)
                dead.append(ws)

        for ws in dead:
            self.disconnect(device_id, ws)

    async def broadcast(self, payload: dict) -> None:
        """Push to ALL connected devices (e.g. system announcements)."""
        tasks = [
            self.send_to_device(did, payload)
            for did in list(self._connections.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def active_devices(self) -> List[str]:
        return [d for d, conns in self._connections.items() if conns]

    def connection_count(self) -> int:
        return sum(len(c) for c in self._connections.values())


# Singleton
ws_manager = ConnectionManager()
