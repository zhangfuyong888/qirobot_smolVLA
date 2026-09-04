"""HTTPS/WSS server that publishes the latest coherent Quest controller frame."""

from __future__ import annotations

import asyncio
import json
import ssl
import threading
from pathlib import Path

from aiohttp import WSMsgType, web

from .protocol import LatestFrameStore, PROTOCOL_VERSION, parse_controller_frame


class QuestWebServer:
    def __init__(
        self,
        store: LatestFrameStore,
        host: str,
        port: int,
        cert_path: Path | None,
        key_path: Path | None,
        web_root: Path,
    ) -> None:
        self.store = store
        self.host = host
        self.port = int(port)
        self.cert_path = cert_path
        self.key_path = key_path
        self.web_root = Path(web_root)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._clients: set[web.WebSocketResponse] = set()

    @property
    def secure(self) -> bool:
        return self.cert_path is not None and self.key_path is not None

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.secure:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(self.cert_path), str(self.key_path))
        return context

    async def _index(self, _request: web.Request) -> web.StreamResponse:
        response = web.FileResponse(self.web_root / "index.html")
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "protocol_version": PROTOCOL_VERSION, **self.store.stats()})

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        if not self.store.client_connected(max_clients=1):
            raise web.HTTPConflict(text="A Quest controller client is already connected.")
        websocket = web.WebSocketResponse(heartbeat=10.0, max_msg_size=64 * 1024, compress=False)
        try:
            await websocket.prepare(request)
            self._clients.add(websocket)
            await websocket.send_json({"type": "server_hello", "version": PROTOCOL_VERSION})
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                        if isinstance(payload, dict) and payload.get("type") == "client_log":
                            level = str(payload.get("level", "info"))[:20]
                            text = str(payload.get("message", ""))[:300]
                            print(f"[TELEOP][WEBXR][{level}] {text}", flush=True)
                            continue
                        frame = parse_controller_frame(message.data)
                        self.store.publish(frame)
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        print(f"[TELEOP][WEBXR][error] rejected message: {str(exc)[:300]}", flush=True)
                        await websocket.send_json({"type": "error", "message": str(exc)[:300]})
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(websocket)
            self.store.client_disconnected()
        return websocket

    def send_status(self, payload: dict) -> None:
        """Push a small HUD/status JSON to the connected Quest page (best-effort)."""
        loop = self._loop
        if loop is None or not loop.is_running() or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), loop)

    async def _broadcast(self, payload: dict) -> None:
        dead: list[web.WebSocketResponse] = []
        for websocket in list(self._clients):
            if websocket.closed:
                dead.append(websocket)
                continue
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    async def _start_async(self) -> None:
        app = web.Application(client_max_size=64 * 1024)
        app.router.add_get("/", self._index)
        app.router.add_get("/health", self._health)
        app.router.add_get("/ws", self._websocket)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port, ssl_context=self._ssl_context())
        await site.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_async())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        if self._runner is not None:
            self._loop.run_until_complete(self._runner.cleanup())
        self._loop.close()

    def start(self, timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="quest-webxr-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("Quest WebXR server did not start in time")
        if self._startup_error is not None:
            raise RuntimeError(f"Quest WebXR server failed: {self._startup_error}") from self._startup_error

    def close(self) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
