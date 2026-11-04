"""Local authenticated API + dashboard. Run: python3 -m bridge.server"""
import argparse
import functools
import json
import mimetypes
import os
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .core import RECIPES, Sandbox, Store, experiment, string, uid
from .grader import grade, validate_frames
from .jobs import Jobs
from .robot import PhysicalArm
from .vision import CameraFeed

ROOT = Path(__file__).resolve().parents[1]


class App:
    def __init__(self, state_dir, config=None):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        token_file = self.state_dir / "bridge-token"
        if not token_file.exists():
            fd = os.open(str(token_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(32))
        self.token = token_file.read_text().strip()
        self.store = Store(self.state_dir / "armature.sqlite3")
        self.jobs = Jobs(self.store, self.state_dir, self.config)
        self.robot = Sandbox(self.store)
        self.camera = CameraFeed()
        self.arm = PhysicalArm(self.store, self.config.get("robot"), camera_ready=self.camera.ready)
        self.grading = set()
        self.grade_lock = threading.Lock()

    def route(self, method, path, data):
        bits = path.strip("/").split("/")
        if method == "GET" and path == "/api/camera/status":
            return self.camera.status()
        if method == "GET" and path == "/api/camera/frame":
            return self.camera.frame()
        if method == "POST" and path == "/api/camera/start":
            return self.camera.start()
        if method == "POST" and path == "/api/camera/frame":
            return self.camera.put(data.get("lease"), data.get("frame"))
        if method == "POST" and path == "/api/camera/stop":
            return self.camera.stop(data.get("lease"))
        if method == "GET" and path == "/api/status":
            return {"version": "0.1.0", "storage": "Laptop SQLite", "robot": self.arm.status(), "physical_robot_connected": self.arm.status()["connected"], "grader_ready": bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("ARMATURE_GRADER_MODEL")), "grader_model": os.environ.get("ARMATURE_GRADER_MODEL", ""), "recipes": RECIPES, "runners": self.jobs.availability()}
        if method == "GET" and path == "/api/library":
            return {kind: self.store.all(kind) for kind in ("experiments", "runs", "episodes", "sessions")}
        if method == "POST" and path == "/api/experiments":
            return self.store.put("experiments", experiment(data))
        if method == "POST" and len(bits) == 4 and bits[1] == "experiments" and bits[3] == "runs":
            return self.jobs.launch(self.store.get("experiments", bits[2]))
        if method == "POST" and len(bits) == 4 and bits[1] == "runs" and bits[3] == "stop":
            return self.jobs.stop(bits[2])
        if method == "GET" and len(bits) == 3 and bits[1] in ("runs", "experiments", "episodes", "sessions"):
            return self.store.get(bits[1], bits[2])
        if method == "POST" and path == "/api/episodes":
            self.store.get("experiments", data.get("experiment_id"))
            value = {"id": uid(), "created_at": time.time(), "experiment_id": data["experiment_id"], "frames": validate_frames(data.get("frames")), "notes": string(data.get("notes", ""), "Notes", 5000, False), "human_success": None, "grades": [], "source": "camera"}
            return self.store.put("episodes", value)
        if method == "POST" and len(bits) == 4 and bits[1] == "episodes":
            eid, action = bits[2], bits[3]
            ep = self.store.get("episodes", eid)
            if action == "review":
                val = data.get("human_success")
                if val is not None and type(val) is not bool:
                    raise ValueError("Human success must be true, false, or null")
                return self.store.change("episodes", eid, lambda e: e.update(human_success=val))
            if action == "grade":
                with self.grade_lock:
                    if eid in self.grading:
                        raise ValueError("This episode is already being graded")
                    self.grading.add(eid)
                try:
                    result = grade(ep, self.store.get("experiments", ep["experiment_id"]))
                    return self.store.change("episodes", eid, lambda e: e["grades"].append(result))
                finally:
                    with self.grade_lock:
                        self.grading.remove(eid)
        if method == "GET" and path == "/api/robot/observe":
            return self.robot.observe()
        if method == "POST" and path == "/api/robot/action":
            return self.robot.action(data)
        if method == "GET" and path == "/api/arm/observe":
            return self.arm.observe()
        if method == "POST" and path == "/api/arm/action":
            return self.arm.action(data)
        raise KeyError("Route not found")


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, app, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, *_):
        pass  # No tokens, frames, or request bodies in server logs.

    def _origin_allowed(self):
        origin = self.headers.get("Origin")
        return not origin or origin == "http://" + self.headers.get("Host", "") or origin in self.app.config.get("allowed_origins", [])

    def _send(self, status, body, content_type="application/json", cors=True):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if cors and self.headers.get("Origin") and self._origin_allowed():
            self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if not self._origin_allowed():
            return self._send(403, {"error": "Origin is not allowed"})
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", ""))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        path = urlparse(self.path).path
        # The bind address is loopback. Host validation also blocks DNS rebinding.
        if self.headers.get("Host", "").split(":")[0] not in ("localhost", "127.0.0.1"):
            return self._send(403, {"error": "Use the loopback address or an SSH tunnel"})
        if path == "/api/pair" and method == "GET":
            # Pairing is readable only by this locally served dashboard, never via CORS.
            same = self.headers.get("Sec-Fetch-Site") == "same-origin"
            if not same or self.headers.get("Origin") not in (None, "http://" + self.headers.get("Host", "")):
                return self._send(403, {"error": "Pair from the local dashboard"}, cors=False)
            return self._send(200, {"token": self.app.token}, cors=False)
        if path.startswith("/api/"):
            if not self._origin_allowed():
                return self._send(403, {"error": "Add this exact site origin to allowed_origins in bridge.local.json"})
            auth = self.headers.get("Authorization", "")
            if not secrets.compare_digest(auth, "Bearer " + self.app.token):
                return self._send(401, {"error": "Connect with the bridge pairing token"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > 13000000:
                    return self._send(413, {"error": "Request is too large"})
                if method == "POST" and not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self._send(415, {"error": "Send application/json"})
                data = json.loads(self.rfile.read(length) or b"{}", parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON"))) if method == "POST" else {}
                if not isinstance(data, dict):
                    raise ValueError("Request must be an object")
                return self._send(200, self.app.route(method, path, data))
            except KeyError:
                return self._send(404, {"error": "Record or route not found"})
            except (ValueError, TypeError) as error:
                return self._send(400, {"error": str(error)})
            except Exception:
                return self._send(500, {"error": "Bridge error. No successful result was recorded."})
        if method != "GET":
            return self._send(404, {"error": "Not found"})
        file = ROOT / "web" / ("index.html" if path == "/" else path.lstrip("/"))
        if not file.resolve().is_relative_to((ROOT / "web").resolve()) or not file.is_file():
            return self._send(404, {"error": "Not found"})
        return self._send(200, file.read_bytes(), mimetypes.guess_type(str(file))[0] or "application/octet-stream")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--state-dir", default=str(ROOT / ".armature"))
    p.add_argument("--config", default=str(ROOT / "bridge.local.json"))
    p.add_argument("--preload-robot", action="store_true", help="Load the optional motor SDK before serving camera requests; does not connect motors")
    args = p.parse_args()
    if args.preload_robot:
        from lerobot.robots.so_follower.so_follower import SO101Follower  # noqa: F401
    config = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    app = App(args.state_dir, config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, app=app))
    print("Armature: http://127.0.0.1:%d — sandbox control; no physical arm connected" % args.port, flush=True)
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.jobs.close()
        app.arm.close()
        server.server_close()


if __name__ == "__main__":
    main()
