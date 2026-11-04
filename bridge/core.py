"""Persistence, validation, and the agent-control sandbox (stdlib only)."""
import copy
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path

RECIPES = {
    "smolvla_sft": {"name": "SmolVLA · demonstrations", "model": "SmolVLA", "kind": "Imitation learning", "environment": "Recorded dataset", "checkpoint": "lerobot/smolvla_base"},
    "pi05_sft": {"name": "π0.5 · demonstrations", "model": "π0.5", "kind": "Imitation learning", "environment": "Recorded dataset", "checkpoint": "lerobot/pi05_base"},
    "smolvla_rl": {"name": "SmolVLA · experimental GRPO", "model": "SmolVLA", "kind": "RL · simulation", "environment": "LIBERO", "checkpoint": "", "note": "Requires a configured FlowVLA-RL checkout and an SFT checkpoint. This recipe does not train on the physical arm."},
    "pi05_rl": {"name": "π0.5 · PPO", "model": "π0.5", "kind": "RL · simulation", "environment": "LIBERO", "checkpoint": "", "note": "Requires a configured RLinf checkout and matching SFT checkpoint. SO101 online RL needs a task adapter."},
    "agent_session": {"name": "Interactive agent session", "model": "Agent", "kind": "In-context exploration", "environment": "Control room", "checkpoint": ""},
    "toy_q": {"name": "Pipeline check · Q-learning", "model": "Tabular Q-learning", "kind": "RL · toy test", "environment": "1D reach", "checkpoint": "", "note": "A real, small Q-learning job to verify tracking. This is not a VLA or a physics simulator."},
}


def uid():
    return uuid.uuid4().hex


def number(value, low, high, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s must be a finite number" % label)
    if not low <= value <= high or (integer and value != int(value)):
        raise ValueError("%s must be between %s and %s%s" % (label, low, high, " (integer)" if integer else ""))
    return int(value) if integer else value


def string(value, label, limit=4000, required=True):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError("%s is required and must be at most %s characters" % (label, limit))
    if "\x00" in value:
        raise ValueError("Invalid null character")
    return value.strip()


def experiment(data):
    recipe = data.get("recipe")
    if recipe not in RECIPES:
        raise ValueError("Choose a known recipe")
    return {
        "id": uid(), "created_at": time.time(), "recipe": recipe,
        "name": string(data.get("name"), "Experiment name", 120),
        "task": string(data.get("task"), "Task", 2000),
        "owner": string(data.get("owner", ""), "Owner", 80, False),
        "dataset": string(data.get("dataset", ""), "Dataset", 300, False),
        "checkpoint": string(data.get("checkpoint", RECIPES[recipe]["checkpoint"]), "Checkpoint", 500, False),
        "rubric": string(data.get("rubric", "The requested task is visibly complete. The object is stable and released by the gripper. Mark unclear if the frames do not establish completion."), "Rubric", 4000),
        "notes": string(data.get("notes", ""), "Notes", 5000, False),
        "seed": number(data.get("seed", 42), 0, 2147483647, "Seed", True),
        "steps": number(data.get("steps", 20000), 1, 10000000, "Steps", True),
        "batch_size": number(data.get("batch_size", 8), 1, 4096, "Batch size", True),
        "learning_rate": number(data.get("learning_rate", {"toy_q": 0.2, "pi05_sft": 0.000025, "pi05_rl": 0.000005, "smolvla_rl": 0.000001}.get(recipe, 0.0001)), 1e-8, 1, "Learning rate"),
    }


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, value TEXT, PRIMARY KEY(kind,id))")
        self.lock = threading.RLock()

    def put(self, kind, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, value["id"], json.dumps(value, allow_nan=False)))
            self.db.commit()
        return copy.deepcopy(value)

    def get(self, kind, identifier):
        with self.lock:
            row = self.db.execute("SELECT value FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        if not row:
            raise KeyError("Record not found")
        return json.loads(row[0])

    def all(self, kind):
        with self.lock:
            rows = self.db.execute("SELECT value FROM records WHERE kind=?", (kind,)).fetchall()
        return sorted([json.loads(r[0]) for r in rows], key=lambda x: x.get("created_at", 0), reverse=True)

    def change(self, kind, identifier, update):
        with self.lock:
            value = self.get(kind, identifier)
            update(value)
            return self.put(kind, value)


class Sandbox:
    """A 2D reach sandbox, not a physical-arm or physics simulation."""
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.state = {"x": 0.35, "y": 0.6, "gripper": 1, "stopped": True, "revision": 0}
        self.session_id = uid()
        self.observed_at = 0
        self.experiment_id = None
        self.log = []

    def observe(self):
        with self.lock:
            self.observed_at = time.monotonic()
            return {**self.state, "session_id": self.session_id, "mode": "sandbox", "target": {"x": 0.72, "y": 0.3}, "history": self.log[-50:], "experiment_id": self.experiment_id}

    def action(self, data):
        with self.lock:
            kind = data.get("action")
            if kind == "stop":
                self.state["stopped"] = True
            elif kind == "start":
                eid = data.get("experiment_id")
                self.store.get("experiments", eid)
                self.experiment_id = eid
                self.session_id = uid()
                self.log = []
                self.state["stopped"] = False
            elif kind == "move":
                if self.state["stopped"]:
                    raise ValueError("Session is stopped. Start a session first.")
                if data.get("revision") != self.state["revision"] or time.monotonic() - self.observed_at > 5:
                    raise ValueError("Observation is stale. Observe again before moving.")
                dx = number(data.get("dx", 0), -0.08, 0.08, "x step")
                dy = number(data.get("dy", 0), -0.08, 0.08, "y step")
                grip = number(data.get("gripper", self.state["gripper"]), 0, 1, "Gripper")
                x, y = self.state["x"] + dx, self.state["y"] + dy
                if not (0.08 <= x <= 0.92 and 0.08 <= y <= 0.92):
                    raise ValueError("Action exceeds sandbox workspace")
                self.state.update(x=x, y=y, gripper=grip)
            elif kind == "note":
                if not self.experiment_id:
                    raise ValueError("Start a session before adding a note")
            else:
                raise ValueError("Unknown action")
            self.state["revision"] += 1
            entry = {"time": time.time(), "action": kind, "state": dict(self.state), "note": string(data.get("note", ""), "Note", 2000, False)}
            self.log.append(entry)
            if self.experiment_id:
                self.store.put("sessions", {"id": self.session_id, "created_at": self.log[0]["time"], "experiment_id": self.experiment_id, "mode": "sandbox", "history": self.log, "state": dict(self.state)})
            return self.observe()
