"""Run only operator-configured commands; requests never contain shell commands."""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .core import RECIPES, number, uid

ROOT = Path(__file__).resolve().parents[1]


class Jobs:
    def __init__(self, store, state_dir, config):
        self.store, self.state_dir = store, Path(state_dir)
        self.runners = config.get("runners", {})
        self.processes = {}
        self.watchers = []
        self.lock = threading.RLock()
        for run in store.all("runs"):
            if run["status"] in ("running", "starting", "stopping"):
                run.update(status="interrupted", ended_at=time.time())
                run["logs"].append("Bridge restarted; the previous process state is unknown. Check external jobs before restarting.")
                store.put("runs", run)

    def availability(self):
        result = {}
        for key in RECIPES:
            runner = self.runners.get(key)
            if key == "toy_q":
                result[key] = {"ready": True, "reason": "Built-in CPU pipeline check"}
            elif key == "agent_session":
                result[key] = {"ready": False, "reason": "Use the Control room to start a session"}
            elif not runner:
                result[key] = {"ready": False, "reason": "Configure this runner in bridge.local.json"}
            else:
                command = runner.get("command", [])
                executable = bool(command and isinstance(command, list) and isinstance(command[0], str) and shutil.which(command[0]))
                cwd = Path(runner.get("cwd", str(ROOT))).expanduser()
                result[key] = {"ready": executable and cwd.is_dir(), "reason": "Configured runner; model/data readiness checked at launch" if executable and cwd.is_dir() else "Runner executable or working directory is missing"}
        return result

    def launch(self, exp):
        recipe = exp["recipe"]
        if not self.availability()[recipe]["ready"]:
            raise ValueError(self.availability()[recipe]["reason"])
        if recipe.endswith("_sft") and not exp["dataset"]:
            raise ValueError("Provide a LeRobot dataset repository before training")
        if recipe.endswith("_rl") and not exp["checkpoint"]:
            raise ValueError("Provide the matching SFT checkpoint before RL")
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError("A local runner is already active. Finish or stop it before starting another.")
            run_id = uid()
            out = self.state_dir / "runs" / run_id
            out.mkdir(parents=True)
            values = {**exp, "run_id": run_id, "output_dir": str(out / "training"), "config_path": str(out / "experiment.json")}
            if recipe == "toy_q":
                command = [sys.executable, "-u", str(ROOT / "scripts/toy_train.py"), "--seed", str(exp["seed"]), "--episodes", str(min(exp["steps"], 500))]
                cwd = ROOT
            else:
                runner = self.runners[recipe]
                command = []
                for arg in runner["command"]:
                    # Values remain single argv elements. No shell interpolation.
                    command.append(re.sub(r"\{([a-z_]+)\}", lambda m: str(values[m[1]]) if m[1] in values else m[0], arg))
                cwd = Path(runner.get("cwd", str(ROOT))).expanduser()
            run = {"id": run_id, "experiment_id": exp["id"], "created_at": time.time(), "ended_at": None, "status": "starting", "config": exp, "command": command, "logs": [], "metrics": [], "exit_code": None}
            (out / "experiment.json").write_text(json.dumps(exp, indent=2))
            self.store.put("runs", run)
            try:
                env = dict(os.environ)
                # Training subprocesses do not need the grader credential or bridge token.
                env.pop("OPENAI_API_KEY", None)
                env["PYTHONUNBUFFERED"] = "1"
                p = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1, start_new_session=True)
            except OSError as e:
                run.update(status="failed", ended_at=time.time(), logs=[str(e)])
                return self.store.put("runs", run)
            self.processes[run_id] = p
            self.store.change("runs", run_id, lambda r: r.update(status="running"))
            watcher = threading.Thread(target=self._watch, args=(run_id, p, out), daemon=True)
            self.watchers.append(watcher)
            watcher.start()
            return self.store.get("runs", run_id)

    def _watch(self, run_id, process, out):
        with (out / "stdout.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                line = line.rstrip()[:2000]
                metric = None
                if line.startswith("ARMATURE_METRIC "):
                    try:
                        candidate = json.loads(line[len("ARMATURE_METRIC "):])
                        metric = {k: number(v, -1e15, 1e15, k) for k, v in candidate.items() if k in ("step", "reward", "success_rate", "loss")}
                        if "step" not in metric:
                            metric = None
                    except (ValueError, TypeError, AttributeError):
                        metric = None
                def append(r):
                    r["logs"] = (r["logs"] + [line])[-600:]
                    if metric:
                        r["metrics"] = (r["metrics"] + [metric])[-10000:]
                self.store.change("runs", run_id, append)
        code = process.wait()
        process.stdout.close()
        def finish(r):
            r.update(exit_code=code, ended_at=time.time(), status="cancelled" if r["status"] == "stopping" else ("completed" if code == 0 else "failed"))
        self.store.change("runs", run_id, finish)

    def stop(self, run_id):
        with self.lock:
            run = self.store.get("runs", run_id)
            p = self.processes.get(run_id)
            if not p or p.poll() is not None:
                raise ValueError("This run has no active local process")
            self.store.change("runs", run_id, lambda r: r.update(status="stopping"))
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            def kill_later():
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            threading.Thread(target=kill_later, daemon=True).start()
            return self.store.get("runs", run_id)

    def close(self):
        for run_id, process in list(self.processes.items()):
            if process.poll() is None:
                try:
                    self.stop(run_id)
                except ValueError:
                    pass
        for watcher in self.watchers:
            watcher.join(timeout=7)
