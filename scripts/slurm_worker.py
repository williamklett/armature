"""Compute-node supervisor: stop training if the laptop stops renewing its lease."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def lease_valid(directory, now=None):
    try:
        return not (directory / 'cancelled').exists() and (now or time.time()) - (directory / 'lease').stat().st_mtime < 120
    except FileNotFoundError:
        return False


def main():
    directory = Path(sys.argv[1]).resolve()
    launch = json.loads((directory / 'launch.json').read_text())
    if not lease_valid(directory):
        raise SystemExit('Laptop lease expired before the job started')
    env = dict(os.environ)
    env.pop('OPENAI_API_KEY', None)
    env.update(launch['env'], PYTHONUNBUFFERED='1')
    process = subprocess.Popen(launch['command'], cwd=str(directory), env=env, start_new_session=True)
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while process.poll() is None:
            if stopped or not lease_valid(directory):
                print('Laptop lease expired or cancellation requested', flush=True)
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise SystemExit(143)
            time.sleep(1)
        raise SystemExit(process.returncode)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


if __name__ == '__main__':
    main()
