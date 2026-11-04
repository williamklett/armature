"""Prepare isolated LeRobot dependencies; no weights, camera, or motors are opened."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

REVISION = '9e5ee9859480e425a97a6836bb3fee479539beb5'
ROOT = Path(__file__).resolve().parents[1]


def disk_bytes(directory):
    result = subprocess.run(['du', '-sk', str(directory)], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, 'LC_ALL': 'C'})
    # uv atomically replaces temporary directories while du walks the cache.
    # A vanished file can be ignored; unreadable paths must fail closed.
    errors = result.stderr.splitlines()
    if result.returncode and (not errors or any('No such file or directory' not in line for line in errors)):
        raise RuntimeError('Cannot measure project storage: ' + result.stderr.strip())
    return int(result.stdout.split()[0]) * 1024


def bounded_install(command, budget_root, max_gib):
    """Guard the project allocation, including downloads, cache, and environments."""
    limit = max_gib * 1024**3
    if disk_bytes(budget_root) >= limit:
        raise SystemExit('Project storage limit reached before installation')
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while process.poll() is None:
            if disk_bytes(budget_root) >= limit:
                raise SystemExit('Project storage limit reached; installation stopped. Keep additional quota headroom for in-flight writes.')
            time.sleep(1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['smolvla', 'pi05', 'robot'])
    p.add_argument('--directory', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--budget-root', type=Path)
    p.add_argument('--max-disk-gib', type=float)
    args = p.parse_args()
    if bool(args.budget_root) != bool(args.max_disk_gib) or (args.max_disk_gib is not None and args.max_disk_gib <= 0):
        p.error('Provide both --budget-root and a positive --max-disk-gib')
    uv = shutil.which('uv')
    if not uv:
        raise SystemExit('Install uv from https://docs.astral.sh/uv/getting-started/installation/')
    env = (args.directory or ROOT / ('.venv-' + args.model)).resolve()
    extras = {'smolvla': 'training,smolvla', 'pi05': 'training,pi', 'robot': 'core_scripts,feetech'}[args.model]
    spec = 'lerobot[%s] @ git+https://github.com/huggingface/lerobot.git@%s' % (extras, REVISION)
    commands = [[uv, 'venv', '--python', '3.12', str(env)], [uv, 'pip', 'install', '--python', str(env / 'bin/python'), spec]]
    if env.exists():
        commands = commands[1:]
    for cmd in commands:
        print(json.dumps(cmd), flush=True)
        if not args.dry_run:
            if args.budget_root:
                bounded_install(cmd, args.budget_root.resolve(), args.max_disk_gib)
            else:
                subprocess.run(cmd, check=True)
    if not args.dry_run:
        print('Ready: ' + str(env / 'bin/python'))
        print('Checkpoints download on first training launch. Dataset and GPU readiness remain separate checks.')


if __name__ == '__main__':
    main()
