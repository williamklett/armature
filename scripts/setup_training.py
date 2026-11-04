"""Prepare isolated LeRobot dependencies; no weights, camera, or motors are opened."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

REVISION = '9e5ee9859480e425a97a6836bb3fee479539beb5'
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model', choices=['smolvla', 'pi05', 'robot'])
    p.add_argument('--directory', type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
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
            subprocess.run(cmd, check=True)
    if not args.dry_run:
        print('Ready: ' + str(env / 'bin/python'))
        print('Checkpoints download on first training launch. Dataset and GPU readiness remain separate checks.')


if __name__ == '__main__':
    main()
