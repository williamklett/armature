"""Laptop runner: stage a run, submit Slurm, stream logs, and cancel on Stop."""
import argparse
import json
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def prepare(profile, experiment, run_id, smoke=False):
    directory = str(Path(profile['root']) / 'runs' / run_id)
    model = profile.get('models', {}).get(experiment['recipe'], {})
    python = model.get('python', '/usr/bin/python3')
    if smoke:
        command = ['/usr/bin/python3', directory + '/toy_train.py', '--episodes', str(experiment['steps']), '--seed', str(experiment['seed']), '--learning-rate', str(experiment['learning_rate'])]
    else:
        if not model.get('python'):
            raise ValueError('No environment configured for this recipe')
        adapter = 'lerobot_train.py' if experiment['recipe'].endswith('_sft') else 'rl_train.py'
        command = [python, directory + '/' + adapter, '--config', directory + '/experiment.json', '--output', directory + '/training']
        command += ['--device', 'cuda'] if adapter == 'lerobot_train.py' else ['--root', model['backend_root']]
    env = {'HF_HOME': profile['root'] + '/cache/huggingface', 'XDG_CACHE_HOME': profile['root'] + '/cache', 'TMPDIR': directory}
    return {'action': 'submit', 'root': profile['root'], 'run_id': run_id, 'experiment': experiment,
            'resources': profile['smoke_resources'] if smoke else {**profile['resources'], **model.get('resources', {})},
            'launch': {'command': command, 'env': env},
            'files': {name: (SCRIPTS / name).read_text() for name in ['slurm_worker.py', 'lerobot_train.py', 'rl_train.py', 'toy_train.py']}}


class Remote:
    def __init__(self, profile, run_id):
        self.profile, self.run_id = profile, run_id

    def call(self, action, **values):
        payload = {'root': self.profile['root'], 'run_id': self.run_id, 'action': action, **values}
        # SSH invokes a remote shell; quote the fixed helper source and keep all
        # experiment data out of that shell in JSON stdin.
        cmd = self.profile['ssh'] + ['python3 -c ' + shlex.quote((SCRIPTS / 'slurm_remote.py').read_text())]
        result = subprocess.run(cmd, input=json.dumps(payload), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25, start_new_session=True)
        try:
            data = json.loads(result.stdout)
        except ValueError:
            raise RuntimeError('SSH failed: ' + result.stderr.strip()[-400:])
        if result.returncode or data.get('error'):
            raise RuntimeError(data.get('error', 'Remote command failed'))
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[a-f0-9]{32}', args.run_id):
        raise SystemExit('Invalid run ID')
    profile = json.loads(Path(args.profile).read_text())
    experiment = json.loads(Path(args.config).read_text())
    payload = prepare(profile, experiment, args.run_id, args.smoke)
    remote = Remote(profile, args.run_id)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    destination = Path(args.output).parent / 'remote.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset, pending, last_state, finished = 0, '', None, False
    terminal_seen = False
    unknown_since = None
    submitted = False
    try:
        submitted = True  # Cancellation also covers a lost submission response.
        job = remote.call(**payload)
        destination.write_text(json.dumps(job, indent=2))
        while not stopped.is_set():
            data = remote.call('status', offset=offset)
            offset = data['offset']
            pending += data['log']
            lines = pending.split('\n')
            pending = lines.pop()
            for line in lines:
                print(line, flush=True)
            state = data['state']
            if state != last_state:
                info = {key: data[key] for key in ['job_id', 'directory', 'state', 'exit_code']}
                destination.write_text(json.dumps(info, indent=2))
                print('ARMATURE_REMOTE ' + json.dumps(info), flush=True)
                last_state = state
            if state in TERMINAL and not data['more'] and terminal_seen:
                if pending:
                    print(pending, flush=True)
                finished = True
                raise SystemExit(0 if state == 'COMPLETED' and data['exit_code'] == '0:0' else 1)
            terminal_seen = state in TERMINAL
            if state == 'UNKNOWN':
                # Slurm accounting may lag. Keep the lease renewed while waiting.
                unknown_since = unknown_since or time.monotonic()
                if time.monotonic() - unknown_since > 60:
                    raise RuntimeError('Slurm accounting did not return a final state')
                print('Waiting for Slurm accounting', flush=True)
            else:
                unknown_since = None
            if not data['more']:
                stopped.wait(5)
        raise SystemExit(143)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), flush=True)
        raise SystemExit(1)
    finally:
        if submitted and not finished:
            try:
                remote.call('cancel')
                print('Slurm cancellation requested', flush=True)
            except Exception as error:
                print('Cancellation could not be confirmed: %s. The compute-node lease expires after 120 seconds; check Slurm before restarting.' % error, flush=True)


if __name__ == '__main__':
    main()
