"""JSON RPC helper sent over authenticated SSH. Inputs come from a trusted profile."""
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


def call(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, **kwargs).stdout.strip()


def run(request):
    root = Path(request['root'])
    rid = request['run_id']
    if not root.is_absolute() or not re.fullmatch('[a-f0-9]{32}', rid):
        raise ValueError('Invalid remote root or run ID')
    directory = root / 'runs' / rid
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    metadata = directory / 'job.json'
    name = 'armature-' + rid
    # Serialize submit/cancel and never submit twice after an ambiguous failure.
    with (directory / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = json.loads(metadata.read_text()) if metadata.exists() else None
        action = request['action']
        if action == 'submit':
            if (directory / 'cancelled').exists():
                raise ValueError('Run has been cancelled')
            if not job:
                if (directory / 'submitting').exists():
                    raise ValueError('Submission state is uncertain; inspect Slurm job ' + name)
                for filename, content in request['files'].items():
                    if filename not in {'slurm_worker.py', 'lerobot_train.py', 'rl_train.py', 'toy_train.py'}:
                        raise ValueError('Unexpected source filename')
                    (directory / filename).write_text(content)
                (directory / 'experiment.json').write_text(json.dumps(request['experiment'], indent=2))
                (directory / 'launch.json').write_text(json.dumps(request['launch'], indent=2))
                (directory / 'lease').touch()
                command = request['launch']['command']
                if not Path(command[0]).is_file():
                    raise ValueError('Training environment is missing: ' + command[0])
                options = request['resources']
                allowed = {'partition', 'account', 'gres', 'constraint', 'cpus-per-task', 'mem', 'time'}
                if set(options) - allowed:
                    raise ValueError('Unknown Slurm resource option')
                argv = ['sbatch', '--parsable', '--job-name=' + name, '--chdir=' + str(directory), '--output=' + str(directory / 'stdout.log')]
                argv += ['--' + key + '=' + str(value) for key, value in options.items()]
                script = '#!/bin/sh\nexec /usr/bin/python3 ' + shlex.join([str(directory / 'slurm_worker.py'), str(directory)]) + '\n'
                (directory / 'submitting').touch()
                job_id = call(argv, input=script).split(';')[0]
                if not job_id.isdigit():
                    raise ValueError('Unexpected Slurm job ID')
                job = {'job_id': job_id, 'directory': str(directory)}
                metadata.write_text(json.dumps(job))
            return {**job, 'state': 'SUBMITTED'}
        if action == 'cancel':
            (directory / 'cancelled').touch()
            call(['scancel', '--name=' + name])
            return {'job_id': job['job_id'] if job else None, 'state': 'CANCEL_REQUESTED'}
        if action != 'status' or not job:
            raise ValueError('Unknown job')
        (directory / 'lease').touch()
        output = directory / 'stdout.log'
        offset = max(0, int(request.get('offset', 0)))
        content = b''
        if output.exists():
            with output.open('rb') as stream:
                stream.seek(offset)
                content = stream.read(128 * 1024)
        state = call(['squeue', '--noheader', '--jobs=' + job['job_id'], '--format=%T']).splitlines()
        exit_code = None
        if not state:
            rows = call(['sacct', '-X', '-n', '-P', '-j', job['job_id'], '--format=JobIDRaw,State,ExitCode']).splitlines()
            match = next((row.split('|') for row in rows if row.split('|')[0] == job['job_id']), None)
            state = [match[1].split()[0].rstrip('+')] if match else ['UNKNOWN']
            exit_code = match[2] if match else None
        return {**job, 'state': state[0], 'exit_code': exit_code, 'offset': offset + len(content), 'log': content.decode('utf-8', errors='replace'), 'more': len(content) == 128 * 1024}


if __name__ == '__main__':
    try:
        print(json.dumps(run(json.load(sys.stdin))))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}))
        sys.exit(1)
