"""Preview or execute a bounded, logged motion through the local bridge.

No serial access, hidden fault resets, or automatic retries of movement requests.
Camera frames and logs stay in the private state directory.
"""
import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--experiment', help='Required to start a stopped session')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--speed', type=float, default=4)
    parser.add_argument('--state-dir', type=Path, default=ROOT / '.armature')
    args = parser.parse_args()
    token = (args.state_dir / 'bridge-token').read_text().strip()

    def request(path, data=None):
        req = urllib.request.Request('http://127.0.0.1:%d/api/%s' % (args.port, path),
            data=None if data is None else json.dumps(data, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from None

    def action(name, **kwargs):
        return request('arm/action', {'action': name, **kwargs})

    state = request('arm/observe')
    if state['fault'] or not state['connected']:
        raise RuntimeError(state['fault'] or 'Connect the arm first')
    if state.get('motion') and state['motion']['state'] == 'running':
        raise RuntimeError('A motion is already running; inspect or stop it first')
    points = json.loads(args.waypoints.read_text())
    if args.execute and state['stopped']:
        if not args.experiment:
            raise RuntimeError('Provide --experiment to start the stopped session')
        state = action('start', experiment_id=args.experiment)
    plan = action('plan', waypoints=points, speed=args.speed)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    directory = args.state_dir / 'motions' / plan['id']
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    def camera(name):
        frame = request('camera/frame')
        if not frame['fresh']:
            raise RuntimeError('Camera frame unavailable')
        (directory / (name + '.jpg')).write_bytes(base64.b64decode(frame.pop('data_url').split(',', 1)[1], validate=True))
        (directory / (name + '.json')).write_text(json.dumps(frame, indent=2) + '\n')

    attempted = False
    try:
        camera('before')
        attempted = True
        state = action('execute', plan_id=plan['id'])
        started, next_frame, observations = time.monotonic(), 0, []
        while state['motion']['state'] == 'running':
            elapsed = time.monotonic() - started
            if elapsed > plan['duration'] + 8:
                raise RuntimeError('Motion client deadline exceeded')
            if elapsed >= next_frame:
                camera('frame-%03d' % len(observations))
                next_frame = elapsed + 1
            observations.append({k: state[k] for k in ('positions', 'motion', 'telemetry', 'stopped', 'fault')})
            time.sleep(.1)
            state = request('arm/observe')
        camera('after')
        (directory / 'observations.json').write_text(json.dumps(observations) + '\n')
        (directory / 'result.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps({'motion': state['motion'], 'positions': state['positions'],
                          'fault': state['fault'], 'log_dir': str(directory.resolve())}, indent=2))
        if state['motion']['state'] != 'completed':
            raise RuntimeError(state['motion']['reason'])
    except BaseException:
        if attempted:
            try:
                action('stop')
            except Exception:
                print('Stop request could not be confirmed; check the arm locally.')
        raise


if __name__ == '__main__':
    main()
