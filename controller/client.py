"""Only the authenticated loopback bridge can command hardware."""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BridgeClient:
    def __init__(self, port=8787, state_dir=ROOT / '.armature'):
        self.state_dir = Path(state_dir)
        self.base = 'http://127.0.0.1:%d/api/' % int(port)

    def request(self, path, data=None):
        token = (self.state_dir / 'bridge-token').read_text().strip()
        request = urllib.request.Request(self.base + path,
            data=None if data is None else json.dumps(data, allow_nan=False).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=4) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from None

    def action(self, name, **kwargs):
        return self.request('arm/action', {'action': name, **kwargs})

    def observe(self):
        state = self.request('arm/observe')
        state.pop('history', None)
        state['observed_at'] = time.time()
        return state

    def snapshot(self, after=0, timeout=2):
        """Fresh image and measured state. Timing is exposed, not called synchronized."""
        deadline = time.monotonic() + timeout
        while True:
            frame = self.request('camera/frame')
            state = self.observe()
            if frame['fresh'] and frame['age_seconds'] <= .8 and frame['captured_at'] > after:
                return {'frame': frame, 'arm': state,
                        'state_frame_skew_seconds': state['observed_at'] - frame['captured_at']}
            if time.monotonic() >= deadline:
                raise RuntimeError('No new camera frame within the observation deadline')
            time.sleep(.05)

    def run(self, positions, speed=3, on_tick=None):
        """Execute one bounded chunk. No implicit start, retry, or fault clearing."""
        state = self.observe()
        if not state['connected'] or state['stopped'] or state['fault']:
            raise RuntimeError('Start a healthy arm session before moving')
        plan = self.action('plan', waypoints=[{'positions': positions}], speed=speed)
        attempted = False
        try:
            attempted = True
            state = self.action('execute', plan_id=plan['id'])
            deadline = time.monotonic() + plan['duration'] + 7
            while state['motion']['state'] == 'running':
                if time.monotonic() > deadline:
                    raise RuntimeError('Controller motion deadline exceeded')
                if on_tick:
                    on_tick(state)
                time.sleep(.06)
                state = self.observe()
            if state['motion']['state'] != 'completed' or state['fault']:
                raise RuntimeError(state['fault'] or state['motion']['reason'])
            return state
        except BaseException:
            if attempted:
                self.action('stop')
            raise


def save_snapshot(snapshot, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = dict(snapshot['frame'])
    raw = base64.b64decode(frame.pop('data_url').split(',', 1)[1], validate=True)
    (directory / (name + '.jpg')).write_bytes(raw)
    (directory / (name + '.json')).write_text(json.dumps({**snapshot, 'frame': frame}, indent=2) + '\n')
