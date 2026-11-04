"""Resident, bounded visual-feedback loop over the guarded motor bridge."""
import json
import time
import uuid

import numpy as np

from .client import save_snapshot
from .dynamics import LocalDynamics
from .tracking import PointTracker


class VisualSession:
    def __init__(self, client, joints, box, anchor_box):
        if not joints or len(set(joints)) != len(joints) or len(joints) > 5 or 'gripper' in joints:
            raise ValueError('Choose one to five distinct body joints; keep gripper geometry fixed')
        self.client, self.joints = client, list(joints)
        snapshot = client.snapshot()
        state = snapshot['arm']
        if not state['connected'] or state['fault'] or (state['motion'] and state['motion']['state'] == 'running'):
            raise ValueError('Initialize from a connected, healthy, stationary arm')
        if set(joints) - set(state['positions']):
            raise ValueError('Unknown control joint')
        self.origin = dict(state['positions'])
        self.lower = [max(state['limits'][j][0], self.origin[j] - 6) for j in joints]
        self.upper = [min(state['limits'][j][1], self.origin[j] + 6) for j in joints]
        self.tracker = PointTracker(snapshot['frame'], box, anchor_box)
        self.model = LocalDynamics(len(joints), 2)
        self.directory = client.state_dir / 'controllers' / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records, self.sequence, self.valid = [], 0, True
        self._save(snapshot)
        (self.directory / 'config.json').write_text(json.dumps({
            'joints': joints, 'box': box, 'anchor_box': anchor_box, 'origin': self.origin,
            'bounds': [self.lower, self.upper], 'model': 'local-ridge-visual-dynamics-v1',
            'camera_shape': list(self.tracker.previous.shape),
            'units': 'joint degrees and image pixels', 'grasp_verified': False}, indent=2) + '\n')

    def _save(self, snapshot):
        snapshot = {**snapshot, 'tracked_point': self.tracker.position.tolist()}
        save_snapshot(snapshot, self.directory, '%04d' % self.sequence)
        self.sequence += 1

    def _pose(self, state):
        if not self.valid:
            raise ValueError('Visual session invalidated; inspect and initialize again')
        if not state['connected'] or state['fault']:
            raise ValueError('Arm disconnected or faulted')
        for joint in self.origin.keys() - set(self.joints):
            if abs(state['positions'][joint] - self.origin[joint]) > 1:
                raise ValueError('An unmodeled joint moved; initialize a new visual session')
        q = np.array([state['positions'][j] for j in self.joints])
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError('Arm left the local exploration envelope')
        return q

    def observe(self, after=0):
        snapshot = self.client.snapshot(after=after)
        self._pose(snapshot['arm'])
        previous = self.tracker.frame_id
        self.tracker.update(snapshot['frame'])
        if previous != self.tracker.frame_id:
            self._save(snapshot)
        return snapshot

    def _record(self, record):
        self.records.append(record)
        with (self.directory / 'transitions.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        (self.directory / 'model.json').write_text(json.dumps(self.model.samples, indent=2) + '\n')

    def step(self, delta, speed=3, prediction=None):
        """A probe or policy action; use ACTUAL displacement as the learning label."""
        before = self.observe()
        q = self._pose(before['arm'])
        delta = self.model.vector(delta, len(self.joints))
        if np.max(np.abs(delta)) > 2 or not 1 <= speed <= 4:
            raise ValueError('Visual steps require at most 2 degrees and 1-4 degrees/s')
        target = q + delta
        if np.any(target < self.lower) or np.any(target > self.upper):
            raise ValueError('Requested action leaves the local exploration envelope')
        y = self.tracker.position.copy()
        started, last_frame = time.time(), 0.
        before_index = self.sequence - 1

        def tick(state):
            nonlocal last_frame
            self._pose(state)
            if time.monotonic() - last_frame >= .2:
                self.observe()
                last_frame = time.monotonic()

        try:
            end_state = self.client.run(dict(zip(self.joints, target.tolist())), speed, on_tick=tick)
            # The image must be captured AFTER motion completed, then the pose
            # checked for settling. No use of a stale pre-completion frame label.
            after = self.observe(after=time.time() + .1)
            end = self._pose(after['arm'])
            if max(abs(after['arm']['positions'][j] - end_state['positions'][j]) for j in self.origin) > .35:
                raise ValueError('Arm did not settle for the visual transition label')
            entry = self.model.update(q, end - q, self.tracker.position - y)
            result = {**entry, 'commanded_delta': delta.tolist(), 'before_frame': before_index,
                      'after_frame': self.sequence - 1, 'goal_prediction': prediction,
                      'elapsed_seconds': time.time() - started, 'motion': end_state['motion'],
                      'features': self.tracker.position.tolist()}
            self._record(result)
            return result
        except BaseException as error:
            self.valid = False
            self._record({'failure': str(error), 'commanded_delta': delta.tolist(), 'at': time.time()})
            self.client.action('stop')
            raise

    def reach(self, goal, max_steps=6, tolerance=2., speed=3):
        goal = self.model.vector(goal, 2)
        height, width = self.tracker.previous.shape
        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            raise ValueError('Visual goal is outside the camera image')
        if not 1 <= max_steps <= 12 or not 1 <= tolerance <= 10:
            raise ValueError('Invalid feedback-loop budget')
        start = self.observe()
        initial = float(np.linalg.norm(goal - self.tracker.position))
        if initial > 40:
            raise ValueError('Goal exceeds the local 40-pixel workspace')
        errors, begun, state = [initial], time.monotonic(), start['arm']
        try:
            for index in range(max_steps):
                if errors[-1] <= tolerance:
                    break
                if time.monotonic() - begun > 40:
                    raise ValueError('Visual control time budget exhausted')
                action = self.model.suggest(self._pose(state), self.tracker.position, goal,
                                            self.lower, self.upper, seed=index)
                self.step(action['delta'], speed, prediction=action)
                state = self.client.observe()
                errors.append(float(np.linalg.norm(goal - self.tracker.position)))
                if len(errors) >= 3 and errors[-1] > errors[-2] + 2 and errors[-2] > errors[-3] + 2:
                    raise ValueError('Two consecutive visual errors increased; inspect before resuming')
            report = {'goal': goal.tolist(), 'errors_px': errors, 'reached': errors[-1] <= tolerance,
                      'elapsed_seconds': time.monotonic() - begun, 'model_updates': len(self.model.samples),
                      'log_dir': str(self.directory.resolve()), 'grasp_verified': False}
            self._record({'evaluation': report})
            return report
        finally:
            self.client.action('stop')

    def status(self):
        return {'valid': self.valid, 'joints': self.joints, 'features': self.tracker.position.tolist(),
                'captured_at': self.tracker.captured_at, 'frame_id': self.tracker.frame_id,
                'transitions': len(self.model.samples), 'bounds': [self.lower, self.upper],
                'log_dir': str(self.directory.resolve())}
