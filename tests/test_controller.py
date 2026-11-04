import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import numpy as np
    from controller.dynamics import LocalDynamics
    from controller.tracking import PointTracker
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Install configs/controller-requirements.txt for visual controller tests')
class ControllerTests(unittest.TestCase):
    def frame(self, image, serial):
        ok, raw = cv2.imencode('.png', image)
        self.assertTrue(ok)
        return {'data_url': 'data:image/png;base64,' + base64.b64encode(raw).decode(),
                'frame_id': str(serial), 'captured_at': float(serial), 'fresh': True, 'age_seconds': 0}

    def test_unseen_action_predictions_and_new_goals_without_robot_geometry(self):
        rng = np.random.default_rng(72)
        successes, prediction_errors = 0, []
        for trial in range(30):
            # Unknown rotated/signed control mapping for each independent plant.
            angle = rng.uniform(-np.pi, np.pi)
            matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            matrix = np.diag(rng.uniform(2, 4, 2)) @ matrix
            model = LocalDynamics(2, 2)
            q = np.zeros(2)
            for dq in ([2, 0], [-2, 0], [0, 2], [0, -2]):
                dq = np.array(dq, dtype=float)
                entry = model.update(q, dq, dq @ matrix + rng.normal(0, .08, 2))
                self.assertIsNone(entry['prediction_before_update'])
                q += dq
            goal = rng.uniform(-7, 7, 2)
            y = np.zeros(2)
            for step in range(6):
                if np.linalg.norm(y - goal) <= 1:
                    break
                plan = model.suggest(q, y, goal, [-6, -6], [6, 6], seed=step)
                dq = np.array(plan['delta'])
                dy = dq @ matrix + rng.normal(0, .08, 2)
                record = model.update(q, dq, dy)
                prediction_errors.append(record['error_before_update_px'])
                self.assertLessEqual(np.max(np.abs(dq)), 2)
                q += dq
                y += dy
            successes += np.linalg.norm(y - goal) <= 1
        self.assertGreaterEqual(successes, 28)
        self.assertLess(float(np.mean(prediction_errors)), .3)

    def test_degenerate_excitation_and_nonfinite_data_rejected(self):
        model = LocalDynamics(2, 2)
        for _ in range(6):
            model.update([0, 0], [1, 0], [2, 0])
        with self.assertRaisesRegex(ValueError, 'underdetermined'):
            model.suggest([0, 0], [0, 0], [3, 3], [-6, -6], [6, 6])
        with self.assertRaises(ValueError):
            model.update([0, 0], [float('nan'), 1], [1, 1])
        with self.assertRaises(ValueError):
            model.update([0, 0], [0, 0], [1, 1])

    def test_tracking_translation_and_camera_motion_or_occlusion_stop(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (160, 220), dtype=np.uint8)
        initial = self.frame(image, 1)
        tracker = PointTracker(initial, [80, 70, 30, 30], [10, 10, 35, 35])
        moved = image.copy()
        moved[60:120, 70:130] = np.roll(image[60:120, 70:130], [3, 4], axis=(0, 1))
        point = tracker.update(self.frame(moved, 2))
        np.testing.assert_allclose(point, [99, 88], atol=.6)
        np.testing.assert_allclose(tracker.update(self.frame(moved, 2)), point)
        shifted = cv2.warpAffine(moved, np.float32([[1, 0, 4], [0, 1, 0]]), (220, 160))
        with self.assertRaises(ValueError):
            tracker.update(self.frame(shifted, 3))
        tracker = PointTracker(initial, [80, 70, 30, 30], [10, 10, 35, 35])
        occluded = image.copy()
        occluded[55:125, 65:135] = 0
        with self.assertRaises(ValueError):
            tracker.update(self.frame(occluded, 3))

    def test_actual_not_commanded_displacement_is_used_and_failure_stops(self):
        from controller.session import VisualSession
        from unittest.mock import Mock, patch
        from controller.client import BridgeClient
        client = Mock(spec=BridgeClient)
        session = VisualSession.__new__(VisualSession)
        session.client, session.joints = client, ['a', 'b']
        session.origin = {'a': 0, 'b': 0, 'c': 0}
        session.lower, session.upper = [-6, -6], [6, 6]
        session.valid, session.sequence = True, 1
        session.model = LocalDynamics(2, 2)
        session.tracker = Mock(position=np.zeros(2))
        session._record = Mock()
        state = {'connected': True, 'fault': '', 'positions': {'a': 0, 'b': 0, 'c': 0}}
        end = {**state, 'positions': {'a': 1.5, 'b': 0, 'c': 0}, 'motion': {'state': 'completed'}}
        def observe(after=0):
            if after:
                session.tracker.position = np.array([3., 1.])
                return {'arm': end}
            return {'arm': state}
        session.observe = observe
        client.run.return_value = end
        result = session.step([2, 0])
        self.assertEqual(result['dq'], [1.5, 0])
        self.assertEqual(result['dy'], [3, 1])
        self.assertEqual(result['commanded_delta'], [2, 0])
        client.run.side_effect = ValueError('Tracking lost')
        with self.assertRaisesRegex(ValueError, 'Tracking lost'):
            session.step([2, 0])
        client.action.assert_called_with('stop')
        self.assertFalse(session.valid)


class MCPTests(unittest.TestCase):
    def test_native_observation_returns_image_without_motion_or_token(self):
        try:
            from mcp import Client
            from scripts.armature_mcp import build_server
        except ImportError:
            self.skipTest('Install MCP controller environment')
        class Bridge:
            def __init__(self): self.calls = []
            def snapshot(self):
                return {'frame': {'data_url': 'data:image/jpeg;base64,YWJj', 'frame_id': 'test'},
                        'arm': {'positions': {'wrist_flex': 12}}, 'state_frame_skew_seconds': .2}
            def action(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {'stopped': True}
        bridge = Bridge()
        async def check():
            async with Client(build_server(bridge)) as client:
                result = await client.call_tool('arm_observe', {})
                self.assertFalse(result.is_error)
                self.assertEqual(result.content[1].type, 'image')
                self.assertEqual(result.content[1].data, 'YWJj')
                metadata = json.loads(result.content[0].text)
                self.assertNotIn('data_url', metadata['frame'])
                self.assertEqual(bridge.calls, [])
                result = await client.call_tool('arm_stop', {})
                self.assertFalse(result.is_error)
                self.assertEqual(bridge.calls, [(('stop',), {})])
        asyncio.run(check())


if __name__ == '__main__':
    unittest.main()
