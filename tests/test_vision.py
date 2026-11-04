import base64
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from bridge.core import Store, experiment
from bridge.robot import JOINTS, PhysicalArm, configure_position_hold, validate_calibration_ranges
from bridge.vision import CameraFeed


class VisionTests(unittest.TestCase):
    def test_incomplete_hand_movement_is_not_accepted_as_calibration(self):
        calibration = {j: SimpleNamespace(range_min=1000, range_max=3000) for j in JOINTS}
        validate_calibration_ranges(calibration)
        calibration['shoulder_lift'].range_max = 1006
        with self.assertRaisesRegex(ValueError, 'shoulder_lift'):
            validate_calibration_ranges(calibration)
        calibration['shoulder_lift'].range_max = 3000
        del calibration['elbow_flex']
        with self.assertRaisesRegex(ValueError, 'all six'):
            validate_calibration_ranges(calibration)

    def setUp(self):
        self.now = 1000
        self.feed = CameraFeed(clock=lambda: self.now, wall=lambda: self.now)
        self.frame = {'data_url': 'data:image/png;base64,' + base64.b64encode(b'\x89PNG\r\n\x1a\ntest').decode(), 'captured_at': self.now}

    def test_opt_in_expiry_and_capture_age(self):
        self.assertFalse(self.feed.ready())
        with self.assertRaises(ValueError):
            self.feed.frame()
        lease = self.feed.start()['lease']
        old = {**self.frame, 'captured_at': self.now - 2}
        self.feed.put(lease, old)
        self.assertTrue(self.feed.ready())
        self.now += 2
        self.assertFalse(self.feed.ready())
        with self.assertRaises(ValueError):
            self.feed.frame()
        with self.assertRaises(ValueError):
            self.feed.put(lease, old)

    def test_stopped_or_superseded_upload_cannot_reactivate_camera(self):
        first = self.feed.start()['lease']
        self.feed.put(first, self.frame)
        second = self.feed.start()['lease']
        self.feed.stop(first)
        self.feed.put(second, self.frame)
        with self.assertRaises(ValueError):
            self.feed.put(first, self.frame)
        with self.assertRaises(ValueError):
            self.feed.put(second, self.frame)
        self.feed.stop(second)
        with self.assertRaises(ValueError):
            self.feed.put(second, {**self.frame, 'captured_at': self.now + .1})
        self.assertIsNone(self.feed.latest)

    def test_physical_motion_requires_feed_and_watchdog_holds_on_loss(self):
        class Robot:
            is_connected = False
            is_calibrated = True
            def __init__(self): self.commands = []
            def connect(self, **kwargs): self.is_connected = True
            def disconnect(self): self.is_connected = False
            def get_observation(self): return {j + '.pos': 50 if j == 'gripper' else 0 for j in JOINTS}
            def send_action(self, target): self.commands.append(target)
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'test.sqlite')
            exp = store.put('experiments', experiment({'recipe': 'agent_session', 'name': 'test', 'task': 'Move'}))
            robot = Robot()
            config = {'enabled': True, 'require_camera': True, 'port': 'test', 'id': 'test', 'calibration_dir': temp,
                      'limits': {j: [0, 100] if j == 'gripper' else [-90, 90] for j in JOINTS}}
            arm = PhysicalArm(store, config, factory=lambda _: robot, camera_ready=self.feed.ready)
            try:
                with self.assertRaisesRegex(ValueError, 'camera'):
                    arm.action({'action': 'connect'})
                self.assertFalse(robot.is_connected)
                lease = self.feed.start()['lease']
                self.feed.put(lease, self.frame)
                arm.action({'action': 'connect'})
                arm.action({'action': 'start', 'experiment_id': exp['id']})
                self.feed.stop(lease)
                deadline = time.monotonic() + 1
                while not arm.stopped and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(arm.stopped)
                self.assertTrue(robot.commands)  # Watchdog requests a hold, never a new trajectory.
                self.assertEqual(arm.session['history'][-1]['note'], 'Camera feed lost')
                with self.assertRaisesRegex(ValueError, 'camera'):
                    arm.action({'action': 'start', 'experiment_id': exp['id']})
            finally:
                arm.close()
                store.db.close()

    def test_connect_seeds_measured_goals_and_never_enables_after_failure(self):
        class Bus:
            def __init__(self): self.events = []; self.goals = {j: 999 for j in JOINTS}
            def disable_torque(self): self.events.append('off')
            def configure_motors(self): pass
            def set_operating_mode(self, *args): pass
            def write(self, *args, **kwargs): pass
            def sync_read(self, name, **kwargs):
                return {j: 50 if j == 'gripper' else 5 for j in JOINTS} if name == 'Present_Position' else self.goals
            def sync_write(self, name, values, **kwargs): self.goals = dict(values); self.events.append('targets')
            def enable_torque(self): self.events.append('on')
        bus = Bus()
        robot = SimpleNamespace(bus=bus, config=SimpleNamespace(position_p_coefficient=16, position_i_coefficient=0, position_d_coefficient=32))
        limits = {j: [0, 100] if j == 'gripper' else [-90, 90] for j in JOINTS}
        configure_position_hold(robot, limits)
        self.assertEqual(bus.events, ['off', 'targets', 'on'])
        self.assertEqual(bus.goals['shoulder_pan'], 5)
        bus.events.clear()
        limits['shoulder_pan'] = [-1, 1]
        with self.assertRaises(ValueError): configure_position_hold(robot, limits)
        self.assertEqual(bus.events, ['off'])


if __name__ == '__main__':
    unittest.main()
