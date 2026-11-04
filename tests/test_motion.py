import math
import tempfile
import time
import unittest
from pathlib import Path

from bridge.core import Store, experiment
from bridge.motion import plan_motion, sample_motion
from bridge.robot import JOINTS, PhysicalArm


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.limits = {j: [0, 100] if j == 'gripper' else [-90, 90] for j in JOINTS}
        self.start = {j: 50. if j == 'gripper' else 0. for j in JOINTS}

    def test_continuous_path_limits_speed_acceleration_and_endpoints(self):
        plan = plan_motion(self.start, [{'positions': {'shoulder_pan': 24, 'wrist_flex': -12}},
                                       {'positions': {'shoulder_pan': -12}}], self.limits, 12, 40)
        self.assertEqual(sample_motion(plan, 0), self.start)
        self.assertEqual(sample_motion(plan, 100), plan['target'])
        dt = .001
        previous = sample_motion(plan, 0)
        velocity = {j: 0 for j in JOINTS}
        for i in range(1, math.ceil(plan['duration'] / dt) + 1):
            pose = sample_motion(plan, i * dt)
            for joint in JOINTS:
                v = (pose[joint] - previous[joint]) / dt
                self.assertLessEqual(abs(v), 12.001)
                self.assertLessEqual(abs(v - velocity[joint]) / dt, 40.01)
                self.assertTrue(self.limits[joint][0] <= pose[joint] <= self.limits[joint][1])
                velocity[joint] = v
            previous = pose
        self.assertEqual(plan['target']['wrist_flex'], -12)  # Inherited, never reset to zero.

    def test_rejects_invalid_paths_before_any_actuation(self):
        for points in [[], [{'positions': {'wrong': 1}}], [{'positions': {'shoulder_pan': 91}}],
                       [{'positions': {'shoulder_pan': float('nan')}}],
                       [{'positions': {'shoulder_pan': 30}, 'duration': .1}]]:
            with self.assertRaises(ValueError):
                plan_motion(self.start, points, self.limits)

    def make_arm(self):
        class Robot:
            is_connected, is_calibrated = False, True
            def __init__(self, start):
                self.positions, self.commands, self.stuck = dict(start), [], False
                self.current = 0
            def connect(self, **kwargs): self.is_connected = True
            def disconnect(self): self.is_connected = False
            def get_observation(self): return {j + '.pos': v for j, v in self.positions.items()}
            def send_action(self, target):
                self.commands.append(target)
                if not self.stuck:
                    self.positions = {j: target[j + '.pos'] for j in JOINTS}
                return target
            def read_safety(self):
                return {name: {j: value for j in JOINTS} for name, value in
                        [('temperature_c', 30), ('current_raw', self.current), ('load_raw', 0),
                         ('voltage_raw', 120), ('torque_enabled', 1)]}
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        store = Store(Path(temp.name) / 'test.sqlite')
        self.addCleanup(store.db.close)
        exp = store.put('experiments', experiment({'recipe': 'agent_session', 'name': 'test', 'task': 'Reach'}))
        robot = Robot(self.start)
        config = {'enabled': True, 'port': 'fake', 'id': 'fake', 'calibration_dir': temp.name,
                  'limits': self.limits, 'require_camera': True}
        self.camera_ready = True
        arm = PhysicalArm(store, config, factory=lambda _: robot, watchdog=False, camera_ready=lambda: self.camera_ready)
        self.addCleanup(arm.close)
        arm.action({'action': 'connect'})
        arm.action({'action': 'start', 'experiment_id': exp['id']})
        return arm, robot

    def queue(self, arm, target=10):
        plan = arm.action({'action': 'plan', 'waypoints': [{'positions': {'shoulder_pan': target}}]})
        arm.action({'action': 'execute', 'plan_id': plan['id']})
        return plan

    def test_plan_is_read_only_single_use_and_completion_requires_feedback(self):
        arm, robot = self.make_arm()
        plan = arm.action({'action': 'plan', 'waypoints': [{'positions': {'shoulder_pan': 10}}]})
        self.assertEqual(robot.commands, [])
        arm.action({'action': 'execute', 'plan_id': plan['id']})
        with self.assertRaises(ValueError): arm.action({'action': 'execute', 'plan_id': plan['id']})
        now = arm.motion['started']
        for i in range(1, 200):
            arm._advance_motion(now + i / 30)
            if arm.motion['state'] == 'completed': break
        self.assertEqual(arm.motion['state'], 'completed')
        self.assertLess(abs(robot.positions['shoulder_pan'] - 10), .01)
        self.assertTrue(arm.session['motions'][-1]['samples'])

    def test_rejects_moved_pose_expired_plan_and_camera_loss(self):
        arm, robot = self.make_arm()
        plan = arm.action({'action': 'plan', 'waypoints': [{'positions': {'shoulder_pan': 10}}]})
        robot.positions['elbow_flex'] = 3
        with self.assertRaisesRegex(ValueError, 'moved'): arm.action({'action': 'execute', 'plan_id': plan['id']})
        plan = arm.action({'action': 'plan', 'waypoints': [{'positions': {'shoulder_pan': 10}}]})
        arm.plan['created'] -= 21
        with self.assertRaisesRegex(ValueError, 'expired'): arm.action({'action': 'execute', 'plan_id': plan['id']})
        self.queue(arm)
        self.camera_ready = False
        with self.assertRaisesRegex(ValueError, 'camera'): arm._require_camera()
        self.assertEqual(arm.motion['state'], 'stopped')
        self.assertTrue(arm.stopped)

    def test_deadline_tracking_current_and_stop_preempt_motion(self):
        arm, robot = self.make_arm()
        self.queue(arm)
        with self.assertRaisesRegex(ValueError, 'deadline'): arm._advance_motion(arm.motion['started'] + .3)
        robot.positions['elbow_flex'] = 4
        with self.assertRaisesRegex(ValueError, 'tracking'): arm._advance_motion(arm.motion['started'] + .03)
        robot.positions['elbow_flex'] = 0
        robot.current = 151
        arm.last_health = arm.motion['started'] - 1
        with self.assertRaisesRegex(ValueError, 'current'): arm._advance_motion(arm.motion['started'] + .06)
        arm.action({'action': 'stop'})
        self.assertTrue(arm.stopped)
        self.assertEqual(arm.motion['state'], 'stopped')
        self.assertNotEqual(arm.motion['state'], 'completed')

    def test_static_offset_trim_is_bounded_and_does_not_squeeze_gripper(self):
        arm, robot = self.make_arm()
        self.queue(arm, 1.8)
        robot.stuck = True
        now = arm.motion['started']
        with self.assertRaisesRegex(ValueError, 'tracking'):
            for i in range(1, 100):
                arm._advance_motion(now + i / 30)
        self.assertLessEqual(abs(arm.motion['correction']['shoulder_pan']), 2)
        self.assertGreater(arm.motion['correction']['shoulder_pan'], 0)
        self.assertEqual(arm.motion['correction']['gripper'], 0)
        self.assertNotEqual(arm.motion['state'], 'completed')
        self.assertLessEqual(max(abs(c['shoulder_pan.pos']) for c in robot.commands), 4)

    def test_fault_clear_requires_healthy_feedback_and_keeps_motion_stopped(self):
        arm, robot = self.make_arm()
        arm.action({'action': 'stop'})
        arm.fault = 'Previously inspected endpoint error'
        robot.current = 151
        with self.assertRaisesRegex(ValueError, 'current'): arm.action({'action': 'clear_fault'})
        self.assertTrue(arm.fault)
        robot.current = 0
        state = arm.action({'action': 'clear_fault'})
        self.assertFalse(state['fault'])
        self.assertTrue(state['stopped'])


if __name__ == '__main__':
    unittest.main()
