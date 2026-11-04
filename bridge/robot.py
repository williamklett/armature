"""Optional SO101 joint control. Importing/starting the bridge never opens motors."""
import threading
import time
from pathlib import Path

from .core import number, string, uid

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def configure_position_hold(robot, limits):
    """Configure the pinned SO101 bus, seeding current goals before torque on."""
    bus = robot.bus
    bus.disable_torque()
    # Do not use torque_disabled(): its finally block enables torque on failure.
    bus.configure_motors()
    for joint in JOINTS:
        bus.set_operating_mode('position', joint)
        bus.write('P_Coefficient', joint, robot.config.position_p_coefficient)
        bus.write('I_Coefficient', joint, robot.config.position_i_coefficient)
        bus.write('D_Coefficient', joint, robot.config.position_d_coefficient)
        bus.write('Goal_Velocity', joint, 100, normalize=False)
        if joint == 'gripper':
            bus.write('Max_Torque_Limit', joint, 500)
            bus.write('Protection_Current', joint, 250)
            bus.write('Overload_Torque', joint, 25)
    current = bus.sync_read('Present_Position')
    for joint in JOINTS:
        number(current[joint], *limits[joint], joint)
    raw = bus.sync_read('Present_Position', normalize=False)
    bus.sync_write('Goal_Position', raw, normalize=False)
    goals = bus.sync_read('Goal_Position', normalize=False)
    if any(goals[joint] != raw[joint] for joint in JOINTS):
        raise ValueError('Initial hold targets did not verify; torque remains off')
    bus.enable_torque()


def lerobot_factory(config):
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    class CalibratedFollower(SO101Follower):
        def configure(self):
            configure_position_hold(self, config['limits'])

        def calibrate(self):
            # LeRobot checks calibration before configuring motor torque.
            # Never enter an interactive calibration flow from a web request.
            raise ValueError('Calibration mismatch; recalibrate locally with LeRobot')

    robot = CalibratedFollower(SO101FollowerConfig(
        port=config['port'], id=config['id'], calibration_dir=Path(config['calibration_dir']),
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
        max_relative_target={j: config['max_gripper_step'] if j == 'gripper' else config['max_step'] for j in JOINTS}))
    if not robot.calibration:
        raise ValueError('Calibration missing; calibrate with LeRobot before connecting')
    return robot


class PhysicalArm:
    def __init__(self, store, config=None, factory=lerobot_factory, watchdog=True, camera_ready=lambda: False):
        self.store, self.config, self.factory = store, config or {}, factory
        self.camera_ready = camera_ready
        self.lock = threading.RLock()
        self.robot = None
        self.stopped = True
        self.fault = ''
        self.revision = 0
        self.observation_time = 0
        self.positions = {}
        self.last_action = 0
        self.session = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True) if watchdog else None
        if self.thread:
            self.thread.start()

    def status(self):
        return {'configured': self.config.get('enabled') is True, 'connected': self.robot is not None,
                'stopped': self.stopped, 'fault': self.fault, 'mode': 'so101',
                'camera_required': self.config.get('require_camera', False), 'camera_ready': self.camera_ready()}

    def _require_camera(self):
        if self.config.get('require_camera') and not self.camera_ready():
            if self.robot and not self.stopped:
                self._stop('Camera feed lost')
            raise ValueError('Fresh camera feed required; enable Agent view')

    def _config(self):
        if self.config.get('enabled') is not True:
            raise ValueError('Physical arm is disabled in bridge.local.json')
        for key in ('port', 'id', 'calibration_dir'):
            string(self.config.get(key), key, 500)
        limits = self.config.get('limits', {})
        if set(limits) != set(JOINTS):
            raise ValueError('Set measured limits for all six joints')
        for j in JOINTS:
            pair = limits[j]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Set measured limits for ' + j)
            low, high = [number(v, 0 if j == 'gripper' else -180, 100 if j == 'gripper' else 180, j) for v in pair]
            if low >= high:
                raise ValueError('Invalid joint range: ' + j)
        self.config['max_step'] = number(self.config.get('max_step', 1), 0.1, 2, 'Joint step')
        self.config['max_gripper_step'] = number(self.config.get('max_gripper_step', 2), 0.1, 3, 'Gripper step')

    def _read(self):
        raw = self.robot.get_observation()
        return {j: number(float(raw[j + '.pos']), -360, 360, j) for j in JOINTS}

    def _record(self, action, note='', target=None):
        if self.session:
            self.session['history'].append({'time': time.time(), 'action': action, 'note': note, 'positions': dict(self.positions), 'target': target})
            self.session['state'] = self.status()
            self.store.put('sessions', self.session)

    def observe(self):
        with self.lock:
            if self.robot:
                try:
                    self.positions = self._read()
                except Exception:
                    self.stopped = True
                    self.fault = 'Feedback unavailable; check the arm locally'
                    self.revision += 1
                    self._record('fault', self.fault)
                    raise ValueError(self.fault)
            self.revision += 1
            self.observation_time = time.monotonic()
            return {**self.status(), 'revision': self.revision, 'positions': dict(self.positions),
                    'limits': self.config.get('limits', {}), 'session_id': self.session['id'] if self.session else None,
                    'history': self.session['history'][-50:] if self.session else []}

    def _stop(self, reason=''):
        self.stopped = True  # Latch even if the bus fails.
        self.revision += 1
        if self.robot:
            try:
                self.positions = self._read()
                self.robot.send_action({j + '.pos': v for j, v in self.positions.items()})
            except Exception:
                self.fault = 'Hold failed; check the arm locally'
        self._record('stop', reason or self.fault)

    def action(self, data):
        with self.lock:
            action = data.get('action')
            if action == 'connect':
                if self.robot:
                    raise ValueError('Arm already connected')
                self._config()
                self._require_camera()
                robot = self.factory(self.config)
                try:
                    self._require_camera()
                    robot.connect(calibrate=True)
                    if not robot.is_calibrated:
                        raise ValueError('Calibration does not match the motors')
                except Exception:
                    if robot.is_connected:
                        robot.disconnect()
                    raise
                self.robot = robot
                self.stopped, self.fault = True, ''
                return self.observe()
            if action == 'stop':
                self._stop()
                return self.observe()
            if not self.robot:
                raise ValueError('Connect the arm first')
            if action == 'disconnect':
                self._stop('Disconnected')
                self.robot.disconnect()  # Keep torque; never deliberately drop a held object.
                self.robot = None
                return self.observe()
            if action == 'start':
                self._require_camera()
                self.store.get('experiments', data.get('experiment_id'))
                if self.fault:
                    raise ValueError(self.fault)
                self.session = {'id': uid(), 'created_at': time.time(), 'experiment_id': data['experiment_id'], 'mode': 'so101', 'history': []}
                self.stopped = False
                self.last_action = time.monotonic()
                self._record('start')
                return self.observe()
            if action == 'note':
                if not self.session:
                    raise ValueError('Start a session first')
                self._record('note', string(data.get('note'), 'Note', 2000))
                return self.observe()
            if action != 'jog':
                raise ValueError('Unknown arm action')
            if self.stopped or self.fault:
                raise ValueError('Arm is stopped')
            self._require_camera()
            now = time.monotonic()
            if data.get('revision') != self.revision or now - self.observation_time > 0.75:
                raise ValueError('Stale observation; observe again')
            if now - self.last_action < 0.1:
                raise ValueError('Wait before the next step')
            joint = data.get('joint')
            if joint not in JOINTS:
                raise ValueError('Unknown joint')
            step = self.config['max_gripper_step'] if joint == 'gripper' else self.config['max_step']
            delta = number(data.get('delta'), -step, step, 'Step')
            try:
                current = self._read()
            except Exception:
                self.fault = 'Feedback unavailable; check the arm locally'
                self._stop(self.fault)
                raise ValueError(self.fault)
            if any(abs(current[j] - self.positions[j]) > (self.config['max_gripper_step'] if j == 'gripper' else self.config['max_step']) for j in JOINTS):
                raise ValueError('Arm moved since observation; observe again')
            target = dict(current)
            target[joint] += delta
            if any(not self.config['limits'][j][0] <= target[j] <= self.config['limits'][j][1] for j in JOINTS):
                raise ValueError('Joint limit exceeded')
            note = string(data.get('note', ''), 'Note', 2000, False)
            try:
                self.robot.send_action({j + '.pos': v for j, v in target.items()})
                self.last_action = now
                self._record('jog', note, target)
            except Exception:
                self.fault = 'Motor command failed; check the arm locally'
                self._stop(self.fault)
                raise ValueError(self.fault)
            return self.observe()

    def _watch(self):
        while not self.closed.wait(0.1):
            with self.lock:
                if self.robot and not self.stopped:
                    if self.config.get('require_camera') and not self.camera_ready():
                        self._stop('Camera feed lost')
                    elif time.monotonic() - self.last_action > 10:
                        self._stop('Idle timeout')

    def close(self):
        self.closed.set()
        with self.lock:
            if self.robot:
                self._stop('Bridge closing')
                self.robot.disconnect()
                self.robot = None
        if self.thread:
            self.thread.join(timeout=2)
