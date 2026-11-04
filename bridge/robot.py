"""Optional SO101 joint control. Importing/starting the bridge never opens motors."""
import threading
import time
from pathlib import Path

from .core import number, string, uid
from .motion import plan_motion, sample_motion

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def validate_calibration_ranges(calibration):
    if set(calibration) != set(JOINTS):
        raise ValueError('Calibration must include all six joints')
    invalid = [name for name, value in calibration.items()
               if not 0 <= value.range_min < value.range_max <= 4095
               or (name != 'wrist_roll' and value.range_max - value.range_min < 200)]
    if invalid:
        raise ValueError('Incomplete calibration range: ' + ', '.join(invalid))


def configure_position_hold(robot, limits, dynamics=None):
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
        if dynamics:
            for register, value in dynamics[joint].items():
                bus.write(register, joint, value, normalize=False)
                if bus.read(register, joint, normalize=False) != value:
                    raise ValueError('Motor setting did not verify: ' + joint + ' ' + register)
    current = bus.sync_read('Present_Position')
    for joint in JOINTS:
        number(current[joint], *limits[joint], joint)
    raw = bus.sync_read('Present_Position', normalize=False)
    if getattr(robot, 'calibration', None):
        for joint, value in raw.items():
            calibration = robot.calibration[joint]
            number(value, calibration.range_min, calibration.range_max, joint + ' raw position')
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
            configure_position_hold(self, config['limits'], config.get('motor_settings'))

        def read_safety(self):
            return {name: self.bus.sync_read(register, normalize=False, num_retry=0)
                    for name, register in [('temperature_c', 'Present_Temperature'),
                                           ('current_raw', 'Present_Current'),
                                           ('load_raw', 'Present_Load'),
                                           ('voltage_raw', 'Present_Voltage'),
                                           ('torque_enabled', 'Torque_Enable')]}

        def send_motion_action(self, target):
            # Continuous trajectories have their own measured lead, speed, and
            # acceleration checks. The separate one-step jog keeps its SDK cap.
            self.bus.sync_write('Goal_Position', {j: target[j + '.pos'] for j in JOINTS})
            return target

        def calibrate(self):
            # LeRobot checks calibration before configuring motor torque.
            # Never enter an interactive calibration flow from a web request.
            raise ValueError('Calibration mismatch; recalibrate locally with LeRobot')

    robot = CalibratedFollower(SO101FollowerConfig(
        port=config['port'], id=config['id'], calibration_dir=Path(config['calibration_dir']),
        use_degrees=True, cameras={}, disable_torque_on_disconnect=False,
        position_p_coefficient=config.get('position_p_coefficient', 16),
        max_relative_target={j: config['max_gripper_step'] if j == 'gripper' else config['max_step'] for j in JOINTS}))
    if not robot.calibration:
        raise ValueError('Calibration missing; calibrate with LeRobot before connecting')
    validate_calibration_ranges(robot.calibration)
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
        self.plan = None
        self.motion = None
        self.telemetry = {}
        self.last_health = 0
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
        self.config['motion_speed'] = number(self.config.get('motion_speed', 6), .1, 30, 'Motion speed')
        self.config['motion_acceleration'] = number(self.config.get('motion_acceleration', 20), .1, 100, 'Motion acceleration')
        self.config['max_current_raw'] = number(self.config.get('max_current_raw', 150), 1, 250, 'Current threshold')
        self.config['position_p_coefficient'] = number(self.config.get('position_p_coefficient', 16), 8, 32, 'Position gain', integer=True)
        settings = self.config.get('motor_settings')
        if settings is not None:
            if not isinstance(settings, dict) or set(settings) != set(JOINTS):
                raise ValueError('Provide motor settings for all six joints')
            for joint, values in settings.items():
                if not isinstance(values, dict) or set(values) != {'Goal_Velocity', 'Acceleration', 'Torque_Limit'}:
                    raise ValueError('Motor settings require velocity, acceleration and torque limit')
                number(values['Goal_Velocity'], 1, 400, 'Motor velocity', integer=True)
                number(values['Acceleration'], 1, 12, 'Motor acceleration', integer=True)
                number(values['Torque_Limit'], 1, 500 if joint == 'gripper' else 800, 'Torque limit', integer=True)

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
                    self.fault = 'Feedback unavailable; check the arm locally'
                    self._stop(self.fault)
                    self._record('fault', self.fault)
                    raise ValueError(self.fault)
            self.revision += 1
            self.observation_time = time.monotonic()
            return {**self.status(), 'revision': self.revision, 'positions': dict(self.positions),
                    'limits': self.config.get('limits', {}), 'session_id': self.session['id'] if self.session else None,
                    'history': self.session['history'][-10:] if self.session else [],
                    'motion': self._motion_status(), 'telemetry': self.telemetry}

    def _motion_status(self):
        if not self.motion:
            return None
        return {k: self.motion[k] for k in ('id', 'state', 'duration', 'elapsed', 'steps',
                                           'max_error', 'max_tick_seconds', 'reason', 'target',
                                           'final_error', 'correction')}

    def _health(self):
        if not hasattr(self.robot, 'read_safety'):
            raise ValueError('Motion requires motor health feedback')
        values = self.robot.read_safety()
        for name in ('temperature_c', 'current_raw', 'load_raw', 'voltage_raw', 'torque_enabled'):
            if set(values.get(name, {})) != set(JOINTS):
                raise ValueError('Missing motor health feedback: ' + name)
            for joint, value in values[name].items():
                number(value, -65535, 65535, name + ' ' + joint)
        self.telemetry = {**values, 'read_at': time.time()}
        self.last_health = time.monotonic()
        for joint in JOINTS:
            if values['temperature_c'][joint] >= 50:
                raise ValueError('Motor temperature limit: ' + joint)
            if abs(values['current_raw'][joint]) > self.config['max_current_raw']:
                raise ValueError('Motor current limit: ' + joint)
            if abs(values['load_raw'][joint]) > 900:
                raise ValueError('Motor load limit: ' + joint)
            if not 60 <= values['voltage_raw'][joint] <= 130:
                raise ValueError('Motor voltage limit: ' + joint)
            if values['torque_enabled'][joint] != 1:
                raise ValueError('Motor lost holding torque: ' + joint)

    def _finish_motion(self, state, reason=''):
        if self.motion and self.motion['state'] == 'running':
            self.motion['state'], self.motion['reason'] = state, reason
            if self.session:
                self.session.setdefault('motions', []).append({
                    **self._motion_status(), 'plan': self.motion['plan'],
                    'samples': self.motion['samples'], 'telemetry': self.telemetry})
                self._record('motion_' + state, reason, self.motion['target'])

    def _stop(self, reason=''):
        self.stopped = True  # Latch even if the bus fails.
        self.revision += 1
        self.plan = None
        self._finish_motion('stopped', reason or self.fault or 'Stop requested')
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
                if self.motion and self.motion['state'] == 'running':
                    raise ValueError('Stop the current motion first')
                self.session = {'id': uid(), 'created_at': time.time(), 'experiment_id': data['experiment_id'], 'mode': 'so101', 'history': []}
                self.stopped = False
                self.last_action = time.monotonic()
                self._record('start')
                return self.observe()
            if action == 'clear_fault':
                if not self.stopped:
                    raise ValueError('Stop the arm before clearing a fault')
                self._require_camera()
                self._health()
                current = self._read()
                for joint in JOINTS:
                    number(current[joint], *self.config['limits'][joint], joint)
                self.fault = ''
                self._stop('Fault inspected; current pose held')
                if self.fault:
                    raise ValueError(self.fault)
                return self.observe()
            if action == 'note':
                if not self.session:
                    raise ValueError('Start a session first')
                self._record('note', string(data.get('note'), 'Note', 2000))
                return self.observe()
            if action == 'plan':
                if self.motion and self.motion['state'] == 'running':
                    raise ValueError('Wait for the current motion or stop it first')
                current = self._read()
                speed = number(data.get('speed', self.config['motion_speed']), .1, self.config['motion_speed'], 'Speed')
                acceleration = number(data.get('acceleration', self.config['motion_acceleration']), .1, self.config['motion_acceleration'], 'Acceleration')
                plan = plan_motion(current, data.get('waypoints'), self.config['limits'], speed, acceleration)
                self.plan = {**plan, 'id': uid(), 'created': time.monotonic()}
                return {k: v for k, v in self.plan.items() if k != 'created'}
            if action == 'execute':
                if self.stopped or self.fault:
                    raise ValueError('Arm is stopped')
                self._require_camera()
                if not self.plan or data.get('plan_id') != self.plan['id']:
                    raise ValueError('Unknown or already executed plan')
                if time.monotonic() - self.plan['created'] > 20:
                    self.plan = None
                    raise ValueError('Plan expired; plan again')
                current = self._read()
                if any(abs(current[j] - self.plan['start'][j]) > 1 for j in JOINTS):
                    self.plan = None
                    raise ValueError('Arm moved since planning; plan again')
                try:
                    self._health()
                except Exception as error:
                    self.fault = str(error)
                    self._stop(self.fault)
                    raise ValueError(self.fault)
                now = time.monotonic()
                plan, self.plan = self.plan, None
                self.motion = {'id': plan['id'], 'state': 'running', 'plan': plan, 'started': now,
                               'duration': plan['duration'], 'elapsed': 0., 'steps': 0, 'max_error': 0.,
                               'max_tick_seconds': 0., 'last_tick': now, 'reason': '', 'samples': [],
                               'target': plan['target'], 'previous_target': current, 'settled': 0,
                               'final_error': None, 'correction': {j: 0. for j in JOINTS}}
                self.last_action = now
                self._record('motion_start', target=plan['target'])
                return self.observe()
            if action != 'jog':
                raise ValueError('Unknown arm action')
            if self.stopped or self.fault:
                raise ValueError('Arm is stopped')
            if self.motion and self.motion['state'] == 'running':
                raise ValueError('Stop the trajectory before jogging')
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

    def _advance_motion(self, now):
        motion = self.motion
        gap = now - motion['last_tick']
        motion['max_tick_seconds'] = max(motion['max_tick_seconds'], gap)
        if gap > .25:
            raise ValueError('Motion loop deadline exceeded')
        motion['last_tick'] = now
        current = self._read()
        self.positions = current
        for joint in JOINTS:
            number(current[joint], *self.config['limits'][joint], joint)
        error = max(abs(current[j] - motion['previous_target'][j]) for j in JOINTS)
        motion['max_error'] = max(motion['max_error'], error)
        if error > 3:
            raise ValueError('Motion tracking error exceeds 3 units')
        if now - self.last_health >= .5:
            self._health()
        elapsed = now - motion['started']
        target = sample_motion(motion['plan'], elapsed)
        final_error = max(abs(current[j] - motion['target'][j]) for j in JOINTS)
        motion['final_error'] = final_error
        if elapsed >= motion['duration']:
            # Correct small static joint offsets with a bounded, slow trim.
            # Never increase closing force on an object via gripper integration.
            for joint in JOINTS:
                residual = motion['target'][joint] - current[joint]
                if joint != 'gripper' and abs(residual) > .5:
                    trim = motion['correction'][joint] + max(-1., min(1., residual)) * min(gap, .05)
                    motion['correction'][joint] = max(-2., min(2., trim))
                target[joint] = max(self.config['limits'][joint][0], min(self.config['limits'][joint][1],
                                      target[joint] + motion['correction'][joint]))
        if any(abs(target[j] - current[j]) > 4 for j in JOINTS):
            raise ValueError('Trajectory command exceeds 4-unit feedback lead')
        sender = getattr(self.robot, 'send_motion_action', self.robot.send_action)
        sent = sender({j + '.pos': v for j, v in target.items()})
        if sent is None or any(abs(sent[j + '.pos'] - target[j]) > .15 for j in JOINTS):
            raise ValueError('Motor adapter clipped the trajectory; reduce speed')
        motion['previous_target'] = target
        motion['elapsed'], motion['steps'] = elapsed, motion['steps'] + 1
        if not motion['samples'] or elapsed - motion['samples'][-1]['t'] >= .1:
            motion['samples'].append({'t': elapsed, 'positions': current, 'target': target, 'telemetry': self.telemetry})
        self.last_action = now
        if elapsed >= motion['duration']:
            motion['settled'] = motion['settled'] + 1 if final_error <= 1 else 0
            if motion['settled'] >= 3:
                self._finish_motion('completed')
            elif elapsed > motion['duration'] + 5:
                raise ValueError('Motion failed to reach its endpoint')

    def _watch(self):
        delay = 1 / 30
        while not self.closed.wait(delay):
            tick_started = time.monotonic()
            with self.lock:
                if self.robot and not self.stopped:
                    if self.config.get('require_camera') and not self.camera_ready():
                        self._stop('Camera feed lost')
                    elif self.motion and self.motion['state'] == 'running':
                        try:
                            self._advance_motion(time.monotonic())
                        except Exception as error:
                            self.fault = str(error) or 'Motion feedback failed'
                            self._stop(self.fault)
                    elif time.monotonic() - self.last_action > 10:
                        self._stop('Idle timeout')
            delay = max(.001, 1 / 30 - (time.monotonic() - tick_started))

    def close(self):
        self.closed.set()
        with self.lock:
            if self.robot:
                self._stop('Bridge closing')
                self.robot.disconnect()
                self.robot = None
        if self.thread:
            self.thread.join(timeout=2)
