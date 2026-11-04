"""Interactive SO101 calibration without enabling motor torque on connection."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge.robot import validate_calibration_ranges


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', required=True)
    p.add_argument('--id', default='armature_follower')
    p.add_argument('--calibration-dir', type=Path, default=ROOT / '.armature' / 'calibration')
    args = p.parse_args()
    if (args.calibration_dir / (args.id + '.json')).exists():
        p.error('Calibration already exists; use LeRobot to review or replace it explicitly')
    print('Loading calibration SDK...', flush=True)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import _HOMING_POSITION_DIAGRAM

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, calibration_dir=args.calibration_dir, cameras={}, use_degrees=True))
    # Open just the bus. robot.connect() configures and enables torque, which is
    # unnecessary during hand calibration and can engage a previous goal.
    robot.bus.connect()
    record_ranges = robot.bus.record_ranges_of_motion
    robot.bus.record_ranges_of_motion = lambda motors: record_ranges(motors, display_values=False)
    try:
        print(_HOMING_POSITION_DIAGRAM, flush=True)
        calibration = robot.bus.record_calibration('position', full_turn_motors=['wrist_roll'])
        validate_calibration_ranges(calibration)
        robot.bus.write_calibration(calibration)
        robot.calibration = calibration
        robot._save_calibration()
        print('Calibration saved to ' + str(robot.calibration_fpath), flush=True)
        print(json.dumps({name: asdict(value) for name, value in robot.calibration.items()}, indent=2), flush=True)
    finally:
        robot.bus.disconnect(disable_torque=True)


if __name__ == '__main__':
    main()
