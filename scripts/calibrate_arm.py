"""Interactive SO101 calibration without enabling motor torque on connection."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    robot = SO101Follower(SO101FollowerConfig(
        port=args.port, id=args.id, calibration_dir=args.calibration_dir, cameras={}, use_degrees=True))
    # Open just the bus. robot.connect() configures and enables torque, which is
    # unnecessary during hand calibration and can engage a previous goal.
    robot.bus.connect()
    record_ranges = robot.bus.record_ranges_of_motion
    robot.bus.record_ranges_of_motion = lambda motors: record_ranges(motors, display_values=False)
    try:
        robot.calibrate()
        print(json.dumps({name: asdict(value) for name, value in robot.calibration.items()}, indent=2), flush=True)
    finally:
        robot.bus.disconnect(disable_torque=True)


if __name__ == '__main__':
    main()
