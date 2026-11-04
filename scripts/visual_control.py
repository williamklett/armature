"""Preview or run a bounded visual learning experiment, never a motion recipe.

A spec selects joints and image patches, calibration probes, and visual goals.
The learned controller chooses the goal-reaching actions. Raw experience stays
private. Neither preview nor server startup connects or starts motors.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.client import BridgeClient
from controller.session import VisualSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--experiment')
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    client = BridgeClient(args.port)
    session = VisualSession(client, spec['joints'], spec['tracked_box'], spec['static_box'])
    print(json.dumps({'preview': session.status(), 'experiment': spec}), flush=True)
    if not args.execute:
        return
    if not args.experiment:
        parser.error('--execute requires --experiment')
    try:
        client.action('start', experiment_id=args.experiment)
        for delta in spec.get('probes', []):
            result = session.step(delta, speed=spec.get('speed', 3))
            print(json.dumps({'probe': result}), flush=True)
        for goal in spec.get('goals', []):
            # Each completed goal latches a hold. A new goal is explicitly
            # included in this experiment spec, not an automatic fault retry.
            state = client.observe()
            if state['stopped'] and not state['fault']:
                client.action('start', experiment_id=args.experiment)
            result = session.reach(goal, max_steps=spec.get('max_steps', 6),
                                   tolerance=spec.get('tolerance_px', 2), speed=spec.get('speed', 3))
            print(json.dumps({'evaluation': result}), flush=True)
            if not result['reached']:
                break
    finally:
        client.action('stop')


if __name__ == '__main__':
    main()
