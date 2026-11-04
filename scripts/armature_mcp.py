"""Native image + action tools. No motor connection is made at server startup."""
import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from controller.client import BridgeClient
from bridge.core import number


def build_server(client):
    server = MCPServer('armature', instructions=(
        'Observe the camera and arm before physical action. Plan/execute accepts arbitrary joint targets; '
        'visual control learns local action-image effects from probes. Start sessions explicitly. '
        'Never infer grasp success from 2D alignment. Stop on uncertainty; faults require local inspection. '
        'These tools do not detect people, obstacles, or depth.'))
    visual = None
    movement = threading.Lock()
    read = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
    local_write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)

    @contextmanager
    def exclusive():
        if not movement.acquire(blocking=False):
            raise ToolError('Another controller operation is active; observe or stop it first')
        try:
            yield
        except (ValueError, RuntimeError) as error:
            raise ToolError(str(error)) from None
        finally:
            movement.release()

    def observation(extra=None, after=0):
        try:
            snapshot = client.snapshot(after=after)
        except (ValueError, RuntimeError) as error:
            raise ToolError(str(error)) from None
        frame = dict(snapshot['frame'])
        url = frame.pop('data_url')
        metadata = {**snapshot, 'frame': frame, **(extra or {})}
        return CallToolResult(content=[TextContent(text=json.dumps(metadata, allow_nan=False)),
            ImageContent(data=url.split(',', 1)[1], mimeType=url.split(';', 1)[0].split(':')[1])])

    @server.tool(annotations=read)
    def arm_observe() -> CallToolResult:
        """Current laptop camera image, joint positions, motion and health timestamps."""
        return observation()

    @server.tool(annotations=write)
    def arm_start(experiment_id: str) -> dict:
        """Start a connected healthy arm session. Does not connect motors or clear faults."""
        with exclusive():
            return client.action('start', experiment_id=experiment_id)

    @server.tool(annotations=local_write)
    def arm_plan(positions: dict[str, float], speed: float = 3) -> dict:
        """Preview an arbitrary coordinated joint target without moving. Degrees; gripper 0-100."""
        with exclusive():
            return client.action('plan', waypoints=[{'positions': positions}], speed=speed)

    @server.tool(annotations=write)
    def arm_execute(plan_id: str) -> dict:
        """Execute a reviewed, single-use, unexpired plan asynchronously. Observe to monitor."""
        with exclusive():
            return client.action('execute', plan_id=plan_id)

    @server.tool(annotations=write)
    def arm_steer(experiment_id: str, deltas: dict[str, float], speed: float = 4) -> CallToolResult:
        """Steer from the camera: a short relative joint input, then a NEW camera image.

        Body deltas are degrees (max 10); gripper deltas are percentage points
        (max 20). Explicitly starts a healthy session for this requested action.
        Does not connect motors, clear faults, retry motion, or claim a grasp.
        Observe the returned image before choosing the next input.
        """
        with exclusive():
            before = client.snapshot()
            state = before['arm']
            if not state['connected'] or state['fault']:
                raise ValueError('Connect and inspect the arm before steering')
            if state.get('motion') and state['motion']['state'] == 'running':
                raise ValueError('Wait for the current motion or stop it first')
            if not deltas or set(deltas) - set(state['positions']):
                raise ValueError('Specify known joint deltas')
            speed = number(speed, 1, 6, 'Steering speed')
            target = {}
            for joint, delta in deltas.items():
                bound = 20 if joint == 'gripper' else 10
                delta = number(delta, -bound, bound, 'Steering delta')
                target[joint] = number(state['positions'][joint] + delta, *state['limits'][joint], joint)
            client.action('start', experiment_id=experiment_id)
            started = time.monotonic()
            client.run(target, speed=speed)
            return observation({'steering': {'deltas': deltas, 'target': target,
                'before_frame_id': before['frame']['frame_id'],
                'motion_seconds': time.monotonic() - started}}, after=time.time())

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
    def arm_stop() -> dict:
        """Preempt movement and request a current-pose hold; torque remains enabled."""
        return client.action('stop')

    @server.tool(annotations=local_write)
    def visual_initialize(joints: list[str], tracked_box: list[int], static_box: list[int]) -> dict:
        """Initialize a local learned controller. Boxes are [x,y,width,height] in camera pixels.

        Tracked box must lie on one rigid moving gripper patch. Static box must
        contain textured stationary scenery, not a screen/person/arm. Holds
        other joints fixed, bounds selected joints to ±6 degrees from this pose.
        Writes private observations, but commands no physical motion.
        """
        nonlocal visual
        from controller.session import VisualSession
        with exclusive():
            visual = VisualSession(client, joints, tracked_box, static_box)
            return visual.status()

    @server.tool(annotations=write)
    def visual_probe(delta: list[float], speed: float = 3) -> CallToolResult:
        """Execute a ≤2 degree probe in selected-joint order; learn from measured visual change.

        Requires explicit arm_start. Tracker loss stops motion. Obtain at least
        two independent probes per selected joint before using visual_reach.
        """
        if visual is None:
            raise ToolError('Initialize visual control first')
        with exclusive():
            result = visual.step(delta, speed)
            return observation({'transition': result})

    @server.tool(annotations=write)
    def visual_reach(goal_xy: list[float], max_steps: int = 6, tolerance_px: float = 2, speed: float = 3) -> CallToolResult:
        """Run bounded learned visual feedback toward a nearby pixel goal, then hold.

        Searches continuous actions, measures outcome, updates the predictor and
        replans locally without LLM round trips. 2D reaching is not a grasp/depth check.
        """
        if visual is None:
            raise ToolError('Initialize visual control and collect independent probes first')
        with exclusive():
            result = visual.reach(goal_xy, max_steps, tolerance_px, speed)
            return observation({'evaluation': result})

    @server.tool(annotations=read)
    def visual_status() -> dict:
        """Learned model status and private experience directory; no movement."""
        return visual.status() if visual else {'initialized': False}

    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8787)
    args = parser.parse_args()
    build_server(BridgeClient(args.port)).run(transport='stdio')
