"""Bounded joint-space trajectories. No device access or collision claims."""
import math

from .core import number


def plan_motion(start, waypoints, limits, speed=6, acceleration=20):
    """Quintic segments with zero endpoint velocity/acceleration.

    Duration uses the exact peak derivatives of 10u^3-15u^4+6u^5.
    Partial waypoints inherit the preceding pose, never an implicit zero.
    """
    speed = number(speed, .1, 30, 'Speed')
    acceleration = number(acceleration, .1, 100, 'Acceleration')
    if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 12:
        raise ValueError('Provide 1 to 12 waypoints')
    if set(start) != set(limits):
        raise ValueError('Starting pose must include every joint')
    current = {j: number(v, *limits[j], j) for j, v in start.items()}
    segments, elapsed = [], 0.
    for waypoint in waypoints:
        if not isinstance(waypoint, dict) or set(waypoint) - {'positions', 'duration'}:
            raise ValueError('Waypoint accepts positions and optional duration')
        positions = waypoint.get('positions')
        if not isinstance(positions, dict) or not positions or set(positions) - set(start):
            raise ValueError('Waypoint contains unknown or missing joints')
        target = dict(current)
        target.update({j: number(v, *limits[j], j) for j, v in positions.items()})
        distance = max(abs(target[j] - current[j]) for j in start)
        minimum = max(.4, 1.875 * distance / speed,
                      math.sqrt((10 / math.sqrt(3)) * distance / acceleration))
        duration = minimum if 'duration' not in waypoint else number(waypoint['duration'], minimum, 60, 'Duration')
        segments.append({'start': current, 'target': target, 'start_time': elapsed, 'duration': duration})
        elapsed += duration
        current = target
    if elapsed > 60:
        raise ValueError('Motion exceeds 60 seconds; split it into reviewed chunks')
    return {'segments': segments, 'duration': elapsed, 'speed': speed, 'acceleration': acceleration,
            'start': dict(start), 'target': current}


def sample_motion(plan, elapsed):
    for segment in plan['segments']:
        if elapsed <= segment['start_time'] + segment['duration']:
            u = max(0., min(1., (elapsed - segment['start_time']) / segment['duration']))
            blend = u ** 3 * (10 + u * (-15 + 6 * u))
            return {j: v + blend * (segment['target'][j] - v) for j, v in segment['start'].items()}
    return dict(plan['target'])
