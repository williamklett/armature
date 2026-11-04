"""Explicit, temporary camera sharing with authenticated local clients."""
import threading
import time

from .core import uid
from .grader import validate_frames


class CameraFeed:
    def __init__(self, clock=time.monotonic, wall=time.time, ttl=3):
        self.clock, self.wall, self.ttl = clock, wall, ttl
        self.lock = threading.RLock()
        self.lease = None
        self.latest = None
        self.received = 0

    def start(self):
        with self.lock:
            self.lease, self.latest = uid(), None
            return {'lease': self.lease}

    def stop(self, lease):
        with self.lock:
            if lease == self.lease:
                self.lease, self.latest = None, None
            return self.status()

    def put(self, lease, frame):
        frame = validate_frames([frame])[0]
        with self.lock:
            if not self.lease or lease != self.lease:
                raise ValueError('Camera sharing ended; start Agent view again')
            now = self.wall()
            if not now - self.ttl <= frame['captured_at'] <= now + 1:
                raise ValueError('Camera frame is stale')
            if self.latest and frame['captured_at'] <= self.latest['captured_at']:
                raise ValueError('Camera frame is out of order')
            self.received = self.clock()
            self.latest = {**frame, 'frame_id': uid()}
            return self.status()

    def status(self):
        with self.lock:
            age = max(0, self.clock() - self.received, self.wall() - self.latest['captured_at']) if self.latest else None
            fresh = bool(self.lease and self.latest and age <= self.ttl)
            return {'sharing': bool(self.lease), 'fresh': fresh, 'age_seconds': age,
                    'frame_id': self.latest['frame_id'] if fresh else None}

    def ready(self):
        return self.status()['fresh']

    def frame(self):
        with self.lock:
            status = self.status()
            if not status['fresh']:
                raise ValueError('No fresh camera frame; enable camera and Agent view')
            return {**self.latest, **status}
