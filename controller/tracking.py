"""Local visual feature tracking with explicit loss/camera-motion failures."""
import base64
import cv2
import numpy as np


def decode(frame):
    image = cv2.imdecode(np.frombuffer(base64.b64decode(frame['data_url'].split(',', 1)[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Invalid camera image')
    return image


def corners(image, box):
    x, y, w, h = [int(v) for v in box]
    if w < 8 or h < 8 or x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError('Tracking box outside image or too small')
    mask = np.zeros_like(image)
    mask[y:y+h, x:x+w] = 255
    points = cv2.goodFeaturesToTrack(image, 50, .015, 3, mask=mask, blockSize=3)
    if points is None or len(points) < 6:
        raise ValueError('Select a textured patch with at least six corners')
    return points


def flow(previous, current, points):
    new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3)
    if new is None:
        raise ValueError('Visual tracking lost')
    back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        raise ValueError('Visual tracking lost')
    delta = (new - points).reshape(-1, 2)
    good = status.ravel().astype(bool) & reverse_status.ravel().astype(bool)
    good &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.
    good &= np.linalg.norm(delta, axis=1) < 30
    if np.count_nonzero(good) < max(5, len(points) * .6):
        raise ValueError('Visual tracking confidence lost')
    shift = np.median(delta[good], axis=0)
    good &= np.linalg.norm(delta - shift, axis=1) < 2
    if np.count_nonzero(good) < max(5, len(points) * .6):
        raise ValueError('Tracked patch is not moving coherently')
    return new[good], np.median(delta[good], axis=0)


class PointTracker:
    def __init__(self, frame, box, anchor_box):
        self.previous = decode(frame)
        self.points = corners(self.previous, box)
        self.anchors = corners(self.previous, anchor_box)
        self.position = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], dtype=float)
        self.anchor_shift = np.zeros(2)
        self.frame_id = frame['frame_id']
        self.captured_at = frame['captured_at']

    def update(self, frame):
        if frame['frame_id'] == self.frame_id:
            return self.position.copy()
        if frame['captured_at'] <= self.captured_at:
            raise ValueError('Out-of-order camera observation')
        current = decode(frame)
        if current.shape != self.previous.shape:
            raise ValueError('Camera geometry changed')
        anchors, background_motion = flow(self.previous, current, self.anchors)
        cumulative = self.anchor_shift + background_motion
        if np.linalg.norm(cumulative) > 1.5:
            raise ValueError('Camera or reference surface moved; reinitialize visual control')
        points, displacement = flow(self.previous, current, self.points)
        self.points, self.anchors, self.previous = points, anchors, current
        self.anchor_shift = cumulative
        self.position += displacement - background_motion
        self.frame_id, self.captured_at = frame['frame_id'], frame['captured_at']
        return self.position.copy()
