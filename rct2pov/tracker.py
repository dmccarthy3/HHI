"""Tracks the roller coaster car through isometric video frames."""

import cv2
import numpy as np


class CoasterTracker:
    """
    Detects and tracks the moving roller coaster car using background
    subtraction and a Kalman filter for smooth, continuous position estimates.
    """

    # Frames used purely to build the background model before tracking begins.
    WARMUP_FRAMES = 20

    def __init__(self, min_area: int = 40, max_area: int = 8000):
        self.min_area = min_area
        self.max_area = max_area

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=40, varThreshold=35, detectShadows=False
        )
        self._kalman = self._make_kalman()
        self._frame_idx = 0
        self._last_pos: tuple[float, float] = (0.0, 0.0)

        # Ring buffer of recent (x, y) positions for velocity estimation.
        self._history: list[tuple[float, float]] = []
        self._history_size = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(self, frame: np.ndarray) -> tuple[float, float]:
        """Return the smoothed (x, y) car position in this frame."""
        self._frame_idx += 1

        raw = self._detect(frame)

        predicted = self._kalman.predict()

        if raw is not None and self._frame_idx > self.WARMUP_FRAMES:
            m = np.array([[np.float32(raw[0])], [np.float32(raw[1])]])
            corrected = self._kalman.correct(m)
            pos = (float(corrected[0, 0]), float(corrected[1, 0]))
        else:
            pos = (float(predicted[0, 0]), float(predicted[1, 0]))

        self._last_pos = pos
        self._history.append(pos)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        return pos

    def direction(self, window: int = 7) -> float:
        """Return the current heading in radians (0 = right, π/2 = down)."""
        if len(self._history) < 2:
            return 0.0
        n = min(window, len(self._history))
        dx = self._history[-1][0] - self._history[-n][0]
        dy = self._history[-1][1] - self._history[-n][1]
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            # No net movement in this window — scan all history for last known direction.
            for k in range(len(self._history) - 1, 0, -1):
                ddx = self._history[k][0] - self._history[k - 1][0]
                ddy = self._history[k][1] - self._history[k - 1][1]
                if abs(ddx) + abs(ddy) > 1e-3:
                    return float(np.arctan2(ddy, ddx))
            return 0.0
        return float(np.arctan2(dy, dx))

    def speed(self, window: int = 5) -> float:
        """Return speed in pixels per frame (averaged over recent window)."""
        if len(self._history) < 2:
            return 0.0
        n = min(window, len(self._history))
        dx = self._history[-1][0] - self._history[-n][0]
        dy = self._history[-1][1] - self._history[-n][1]
        return float(np.sqrt(dx ** 2 + dy ** 2) / (n - 1))

    def turn_rate(self, window: int = 5) -> float:
        """Return signed angular velocity (radians per frame)."""
        if len(self._history) < window + 2:
            return 0.0
        mid = len(self._history) - window // 2
        dir_now = self.direction(window)
        # Direction a short while ago
        old_history = self._history[: mid]
        if len(old_history) < 2:
            return 0.0
        dx = old_history[-1][0] - old_history[0][0]
        dy = old_history[-1][1] - old_history[0][1]
        dir_old = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-3 else dir_now

        delta = dir_now - dir_old
        # Wrap to [-π, π]
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        return float(delta / max(window // 2, 1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Apply background subtraction and return largest moving blob centroid."""
        fgmask = self._bg.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        candidates = [
            c for c in contours
            if self.min_area <= cv2.contourArea(c) <= self.max_area
        ]
        if not candidates:
            return None

        largest = max(candidates, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"], M["m01"] / M["m00"]

    @staticmethod
    def _make_kalman() -> cv2.KalmanFilter:
        # State: [x, y, vx, vy]; Measurement: [x, y]
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.04
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        # Seed state at centre — will be corrected on first measurement.
        kf.statePost = np.zeros((4, 1), dtype=np.float32)
        return kf
