"""
Post-processing effects applied to each POV frame.

All functions accept and return uint8 BGR numpy arrays of identical shape.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Motion blur
# ---------------------------------------------------------------------------

def motion_blur(
    frame: np.ndarray,
    direction: float,
    speed: float,
    intensity: float = 0.35,
    max_len: int = 31,
) -> np.ndarray:
    """Directional motion blur proportional to coaster speed."""
    blur_len = int(min(speed * intensity, max_len))
    if blur_len < 3:
        return frame
    if blur_len % 2 == 0:
        blur_len += 1

    # Build a 1-D directional kernel oriented along *direction*.
    kernel = np.zeros((blur_len, blur_len), dtype=np.float32)
    c = blur_len // 2
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    for i in range(blur_len):
        t = i - c
        kx = int(round(c + cos_d * t))
        ky = int(round(c + sin_d * t))
        if 0 <= kx < blur_len and 0 <= ky < blur_len:
            kernel[ky, kx] = 1.0

    s = kernel.sum()
    if s == 0:
        return frame
    kernel /= s

    return cv2.filter2D(frame, -1, kernel)


# ---------------------------------------------------------------------------
# Lens distortion
# ---------------------------------------------------------------------------

def barrel_distortion(
    frame: np.ndarray,
    k1: float = -0.25,
    k2: float = 0.08,
) -> np.ndarray:
    """Simulate barrel lens distortion for a GoPro/action-cam feel."""
    h, w = frame.shape[:2]
    K = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32
    )
    dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float32)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist, None, K, (w, h), cv2.CV_32FC1
    )
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

def vignette(frame: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Dark-corner vignette mask."""
    h, w = frame.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip(1.0 - dist * strength, 0.0, 1.0)[..., np.newaxis]
    return (frame.astype(np.float32) * mask).astype(np.uint8)


# ---------------------------------------------------------------------------
# Colour grade
# ---------------------------------------------------------------------------

def color_grade(
    frame: np.ndarray,
    saturation: float = 1.25,
    contrast: float = 1.08,
    brightness: float = 0.0,
) -> np.ndarray:
    """Boost saturation and contrast for a cinematic look."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    mean = 127.0
    out = np.clip((out - mean) * contrast + mean + brightness, 0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Camera shake
# ---------------------------------------------------------------------------

def camera_shake(
    frame: np.ndarray,
    magnitude: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Subtle random translation to simulate vibration on rough track."""
    if magnitude < 0.5:
        return frame
    if rng is None:
        rng = np.random.default_rng()

    mag = int(np.ceil(magnitude))
    dx = int(rng.integers(-mag, mag + 1))
    dy = int(rng.integers(-mag, mag + 1))

    h, w = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
