"""
POV renderer: converts a single isometric frame to a first-person perspective
frame by computing a perspective homography from a trapezoid centred on the
coaster car (pointing in its direction of travel) to the full output rectangle.

Isometric view geometry
-----------------------
RCT2 uses a 2:1 isometric projection (tiles are 32 px wide × 16 px tall).
The "forward" direction for the rider is simply the direction the car moves
across the screen, so we do not need to un-project into world-space — we just
need to look *ahead* of the car in screen-space along its velocity vector.

Homography trapezoid
--------------------
We define a quadrilateral in the source isometric frame:
  - Bottom edge: wide, just in front of the car (simulates the near ground)
  - Top edge:    narrow, far ahead of the car  (simulates the horizon)
This quad is then warped to fill the output rectangle, giving the classic
forced-perspective / POV look.

Camera banking
--------------
When the car is turning, we apply a subtle roll to the output frame that
matches the direction and magnitude of the turn.
"""

import cv2
import numpy as np


class POVRenderer:
    def __init__(
        self,
        output_size: tuple[int, int] = (1280, 720),
        look_ahead_frac: float = 0.28,
        near_frac: float = 0.055,
        near_half_w_frac: float = 0.20,
        far_half_w_frac: float = 0.040,
        max_roll_deg: float = 18.0,
        roll_smoothing: float = 0.88,
        horizon_lift: float = 0.52,
    ):
        """
        Parameters
        ----------
        output_size       : (width, height) of the rendered POV frame.
        look_ahead_frac   : How far ahead to sample, as a fraction of the
                            shorter input dimension.  Larger → more of the
                            scene is compressed into the horizon.
        near_frac         : How close the "feet" edge is to the car centre.
        near_half_w_frac  : Half-width of the near edge (trapezoid base).
        far_half_w_frac   : Half-width of the far edge (trapezoid top / horizon).
        max_roll_deg      : Maximum camera roll induced by turning.
        roll_smoothing    : Low-pass coefficient for roll (0 = instant, 1 = frozen).
        horizon_lift      : Vertical position of the horizon in output frame [0..1],
                            where 0 = top, 1 = bottom.  0.5 = centre.
        """
        self.out_w, self.out_h = output_size
        self.look_ahead_frac = look_ahead_frac
        self.near_frac = near_frac
        self.near_half_w_frac = near_half_w_frac
        self.far_half_w_frac = far_half_w_frac
        self.max_roll_deg = max_roll_deg
        self.roll_smoothing = roll_smoothing
        self.horizon_lift = horizon_lift

        self._smooth_roll = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        car_pos: tuple[float, float],
        direction: float,
        turn_rate: float = 0.0,
    ) -> np.ndarray:
        """
        Warp *frame* into a POV perspective centred on *car_pos* facing
        *direction* (radians, screen-space).

        Parameters
        ----------
        frame      : Input BGR isometric frame.
        car_pos    : (x, y) of the coaster car in *frame* pixel coordinates.
        direction  : Heading in radians (0 = right, π/2 = down).
        turn_rate  : Signed angular velocity (rad/frame); drives camera roll.
        """
        H = self._homography(frame.shape, car_pos, direction)

        pov = cv2.warpPerspective(
            frame,
            H,
            (self.out_w, self.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        pov = self._apply_roll(pov, turn_rate)
        return pov

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _homography(
        self,
        frame_shape: tuple,
        car_pos: tuple[float, float],
        direction: float,
    ) -> np.ndarray:
        fh, fw = frame_shape[:2]
        base = min(fw, fh)

        near_d = base * self.near_frac
        far_d = base * self.look_ahead_frac
        near_hw = base * self.near_half_w_frac
        far_hw = base * self.far_half_w_frac

        cx, cy = car_pos
        fwd = np.array([np.cos(direction), np.sin(direction)])
        rgt = np.array([-fwd[1], fwd[0]])

        def pt(dist: float, side: float) -> list[float]:
            p = np.array([cx, cy]) + fwd * dist + rgt * side
            return p.tolist()

        # Source trapezoid in the isometric frame (clockwise from bottom-left)
        src = np.float32([
            pt(near_d, -near_hw),  # bottom-left  (near, left)
            pt(near_d,  near_hw),  # bottom-right (near, right)
            pt(far_d,   far_hw),   # top-right    (far,  right)
            pt(far_d,  -far_hw),   # top-left     (far,  left)
        ])

        # Destination rectangle — horizon_lift controls where the horizon sits
        horizon_y = int(self.out_h * (1.0 - self.horizon_lift))
        dst = np.float32([
            [0,          self.out_h],   # bottom-left
            [self.out_w, self.out_h],   # bottom-right
            [self.out_w, horizon_y],    # top-right
            [0,          horizon_y],    # top-left
        ])

        return cv2.getPerspectiveTransform(src, dst)

    def _apply_roll(self, frame: np.ndarray, turn_rate: float) -> np.ndarray:
        """Low-pass filtered camera roll proportional to turn rate."""
        target = float(np.clip(np.degrees(turn_rate) * 6.0,
                               -self.max_roll_deg, self.max_roll_deg))
        self._smooth_roll = (
            self._smooth_roll * self.roll_smoothing
            + target * (1.0 - self.roll_smoothing)
        )

        if abs(self._smooth_roll) < 0.15:
            return frame

        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), self._smooth_roll, 1.0)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
