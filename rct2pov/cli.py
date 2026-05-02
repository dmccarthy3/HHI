"""CLI entry point for rct2pov."""

import sys
import click
import cv2
import numpy as np

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from .tracker import CoasterTracker
from .pov import POVRenderer
from . import effects as fx


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_video", metavar="INPUT", type=click.Path(exists=True))
@click.argument("output_video", metavar="OUTPUT", type=click.Path())
# --- output geometry ---
@click.option("--width", default=1280, show_default=True,
              help="Output frame width.")
@click.option("--height", default=720, show_default=True,
              help="Output frame height.")
# --- POV tuning ---
@click.option("--look-ahead", "look_ahead", default=0.28, show_default=True,
              help="How far ahead to sample (fraction of min input dimension). "
                   "Increase for a more compressed perspective.")
@click.option("--horizon", default=0.52, show_default=True,
              help="Horizon vertical position in output [0=top, 1=bottom].")
# --- effect switches ---
@click.option("--motion-blur/--no-motion-blur", default=True, show_default=True)
@click.option("--distortion/--no-distortion", default=True, show_default=True,
              help="Barrel lens distortion.")
@click.option("--vignette/--no-vignette", default=True, show_default=True)
@click.option("--color-grade/--no-color-grade", default=True, show_default=True)
@click.option("--shake/--no-shake", default=True, show_default=True,
              help="Speed-proportional camera shake.")
# --- effect intensity ---
@click.option("--blur-intensity", default=0.35, show_default=True,
              help="Motion blur length multiplier (pixels per speed unit).")
@click.option("--shake-scale", default=0.25, show_default=True,
              help="Camera shake multiplier.")
# --- range ---
@click.option("--start", default=0, show_default=True,
              help="First frame to convert (0-based).")
@click.option("--end", default=-1, show_default=True,
              help="Last frame (exclusive). -1 = process entire video.")
# --- debug ---
@click.option("--debug", is_flag=True, default=False,
              help="Write a side-by-side debug video showing tracked car.")
def convert(
    input_video, output_video,
    width, height,
    look_ahead, horizon,
    motion_blur, distortion, vignette, color_grade, shake,
    blur_intensity, shake_scale,
    start, end,
    debug,
):
    """
    Convert an RCT2 isometric coaster VIDEO into a first-person POV video.

    \b
    INPUT   Path to the RCT2 gameplay screen-recording (any format OpenCV can
            read: .mp4, .avi, .mov, …).
    OUTPUT  Path for the generated POV video (.mp4 recommended).

    \b
    How it works
    ------------
    Each input frame is analysed with a background-subtraction detector to
    locate the moving roller coaster car.  A perspective homography then warps
    the trapezoidal region ahead of the car into the full output rectangle,
    simulating the rider's eye-level view.  Optional post-processing adds
    motion blur, barrel distortion, vignette, colour grading and camera shake.

    \b
    Tips
    ----
    • Use --debug to verify the car is being tracked correctly.  The debug
      video overlays the detected position and heading arrow.
    • If the car isn't detected (static-heavy scene, lots of guests moving),
      try recording with fewer decorations / guests visible on screen.
    • --look-ahead 0.20 gives a wider, more dramatic FOV;
      --look-ahead 0.35 compresses more of the track toward the horizon.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open '{input_video}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_end = total if end < 0 else min(end, total)
    frames_to_process = max(0, frame_end - start)

    click.echo(f"Input  : {input_video}")
    click.echo(f"         {in_w}×{in_h}  {fps:.2f} fps  {total} frames")
    click.echo(f"Output : {output_video}")
    click.echo(f"         {width}×{height}  frames [{start}…{frame_end}]")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise click.ClickException(f"Cannot create output file '{output_video}'.")

    debug_writer = None
    if debug:
        debug_path = _debug_path(output_video)
        dbg_w = in_w + width
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (dbg_w, max(in_h, height)))
        click.echo(f"Debug  : {debug_path}")

    # Seek to start frame.
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tracker = CoasterTracker()
    renderer = POVRenderer(
        output_size=(width, height),
        look_ahead_frac=look_ahead,
        horizon_lift=horizon,
    )
    rng = np.random.default_rng(0)

    prev_dir = 0.0

    iterator = range(frames_to_process)
    if _TQDM:
        iterator = tqdm(iterator, desc="Converting", unit="fr",
                        dynamic_ncols=True)

    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break

        pos = tracker.track(frame)
        direction = tracker.direction()
        speed = tracker.speed()

        # Turn rate: angular change since last frame (already smooth from history)
        turn_rate = direction - prev_dir
        turn_rate = (turn_rate + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
        prev_dir = direction

        pov = renderer.render(frame, pos, direction, turn_rate)

        if motion_blur:
            pov = fx.motion_blur(pov, direction, speed, intensity=blur_intensity)
        if distortion:
            pov = fx.barrel_distortion(pov)
        if color_grade:
            pov = fx.color_grade(pov)
        if vignette:
            pov = fx.vignette(pov)
        if shake:
            pov = fx.camera_shake(pov, speed * shake_scale, rng)

        writer.write(pov)

        if debug_writer is not None:
            debug_writer.write(_make_debug_frame(frame, pov, pos, direction, in_h, height))

    cap.release()
    writer.release()
    if debug_writer:
        debug_writer.release()

    click.echo("Done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _debug_path(output_path: str) -> str:
    """Derive a debug video path by inserting '_debug' before the extension."""
    if "." in output_path:
        base, ext = output_path.rsplit(".", 1)
        return f"{base}_debug.{ext}"
    return output_path + "_debug.mp4"


def _make_debug_frame(
    iso: np.ndarray,
    pov: np.ndarray,
    car_pos: tuple[float, float],
    direction: float,
    in_h: int,
    out_h: int,
) -> np.ndarray:
    """Side-by-side debug frame: isometric with overlay | POV output."""
    overlay = iso.copy()
    cx, cy = int(car_pos[0]), int(car_pos[1])
    cv2.circle(overlay, (cx, cy), 10, (0, 255, 0), 2)
    arrow_len = 40
    ex = cx + int(np.cos(direction) * arrow_len)
    ey = cy + int(np.sin(direction) * arrow_len)
    cv2.arrowedLine(overlay, (cx, cy), (ex, ey), (0, 60, 255), 2, tipLength=0.3)

    target_h = max(in_h, out_h)
    iso_resized = _pad_to_height(overlay, target_h)
    pov_resized = _pad_to_height(pov, target_h)
    return np.hstack([iso_resized, pov_resized])


def _pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    pad = target_h - h
    return np.pad(img, ((0, pad), (0, 0), (0, 0)), mode="constant")
