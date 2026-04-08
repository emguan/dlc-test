#!/usr/bin/env python3
"""
Interactive CoTracker workflow for manual frame-range and keypoint selection,
with DeepLabCut-compatible export.

Usage:
  python cotracker_dlc_tool.py --video input.mp4 --outdir outputs --scorer surgeon1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


@dataclass
class NamedPoint:
    name: str
    x: float
    y: float


def _read_video_info(video_path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frame_count <= 0:
        raise RuntimeError("Video appears to have no frames.")

    return frame_count, width, height, fps


def _load_video_range(video_path: Path, start_frame: int, end_frame: int) -> np.ndarray:
    """Returns frames as uint8 numpy array [T, H, W, C(BGR)]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    total = end_frame - start_frame + 1

    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames loaded from requested range.")

    return np.stack(frames, axis=0)


def select_frame_range(video_path: Path, frame_count: int) -> Tuple[int, int]:
    """
    Interactive frame-range picker.

    Controls:
      - Use trackbars for Start and End.
      - Press q / Enter to confirm.
      - Press Esc to abort.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    window = "Select Tracking Range (q/Enter=confirm, Esc=abort)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    cv2.createTrackbar("Start", window, 0, frame_count - 1, lambda _: None)
    cv2.createTrackbar("End", window, frame_count - 1, frame_count - 1, lambda _: None)

    last_start, last_end = -1, -1

    while True:
        start = cv2.getTrackbarPos("Start", window)
        end = cv2.getTrackbarPos("End", window)

        if start != last_start or end != last_end:
            if end < start:
                end = start
                cv2.setTrackbarPos("End", window, end)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok_s, frame_s = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, end)
            ok_e, frame_e = cap.read()

            if ok_s and ok_e:
                top = frame_s.copy()
                bot = frame_e.copy()
                cv2.putText(top, f"START frame {start}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(bot, f"END frame {end}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                vis = np.vstack([top, bot])
                cv2.imshow(window, vis)

            last_start, last_end = start, end

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):  # q or Enter
            cv2.destroyWindow(window)
            cap.release()
            return start, end
        if key == 27:  # Esc
            cv2.destroyWindow(window)
            cap.release()
            raise RuntimeError("Frame-range selection aborted by user.")


def select_named_keypoints(frame_bgr: np.ndarray) -> List[NamedPoint]:
    """
    Click points, then type point name in terminal.

    Controls:
      - Left click: add point, then enter name in terminal prompt.
      - u: undo last point
      - q / Enter: confirm
      - Esc: abort
    """
    points: List[NamedPoint] = []
    display = frame_bgr.copy()
    window = "Select Keypoints (click, name in terminal; u=undo; q/Enter=confirm)"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    def redraw() -> None:
        nonlocal display
        display = frame_bgr.copy()
        for idx, p in enumerate(points):
            cv2.circle(display, (int(p.x), int(p.y)), 5, (255, 255, 0), -1)
            cv2.putText(
                display,
                f"{idx}: {p.name}",
                (int(p.x) + 8, int(p.y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        cv2.imshow(window, display)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            candidate = frame_bgr.copy()
            cv2.circle(candidate, (x, y), 6, (0, 255, 255), -1)
            cv2.imshow(window, candidate)
            name = input(f"Name for point at ({x}, {y}) [blank=auto]: ").strip()
            if not name:
                name = f"point_{len(points):02d}"
            points.append(NamedPoint(name=name, x=float(x), y=float(y)))
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            if not points:
                print("Please select at least one keypoint.")
                continue
            cv2.destroyWindow(window)
            return points
        if key == ord("u"):
            if points:
                points.pop()
                redraw()
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Keypoint selection aborted by user.")


def _run_cotracker(frames_bgr: np.ndarray, points: List[NamedPoint], use_cuda: bool):
    """Run CoTracker and return tracks [T, N, 2], visibility [T, N]."""
    try:
        from cotracker.predictor import CoTrackerPredictor
    except Exception as exc:
        raise RuntimeError(
            "Could not import CoTracker. Install it first, e.g.\n"
            "  pip install git+https://github.com/facebookresearch/co-tracker.git"
        ) from exc

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    # Convert [T,H,W,C(BGR)] -> [1,T,3,H,W] RGB float32
    video_rgb = frames_bgr[..., ::-1].copy()
    video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)

    # query format: [batch, N, 3] where each row is [t, x, y]
    q = np.array([[0.0, p.x, p.y] for p in points], dtype=np.float32)
    queries = torch.from_numpy(q).unsqueeze(0).to(device)

    model = CoTrackerPredictor().to(device)
    model.eval()

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries)

    tracks = pred_tracks[0].detach().cpu().numpy()  # [T,N,2]
    visibility = pred_visibility[0].detach().cpu().numpy()  # [T,N]
    return tracks, visibility, device


def export_for_deeplabcut(
    outdir: Path,
    video_path: Path,
    frame_start: int,
    frame_end: int,
    point_names: List[str],
    tracks: np.ndarray,
    visibility: np.ndarray,
    scorer: str,
) -> Tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    frames = np.arange(frame_start, frame_start + tracks.shape[0])

    tuples = []
    for bp in point_names:
        tuples.extend([(scorer, bp, "x"), (scorer, bp, "y"), (scorer, bp, "likelihood")])
    columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodyparts", "coords"])

    rows = []
    index = []
    for i, frame_id in enumerate(frames):
        row = []
        for j in range(len(point_names)):
            row.append(float(tracks[i, j, 0]))
            row.append(float(tracks[i, j, 1]))
            row.append(float(visibility[i, j]))
        rows.append(row)
        index.append(f"labeled-data/{video_path.stem}/img{frame_id:06d}.png")

    df = pd.DataFrame(rows, index=index, columns=columns)

    csv_path = outdir / f"{video_path.stem}_dlc_labels.csv"
    h5_path = outdir / f"{video_path.stem}_dlc_labels.h5"
    meta_path = outdir / f"{video_path.stem}_tracking_metadata.json"

    df.to_csv(csv_path)
    df.to_hdf(h5_path, key="df_with_missing", mode="w")

    metadata = {
        "video": str(video_path),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "num_frames": int(tracks.shape[0]),
        "num_keypoints": int(len(point_names)),
        "keypoints": point_names,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    return csv_path, h5_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CoTracker to DeepLabCut exporter")
    parser.add_argument("--video", type=Path, required=True, help="Path to endoscope video")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--scorer", type=str, default="cotracker", help="DLC scorer name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    frame_count, width, height, fps = _read_video_info(args.video)
    print(f"Video: {args.video} | frames={frame_count} | size={width}x{height} | fps={fps:.2f}")

    start, end = select_frame_range(args.video, frame_count)
    print(f"Selected range: {start} -> {end}")

    clip = _load_video_range(args.video, start, end)
    points = select_named_keypoints(clip[0])
    print("Keypoints:")
    for p in points:
        print(f"  - {p.name}: ({p.x:.1f}, {p.y:.1f})")

    tracks, visibility, device = _run_cotracker(clip, points, use_cuda=not args.cpu)
    print(f"CoTracker done on {device}. Output tracks shape: {tracks.shape}")

    csv_path, h5_path, meta_path = export_for_deeplabcut(
        outdir=args.outdir,
        video_path=args.video,
        frame_start=start,
        frame_end=end,
        point_names=[p.name for p in points],
        tracks=tracks,
        visibility=visibility,
        scorer=args.scorer,
    )

    print("\nExport complete:")
    print(f"  CSV: {csv_path}")
    print(f"  H5 : {h5_path}")
    print(f"  META: {meta_path}")


if __name__ == "__main__":
    main()
