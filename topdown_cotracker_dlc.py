#!/usr/bin/env python3
"""Top-down CoTracker pipeline for two DaVinci tools.

Workflow:
1) Select frame range.
2) Select one bounding box per tool on first frame (or load from JSON).
3) Select named keypoints per tool (or load from JSON).
4) Track box corners + keypoints via CoTracker.
5) Export DeepLabCut-compatible keypoint labels and per-tool box tracks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


@dataclass
class NamedPoint:
    name: str
    x: float
    y: float
    tool: str


def read_video_info(video_path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frame_count, width, height, fps


def load_video_range(video_path: Path, start_frame: int, end_frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: List[np.ndarray] = []
    for _ in range(end_frame - start_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames loaded from selected range")
    return np.stack(frames, axis=0)


def select_frame_range(video_path: Path, frame_count: int) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    window = "Frame range (q/Enter confirm)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Start", window, 0, frame_count - 1, lambda _: None)
    cv2.createTrackbar("End", window, frame_count - 1, frame_count - 1, lambda _: None)

    prev = (-1, -1)
    while True:
        start = cv2.getTrackbarPos("Start", window)
        end = cv2.getTrackbarPos("End", window)
        if end < start:
            end = start
            cv2.setTrackbarPos("End", window, end)

        if (start, end) != prev:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok1, f1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, end)
            ok2, f2 = cap.read()
            if ok1 and ok2:
                cv2.putText(f1, f"START {start}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(f2, f"END {end}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window, np.vstack([f1, f2]))
            prev = (start, end)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            cv2.destroyWindow(window)
            cap.release()
            return start, end
        if key == 27:
            cv2.destroyWindow(window)
            cap.release()
            raise RuntimeError("Aborted")


def select_tool_boxes(frame: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    cv2.namedWindow("Select two tool boxes", cv2.WINDOW_NORMAL)
    rois = cv2.selectROIs("Select two tool boxes", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select two tool boxes")
    if len(rois) != 2:
        raise RuntimeError("Please select exactly 2 tool boxes.")
    boxes = {
        "tool_left": tuple(float(v) for v in rois[0]),
        "tool_right": tuple(float(v) for v in rois[1]),
    }
    return boxes


def _pick_named_points_for_tool(frame: np.ndarray, tool_name: str, box: Tuple[float, float, float, float]) -> List[NamedPoint]:
    x, y, w, h = [int(v) for v in box]
    crop = frame[y : y + h, x : x + w].copy()
    points: List[NamedPoint] = []
    window = f"{tool_name}: click keypoints (q/Enter done, u undo)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def redraw() -> None:
        vis = crop.copy()
        for i, p in enumerate(points):
            cx, cy = int(p.x - x), int(p.y - y)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(vis, f"{i}:{p.name}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow(window, vis)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            gx, gy = x + mx, y + my
            name = input(f"{tool_name} point name @ ({gx}, {gy}) [blank=auto]: ").strip()
            if not name:
                name = f"{tool_name}_pt_{len(points):02d}"
            points.append(NamedPoint(name=name, x=float(gx), y=float(gy), tool=tool_name))
            redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            if not points:
                print(f"Pick at least one keypoint for {tool_name}")
                continue
            cv2.destroyWindow(window)
            return points
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Aborted")


def select_tool_keypoints(frame: np.ndarray, boxes: Dict[str, Tuple[float, float, float, float]]) -> List[NamedPoint]:
    pts: List[NamedPoint] = []
    for tool_name, box in boxes.items():
        pts.extend(_pick_named_points_for_tool(frame, tool_name, box))
    return pts


def corners_from_box(box: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = box
    return np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


def run_cotracker(frames_bgr: np.ndarray, queries_xy: np.ndarray, use_cuda: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        from cotracker.predictor import CoTrackerPredictor
    except Exception as exc:
        raise RuntimeError(
            "CoTracker import failed. Install with: pip install git+https://github.com/facebookresearch/co-tracker.git"
        ) from exc

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    video_rgb = frames_bgr[..., ::-1].copy()
    video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    q = np.concatenate([np.zeros((queries_xy.shape[0], 1), dtype=np.float32), queries_xy], axis=1)
    queries = torch.from_numpy(q).unsqueeze(0).to(device)

    model = CoTrackerPredictor().to(device)
    model.eval()
    with torch.no_grad():
        tracks, vis = model(video, queries=queries)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy(), device


def export_outputs(
    outdir: Path,
    video_path: Path,
    frame_start: int,
    point_names: List[str],
    point_tracks: np.ndarray,
    point_vis: np.ndarray,
    scorer: str,
    tool_box_tracks: Dict[str, np.ndarray],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(frame_start, frame_start + point_tracks.shape[0])

    # DLC export
    cols = []
    for bp in point_names:
        cols.extend([(scorer, bp, "x"), (scorer, bp, "y"), (scorer, bp, "likelihood")])
    midx = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])

    rows, idx = [], []
    for i, fid in enumerate(frames):
        r = []
        for j in range(len(point_names)):
            r.extend([float(point_tracks[i, j, 0]), float(point_tracks[i, j, 1]), float(point_vis[i, j])])
        rows.append(r)
        idx.append(f"labeled-data/{video_path.stem}/img{fid:06d}.png")

    df = pd.DataFrame(rows, index=idx, columns=midx)
    df.to_csv(outdir / f"{video_path.stem}_dlc_labels.csv")
    df.to_hdf(outdir / f"{video_path.stem}_dlc_labels.h5", key="df_with_missing", mode="w")

    # Tool box trajectories
    box_rows = []
    for i, fid in enumerate(frames):
        for tool, corners in tool_box_tracks.items():
            xs = corners[i, :, 0]
            ys = corners[i, :, 1]
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            box_rows.append({"frame": int(fid), "tool": tool, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    pd.DataFrame(box_rows).to_csv(outdir / f"{video_path.stem}_tool_boxes.csv", index=False)


def load_annotation_json(path: Path):
    data = json.loads(path.read_text())
    frame_start, frame_end = int(data["frame_start"]), int(data["frame_end"])
    boxes = {k: tuple(v) for k, v in data["boxes"].items()}
    points = [NamedPoint(name=p["name"], x=float(p["x"]), y=float(p["y"]), tool=p["tool"]) for p in data["points"]]
    return frame_start, frame_end, boxes, points


def save_annotation_json(path: Path, frame_start: int, frame_end: int, boxes, points: List[NamedPoint]):
    payload = {
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "boxes": boxes,
        "points": [{"name": p.name, "x": p.x, "y": p.y, "tool": p.tool} for p in points],
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Top-down CoTracker to DLC exporter")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("outputs"))
    ap.add_argument("--scorer", type=str, default="cotracker")
    ap.add_argument("--annotation-json", type=Path, default=None, help="Predefined frame range, boxes, keypoints for non-interactive runs")
    ap.add_argument("--save-annotation-json", type=Path, default=None, help="Where to save interactive selections")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    fc, w, h, fps = read_video_info(args.video)
    print(f"Video {args.video} | frames={fc} | {w}x{h} | {fps:.2f}fps")

    if args.annotation_json is not None:
        frame_start, frame_end, boxes, points = load_annotation_json(args.annotation_json)
    else:
        frame_start, frame_end = select_frame_range(args.video, fc)
        clip0 = load_video_range(args.video, frame_start, frame_start)[0]
        boxes = select_tool_boxes(clip0)
        points = select_tool_keypoints(clip0, boxes)
        if args.save_annotation_json is not None:
            save_annotation_json(args.save_annotation_json, frame_start, frame_end, boxes, points)
            print(f"Saved annotations: {args.save_annotation_json}")

    clip = load_video_range(args.video, frame_start, frame_end)

    # top-down query set: first box corners, then keypoints
    tool_corner_queries = {}
    all_queries = []
    for tool, box in boxes.items():
        c = corners_from_box(box)
        tool_corner_queries[tool] = c
        all_queries.append(c)

    keypoint_queries = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    all_queries.append(keypoint_queries)
    qxy = np.concatenate(all_queries, axis=0)

    tracks, vis, device = run_cotracker(clip, qxy, use_cuda=not args.cpu)
    print(f"CoTracker completed on {device}")

    # unpack tracks
    offset = 0
    tool_box_tracks: Dict[str, np.ndarray] = {}
    for tool in boxes:
        tool_box_tracks[tool] = tracks[:, offset : offset + 4, :]
        offset += 4

    point_tracks = tracks[:, offset:, :]
    point_vis = vis[:, offset:]
    point_names = [p.name for p in points]

    export_outputs(
        outdir=args.outdir,
        video_path=args.video,
        frame_start=frame_start,
        point_names=point_names,
        point_tracks=point_tracks,
        point_vis=point_vis,
        scorer=args.scorer,
        tool_box_tracks=tool_box_tracks,
    )

    print("Done.")


if __name__ == "__main__":
    main()
