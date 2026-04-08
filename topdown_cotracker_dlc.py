#!/usr/bin/env python3
"""Top-down CoTracker pipeline with split local-annotation and remote-GPU tracking modes."""

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
        raise RuntimeError(f"Cannot open video: {video_path}")
    data = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return data


def load_video_range(video_path: Path, start: int, end: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: List[np.ndarray] = []
    for _ in range(end - start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames loaded for selected range")
    return np.stack(frames, axis=0)


def select_frame_range(video_path: Path, frame_count: int) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    window = "Top-down: select frame range (q/Enter confirm)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Start", window, 0, frame_count - 1, lambda _: None)
    cv2.createTrackbar("End", window, frame_count - 1, frame_count - 1, lambda _: None)

    last = (-1, -1)
    while True:
        start = cv2.getTrackbarPos("Start", window)
        end = cv2.getTrackbarPos("End", window)
        if end < start:
            end = start
            cv2.setTrackbarPos("End", window, end)

        if (start, end) != last:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok1, f1 = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, end)
            ok2, f2 = cap.read()
            if ok1 and ok2:
                cv2.putText(f1, f"START {start}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(f2, f"END {end}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window, np.vstack([f1, f2]))
            last = (start, end)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            cv2.destroyWindow(window)
            cap.release()
            return start, end
        if key == 27:
            cv2.destroyWindow(window)
            cap.release()
            raise RuntimeError("Aborted by user")


def _draw_boxes(frame: np.ndarray, boxes: Dict[str, Tuple[float, float, float, float]]) -> np.ndarray:
    vis = frame.copy()
    colors = {"tool_left": (0, 255, 0), "tool_right": (0, 0, 255)}
    for tool, (x, y, w, h) in boxes.items():
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        color = colors.get(tool, (255, 255, 0))
        cv2.rectangle(vis, p1, p2, color, 2)
        cv2.putText(vis, tool, (int(x), max(25, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def select_tool_boxes(frame: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Select two boxes and explicitly confirm them.
    This avoids losing the selection if users miss selectROI key sequence.
    """
    while True:
        cv2.namedWindow("Select 2 tool boxes", cv2.WINDOW_NORMAL)
        rois = cv2.selectROIs("Select 2 tool boxes", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select 2 tool boxes")

        if len(rois) != 2:
            print(f"Expected exactly 2 boxes, got {len(rois)}. Please reselect.")
            continue

        boxes = {
            "tool_left": tuple(float(v) for v in rois[0]),
            "tool_right": tuple(float(v) for v in rois[1]),
        }
        preview = _draw_boxes(frame, boxes)
        win = "Confirm boxes: c=confirm, r=reselect"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win)
        if key == ord("c"):
            return boxes
        print("Reselecting boxes...")


def _pick_points_tool(frame: np.ndarray, tool: str, box: Tuple[float, float, float, float], all_boxes: Dict[str, Tuple[float, float, float, float]]) -> List[NamedPoint]:
    x, y, w, h = [int(v) for v in box]
    points: List[NamedPoint] = []
    win = f"{tool}: click points inside highlighted box (q/Enter done, u undo)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def redraw() -> None:
        vis = _draw_boxes(frame, all_boxes)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 3)
        for i, p in enumerate(points):
            cx, cy = int(p.x), int(p.y)
            cv2.circle(vis, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(vis, f"{i}:{p.name}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow(win, vis)

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            gx, gy = mx, my
            if not (x <= gx <= x + w and y <= gy <= y + h):
                print(f"{tool}: click must be inside its box.")
                return
            name = input(f"{tool} point at ({gx},{gy}) name [blank=auto]: ").strip()
            if not name:
                name = f"{tool}_pt_{len(points):02d}"
            points.append(NamedPoint(name=name, x=float(gx), y=float(gy), tool=tool))
            redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            if not points:
                print(f"Need at least one point for {tool}")
                continue
            cv2.destroyWindow(win)
            return points
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Aborted by user")


def select_points(frame: np.ndarray, boxes: Dict[str, Tuple[float, float, float, float]]) -> List[NamedPoint]:
    pts: List[NamedPoint] = []
    for tool, box in boxes.items():
        pts.extend(_pick_points_tool(frame, tool, box, boxes))
    return pts


def save_annotations(path: Path, video: Path, frame_start: int, frame_end: int, boxes, points: List[NamedPoint]) -> None:
    payload = {
        "video": str(video),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "boxes": boxes,
        "points": [{"name": p.name, "x": p.x, "y": p.y, "tool": p.tool} for p in points],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_annotations(path: Path):
    data = json.loads(path.read_text())
    boxes = {k: tuple(v) for k, v in data["boxes"].items()}
    points = [NamedPoint(name=p["name"], x=float(p["x"]), y=float(p["y"]), tool=p["tool"]) for p in data["points"]]
    return Path(data["video"]), int(data["frame_start"]), int(data["frame_end"]), boxes, points


def corners_from_box(box: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def run_cotracker(clip_bgr: np.ndarray, qxy: np.ndarray, force_cpu: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    from cotracker.predictor import CoTrackerPredictor

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    video_rgb = clip_bgr[..., ::-1].copy()
    video = torch.from_numpy(video_rgb).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    q = np.concatenate([np.zeros((qxy.shape[0], 1), dtype=np.float32), qxy], axis=1)
    queries = torch.from_numpy(q).unsqueeze(0).to(device)

    model = CoTrackerPredictor().to(device)
    model.eval()
    with torch.no_grad():
        tracks, vis = model(video, queries=queries)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy(), device


def export_dlc_and_boxes(outdir: Path, video: Path, frame_start: int, point_names: List[str], point_tracks: np.ndarray, point_vis: np.ndarray, scorer: str, box_tracks: Dict[str, np.ndarray]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(frame_start, frame_start + point_tracks.shape[0])

    columns = []
    for name in point_names:
        columns.extend([(scorer, name, "x"), (scorer, name, "y"), (scorer, name, "likelihood")])
    columns = pd.MultiIndex.from_tuples(columns, names=["scorer", "bodyparts", "coords"])

    rows, index = [], []
    for i, fid in enumerate(frames):
        row = []
        for j in range(len(point_names)):
            row.extend([float(point_tracks[i, j, 0]), float(point_tracks[i, j, 1]), float(point_vis[i, j])])
        rows.append(row)
        index.append(f"labeled-data/{video.stem}/img{fid:06d}.png")

    df = pd.DataFrame(rows, index=index, columns=columns)
    df.to_csv(outdir / f"{video.stem}_dlc_labels.csv")
    df.to_hdf(outdir / f"{video.stem}_dlc_labels.h5", key="df_with_missing", mode="w")

    box_rows = []
    for i, fid in enumerate(frames):
        for tool, corners in box_tracks.items():
            xs, ys = corners[i, :, 0], corners[i, :, 1]
            box_rows.append({"frame": int(fid), "tool": tool, "x1": float(xs.min()), "y1": float(ys.min()), "x2": float(xs.max()), "y2": float(ys.max())})
    pd.DataFrame(box_rows).to_csv(outdir / f"{video.stem}_tool_boxes.csv", index=False)


def cmd_annotate(args) -> None:
    fc, w, h, fps = read_video_info(args.video)
    print(f"Video: {args.video} | frames={fc} | size={w}x{h} | fps={fps:.2f}")
    s, e = select_frame_range(args.video, fc)
    first = load_video_range(args.video, s, s)[0]
    boxes = select_tool_boxes(first)
    points = select_points(first, boxes)
    save_annotations(args.annotations_out, args.video, s, e, boxes, points)
    print(f"Saved annotation bundle: {args.annotations_out}")


def cmd_track(args) -> None:
    video, s, e, boxes, points = load_annotations(args.annotations_json)
    if args.video is not None:
        video = args.video
    clip = load_video_range(video, s, e)

    query_chunks = []
    for _, box in boxes.items():
        query_chunks.append(corners_from_box(box))
    query_chunks.append(np.array([[p.x, p.y] for p in points], dtype=np.float32))
    qxy = np.concatenate(query_chunks, axis=0)

    tracks, vis, device = run_cotracker(clip, qxy, force_cpu=args.cpu)
    print(f"Tracking complete on: {device}")

    offset = 0
    box_tracks: Dict[str, np.ndarray] = {}
    for tool in boxes:
        box_tracks[tool] = tracks[:, offset : offset + 4, :]
        offset += 4
    point_tracks, point_vis = tracks[:, offset:, :], vis[:, offset:]

    export_dlc_and_boxes(
        args.outdir,
        video,
        s,
        [p.name for p in points],
        point_tracks,
        point_vis,
        args.scorer,
        box_tracks,
    )
    print(f"Outputs written to: {args.outdir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top-down CoTracker with local annotate + remote GPU tracking")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ann = sub.add_parser("annotate", help="Local interactive GUI annotation only")
    p_ann.add_argument("--video", type=Path, required=True)
    p_ann.add_argument("--annotations-out", type=Path, required=True)
    p_ann.set_defaults(func=cmd_annotate)

    p_track = sub.add_parser("track", help="Headless tracking from annotation JSON (SLURM-ready)")
    p_track.add_argument("--annotations-json", type=Path, required=True)
    p_track.add_argument("--video", type=Path, default=None, help="Override video path in JSON")
    p_track.add_argument("--outdir", type=Path, default=Path("outputs"))
    p_track.add_argument("--scorer", type=str, default="cotracker")
    p_track.add_argument("--cpu", action="store_true")
    p_track.set_defaults(func=cmd_track)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
