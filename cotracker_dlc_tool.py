#!/usr/bin/env python3
"""Keypoint-only CoTracker pipeline with local annotate + remote tracking modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


def read_video_info(video_path: Path) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    out = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return out


def load_video_range(video_path: Path, start: int, end: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(end - start + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames loaded")
    return np.stack(frames, axis=0)


def select_frame_range(video_path: Path, frame_count: int) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    win = "Keypoint-only: frame range (q/Enter confirm)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Start", win, 0, frame_count - 1, lambda _: None)
    cv2.createTrackbar("End", win, frame_count - 1, frame_count - 1, lambda _: None)

    while True:
        s = cv2.getTrackbarPos("Start", win)
        e = cv2.getTrackbarPos("End", win)
        if e < s:
            e = s
            cv2.setTrackbarPos("End", win, e)

        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        ok2, f2 = cap.read()
        if ok1 and ok2:
            cv2.putText(f1, f"START {s}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(f2, f"END {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(win, np.vstack([f1, f2]))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 13):
            cv2.destroyWindow(win)
            cap.release()
            return s, e


def select_points(frame: np.ndarray) -> List[dict]:
    points: List[dict] = []
    win = "Click points (q/Enter done, u undo)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def redraw() -> None:
        vis = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(vis, (int(p["x"]), int(p["y"])), 4, (255, 255, 0), -1)
            cv2.putText(vis, f"{i}:{p['name']}", (int(p["x"]) + 6, int(p["y"]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow(win, vis)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            name = input(f"Name for ({x},{y}) [blank=auto]: ").strip()
            if not name:
                name = f"point_{len(points):02d}"
            points.append({"name": name, "x": float(x), "y": float(y)})
            redraw()

    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 13) and points:
            cv2.destroyWindow(win)
            return points
        if k == ord("u") and points:
            points.pop()
            redraw()


def save_annotations(path: Path, video: Path, start: int, end: int, points: List[dict]) -> None:
    payload = {"video": str(video), "frame_start": start, "frame_end": end, "points": points}
    path.write_text(json.dumps(payload, indent=2))


def load_annotations(path: Path):
    data = json.loads(path.read_text())
    return Path(data["video"]), int(data["frame_start"]), int(data["frame_end"]), data["points"]


def run_cotracker(clip: np.ndarray, qxy: np.ndarray, force_cpu: bool):
    from cotracker.predictor import CoTrackerPredictor

    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    video = torch.from_numpy(clip[..., ::-1].copy()).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    queries = np.concatenate([np.zeros((qxy.shape[0], 1), dtype=np.float32), qxy], axis=1)
    queries = torch.from_numpy(queries).unsqueeze(0).to(device)
    model = CoTrackerPredictor().to(device)
    model.eval()
    with torch.no_grad():
        tracks, vis = model(video, queries=queries)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy(), device


def export_dlc(outdir: Path, video: Path, frame_start: int, points: List[dict], tracks: np.ndarray, vis: np.ndarray, scorer: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cols = []
    for p in points:
        cols.extend([(scorer, p["name"], "x"), (scorer, p["name"], "y"), (scorer, p["name"], "likelihood")])
    cols = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])

    rows, idx = [], []
    for i in range(tracks.shape[0]):
        row = []
        for j in range(len(points)):
            row.extend([float(tracks[i, j, 0]), float(tracks[i, j, 1]), float(vis[i, j])])
        rows.append(row)
        idx.append(f"labeled-data/{video.stem}/img{frame_start+i:06d}.png")

    df = pd.DataFrame(rows, index=idx, columns=cols)
    df.to_csv(outdir / f"{video.stem}_dlc_labels.csv")
    df.to_hdf(outdir / f"{video.stem}_dlc_labels.h5", key="df_with_missing", mode="w")


def cmd_annotate(args):
    fc, w, h, fps = read_video_info(args.video)
    print(f"Video: {args.video} | frames={fc} | size={w}x{h} | fps={fps:.2f}")
    s, e = select_frame_range(args.video, fc)
    frame0 = load_video_range(args.video, s, s)[0]
    points = select_points(frame0)
    save_annotations(args.annotations_out, args.video, s, e, points)
    print(f"Saved annotations to {args.annotations_out}")


def cmd_track(args):
    video, s, e, points = load_annotations(args.annotations_json)
    if args.video is not None:
        video = args.video
    clip = load_video_range(video, s, e)
    qxy = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    tracks, vis, device = run_cotracker(clip, qxy, args.cpu)
    print(f"Tracking finished on {device}")
    export_dlc(args.outdir, video, s, points, tracks, vis, args.scorer)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keypoint-only CoTracker local annotate + remote track")
    sub = p.add_subparsers(dest="cmd", required=True)

    ann = sub.add_parser("annotate", help="Local GUI annotation")
    ann.add_argument("--video", type=Path, required=True)
    ann.add_argument("--annotations-out", type=Path, required=True)
    ann.set_defaults(func=cmd_annotate)

    track = sub.add_parser("track", help="Headless tracking from saved annotations")
    track.add_argument("--annotations-json", type=Path, required=True)
    track.add_argument("--video", type=Path, default=None, help="Override video path")
    track.add_argument("--outdir", type=Path, default=Path("outputs"))
    track.add_argument("--scorer", type=str, default="cotracker")
    track.add_argument("--cpu", action="store_true")
    track.set_defaults(func=cmd_track)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
