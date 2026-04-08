#!/usr/bin/env python3
"""
Two-step top-down training pipeline:
1) train detector (tool boxes)
2) train pose model on cropped detections

Uses existing output style from topdown_cotracker_dlc.py:
- <video>_tool_boxes.csv
- <video>_dlc_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _frame_to_path(frames_dir: Path, frame_id: int) -> Path:
    return frames_dir / f"img{frame_id:06d}.png"


def extract_frames(video_path: Path, frame_ids: List[int], outdir: Path) -> None:
    _ensure_dir(outdir)
    frame_set = sorted(set(frame_ids))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    for fid in frame_set:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {fid}")
        cv2.imwrite(str(_frame_to_path(outdir, fid)), frame)
    cap.release()


def prepare_detector_dataset(tool_boxes_csv: Path, frames_dir: Path, outdir: Path) -> Path:
    """Prepare YOLO-style detector dataset from tool boxes."""
    df = pd.read_csv(tool_boxes_csv)
    _ensure_dir(outdir / "images")
    _ensure_dir(outdir / "labels")

    class_map = {"tool_left": 0, "tool_right": 1}

    grouped = df.groupby("frame")
    for frame_id, g in grouped:
        src = _frame_to_path(frames_dir, int(frame_id))
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]

        dst_img = outdir / "images" / src.name
        cv2.imwrite(str(dst_img), img)

        label_path = outdir / "labels" / f"{src.stem}.txt"
        with label_path.open("w") as f:
            for _, row in g.iterrows():
                cls = class_map.get(str(row["tool"]))
                if cls is None:
                    continue
                x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    data_yaml = outdir / "dataset.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {outdir}",
                "train: images",
                "val: images",
                "names:",
                "  0: tool_left",
                "  1: tool_right",
            ]
        )
        + "\n"
    )
    return data_yaml


def _parse_dlc_csv(dlc_csv: Path) -> pd.DataFrame:
    return pd.read_csv(dlc_csv, header=[0, 1, 2], index_col=0)


def _extract_frame_id_from_index(idx: str) -> int:
    # expected labeled-data/<video>/img000123.png
    stem = Path(idx).stem
    return int(stem.replace("img", ""))


def prepare_pose_crop_dataset(tool_boxes_csv: Path, dlc_csv: Path, frames_dir: Path, outdir: Path, scorer: str = "twostep") -> Tuple[Path, Path]:
    """
    Build cropped pose dataset in DLC multiindex style.
    Uses bodyparts encoded as tool__pointname (existing top-down exporter style).
    """
    boxes = pd.read_csv(tool_boxes_csv)
    labels = _parse_dlc_csv(dlc_csv)

    crops_dir = outdir / "images"
    _ensure_dir(crops_dir)

    # identify tool-specific bodyparts from existing labels
    bodyparts = [bp for bp in labels.columns.get_level_values(1).unique()]
    tool_to_parts: Dict[str, List[str]] = {"tool_left": [], "tool_right": []}
    for bp in bodyparts:
        if bp.startswith("tool_left__"):
            tool_to_parts["tool_left"].append(bp)
        elif bp.startswith("tool_right__"):
            tool_to_parts["tool_right"].append(bp)

    out_rows = []
    out_index = []

    by_frame_tool = {(int(r.frame), str(r.tool)): r for r in boxes.itertuples()}

    for idx, row in labels.iterrows():
        frame_id = _extract_frame_id_from_index(str(idx))
        src = _frame_to_path(frames_dir, frame_id)
        if not src.exists():
            continue
        img = cv2.imread(str(src))

        for tool in ("tool_left", "tool_right"):
            key = (frame_id, tool)
            if key not in by_frame_tool:
                continue
            br = by_frame_tool[key]
            x1, y1, x2, y2 = int(br.x1), int(br.y1), int(br.x2), int(br.y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]

            crop_name = f"img{frame_id:06d}_{tool}.png"
            cv2.imwrite(str(crops_dir / crop_name), crop)

            # remap keypoints into crop coordinates
            vals = []
            for bp in tool_to_parts[tool]:
                x = float(row[(row.index[0][0], bp, "x")]) - x1
                y = float(row[(row.index[0][0], bp, "y")]) - y1
                l = float(row[(row.index[0][0], bp, "likelihood")])
                vals.extend([x, y, l])

            out_rows.append(vals)
            out_index.append(f"labeled-data/crops/{crop_name}")

    # flatten columns into DLC multiindex
    cols = []
    for tool in ("tool_left", "tool_right"):
        for bp in tool_to_parts[tool]:
            short_name = bp.split("__", 1)[1] if "__" in bp else bp
            cols.extend([(scorer, f"{tool}__{short_name}", "x"), (scorer, f"{tool}__{short_name}", "y"), (scorer, f"{tool}__{short_name}", "likelihood")])
    columns = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])

    pose_df = pd.DataFrame(out_rows, index=out_index, columns=columns)
    csv_out = outdir / "pose_crops_dlc_labels.csv"
    h5_out = outdir / "pose_crops_dlc_labels.h5"
    pose_df.to_csv(csv_out)
    pose_df.to_hdf(h5_out, key="df_with_missing", mode="w")

    meta = {
        "num_crops": len(out_rows),
        "bodyparts": [c[1] for c in cols if c[2] == "x"],
        "note": "Train DLC pose model on crop images with these labels.",
    }
    (outdir / "pose_crops_metadata.json").write_text(json.dumps(meta, indent=2))
    return csv_out, h5_out


def train_detector_yolo(data_yaml: Path, outdir: Path, epochs: int, imgsz: int, model: str) -> None:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics not installed. pip install ultralytics") from exc
    _ensure_dir(outdir)
    y = YOLO(model)
    y.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(outdir), name="detector")


def train_pose_dlc(config_path: Path) -> None:
    try:
        import deeplabcut
    except Exception as exc:
        raise RuntimeError("deeplabcut not installed in this environment") from exc
    deeplabcut.create_training_dataset(str(config_path))
    deeplabcut.train_network(str(config_path))
    deeplabcut.evaluate_network(str(config_path))


def run_inference_two_step(video: Path, detector_weights: Path, pose_config: Path, outdir: Path, conf: float = 0.25) -> None:
    """Detector -> crop -> pose pipeline scaffold.

    Requires ultralytics + deeplabcut runtime.
    Writes detected boxes; pose on crops is delegated to DLC analyze step.
    """
    try:
        from ultralytics import YOLO
        import deeplabcut
    except Exception as exc:
        raise RuntimeError("Need ultralytics and deeplabcut installed for inference") from exc

    _ensure_dir(outdir)
    det = YOLO(str(detector_weights))
    boxes_csv = outdir / "inference_boxes.csv"

    cap = cv2.VideoCapture(str(video))
    frame_id = 0
    rows = []
    crop_dir = outdir / "inference_crops"
    _ensure_dir(crop_dir)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = det.predict(source=frame, conf=conf, verbose=False)
        if len(res) > 0 and res[0].boxes is not None:
            b = res[0].boxes
            xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.array([])
            cls = b.cls.cpu().numpy().astype(int) if hasattr(b.cls, "cpu") else np.array([])
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                tool = "tool_left" if cls[i] == 0 else "tool_right"
                rows.append({"frame": frame_id, "tool": tool, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    cv2.imwrite(str(crop_dir / f"img{frame_id:06d}_{tool}_{i}.png"), crop)
        frame_id += 1
    cap.release()
    pd.DataFrame(rows).to_csv(boxes_csv, index=False)

    # DLC pose inference on crops
    deeplabcut.analyze_time_lapse_frames(str(pose_config), str(crop_dir), frametype=".png", save_as_csv=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-step top-down DLC training/inference helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Prepare detector and pose-crop datasets")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--tool-boxes-csv", type=Path, required=True)
    prep.add_argument("--dlc-csv", type=Path, required=True)
    prep.add_argument("--outdir", type=Path, required=True)

    td = sub.add_parser("train-detector", help="Train detector model")
    td.add_argument("--data-yaml", type=Path, required=True)
    td.add_argument("--outdir", type=Path, required=True)
    td.add_argument("--epochs", type=int, default=100)
    td.add_argument("--imgsz", type=int, default=640)
    td.add_argument("--model", type=str, default="yolov8n.pt")

    tp = sub.add_parser("train-pose", help="Train DLC pose model")
    tp.add_argument("--dlc-config", type=Path, required=True)

    inf = sub.add_parser("infer", help="Detector->pose inference")
    inf.add_argument("--video", type=Path, required=True)
    inf.add_argument("--detector-weights", type=Path, required=True)
    inf.add_argument("--pose-config", type=Path, required=True)
    inf.add_argument("--outdir", type=Path, required=True)
    inf.add_argument("--conf", type=float, default=0.25)

    args = ap.parse_args()

    if args.cmd == "prepare":
        out = args.outdir
        frames_dir = out / "frames"
        det_dir = out / "detector_dataset"
        pose_dir = out / "pose_crops_dataset"
        boxes_df = pd.read_csv(args.tool_boxes_csv)
        frame_ids = boxes_df["frame"].astype(int).tolist()
        extract_frames(args.video, frame_ids, frames_dir)
        data_yaml = prepare_detector_dataset(args.tool_boxes_csv, frames_dir, det_dir)
        pose_csv, pose_h5 = prepare_pose_crop_dataset(args.tool_boxes_csv, args.dlc_csv, frames_dir, pose_dir)
        print(f"Prepared frames: {frames_dir}")
        print(f"Detector dataset yaml: {data_yaml}")
        print(f"Pose labels: {pose_csv}, {pose_h5}")
    elif args.cmd == "train-detector":
        train_detector_yolo(args.data_yaml, args.outdir, args.epochs, args.imgsz, args.model)
    elif args.cmd == "train-pose":
        train_pose_dlc(args.dlc_config)
    elif args.cmd == "infer":
        run_inference_two_step(args.video, args.detector_weights, args.pose_config, args.outdir, args.conf)


if __name__ == "__main__":
    main()
