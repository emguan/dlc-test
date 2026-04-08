#!/usr/bin/env python3
"""
DLC multi-animal top-down workflow helper (YOLO-free).

Concept mapping requested:
- individuals = tools
- bodyparts = keypoints

Typical flow:
1) init-project (creates config.yaml)
2) extract-frames
3) label-gui (manual labeling in DLC GUI)
4) train / evaluate
5) analyze (inference)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import yaml


def _require_deeplabcut():
    try:
        import deeplabcut
    except Exception as exc:
        raise RuntimeError("deeplabcut is required. Install in your training environment.") from exc
    return deeplabcut


def init_project(
    project_name: str,
    experimenter: str,
    videos: List[Path],
    working_directory: Path,
    bodyparts: List[str],
    individuals: List[str],
    copy_videos: bool,
) -> Path:
    """Create multi-animal DLC project and set config keys for identical tools."""
    deeplabcut = _require_deeplabcut()

    video_strs = [str(v) for v in videos]
    config_path = deeplabcut.create_new_project(
        project=project_name,
        experimenter=experimenter,
        videos=video_strs,
        working_directory=str(working_directory),
        copy_videos=copy_videos,
        multianimal=True,
    )

    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    # Multi-animal setup for identical tools.
    cfg["individuals"] = individuals
    cfg["uniquebodyparts"] = []
    cfg["multianimalbodyparts"] = bodyparts

    # Optional defaults useful for tools/endoscopy.
    if "skeleton" not in cfg:
        cfg["skeleton"] = []

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def extract_frames(config: Path, mode: str, algo: str) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.extract_frames(str(config), mode=mode, algo=algo)


def label_gui(config: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.label_frames(str(config))


def check_labels(config: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.check_labels(str(config))


def create_trainset(config: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.create_training_dataset(str(config))


def train(config: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.train_network(str(config))


def evaluate(config: Path) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.evaluate_network(str(config))


def analyze(config: Path, videos: List[Path], save_as_csv: bool) -> None:
    deeplabcut = _require_deeplabcut()
    deeplabcut.analyze_videos(str(config), [str(v) for v in videos], save_as_csv=save_as_csv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DLC multi-animal top-down training helper (no YOLO)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init-project", help="Create multi-animal DLC project + config.yaml")
    p_init.add_argument("--project-name", required=True)
    p_init.add_argument("--experimenter", required=True)
    p_init.add_argument("--videos", nargs="+", type=Path, required=True)
    p_init.add_argument("--working-directory", type=Path, required=True)
    p_init.add_argument("--bodyparts", nargs="+", required=True, help="Shared points for each tool")
    p_init.add_argument("--individuals", nargs="+", default=["tool_left", "tool_right"])
    p_init.add_argument("--copy-videos", action="store_true")

    p_extract = sub.add_parser("extract-frames", help="Extract frames for labeling")
    p_extract.add_argument("--config", type=Path, required=True)
    p_extract.add_argument("--mode", default="automatic", choices=["automatic", "manual"])
    p_extract.add_argument("--algo", default="kmeans", choices=["kmeans", "uniform"])

    p_label = sub.add_parser("label-gui", help="Open DLC GUI for frame labeling")
    p_label.add_argument("--config", type=Path, required=True)

    p_check = sub.add_parser("check-labels", help="Visual check of manual labels")
    p_check.add_argument("--config", type=Path, required=True)

    p_trainset = sub.add_parser("create-trainset", help="Create DLC training dataset from labels")
    p_trainset.add_argument("--config", type=Path, required=True)

    p_train = sub.add_parser("train", help="Train DLC network")
    p_train.add_argument("--config", type=Path, required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate DLC network")
    p_eval.add_argument("--config", type=Path, required=True)

    p_analyze = sub.add_parser("analyze", help="Run inference on new videos")
    p_analyze.add_argument("--config", type=Path, required=True)
    p_analyze.add_argument("--videos", nargs="+", type=Path, required=True)
    p_analyze.add_argument("--save-as-csv", action="store_true")

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "init-project":
        config = init_project(
            project_name=args.project_name,
            experimenter=args.experimenter,
            videos=args.videos,
            working_directory=args.working_directory,
            bodyparts=args.bodyparts,
            individuals=args.individuals,
            copy_videos=args.copy_videos,
        )
        print(f"Created config: {config}")
    elif args.cmd == "extract-frames":
        extract_frames(args.config, args.mode, args.algo)
    elif args.cmd == "label-gui":
        label_gui(args.config)
    elif args.cmd == "check-labels":
        check_labels(args.config)
    elif args.cmd == "create-trainset":
        create_trainset(args.config)
    elif args.cmd == "train":
        train(args.config)
    elif args.cmd == "evaluate":
        evaluate(args.config)
    elif args.cmd == "analyze":
        analyze(args.config, args.videos, args.save_as_csv)


if __name__ == "__main__":
    main()
