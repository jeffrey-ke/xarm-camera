"""
convert_capture.py
------------------
Serialize capture result dicts into the dataset layout consumed
by StereoImageDataset.

Assumes the 3D poses in each result dict are already expressed in
the target frame — no coordinate transforms are applied here.

Input:  list of result dicts carrying raw numpy arrays
Output: index/indexNNN.json
        render/renderNNN/gripper_{left,right}_rgb/rgb_NNNN.png
        render/renderNNN/gripper_left_depth/distance_to_camera_NNNN.npy
"""
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from datastructs import Capture


def build_index(captures: List[Capture]):
    """Build a minimal index dict from capture results.

    Each result's actual_pose[:3] is taken directly as the offset.
    """
    key_frames = []
    for i, capture in enumerate(captures):
        offset = list(capture.offset)
        key_frames.append({
            "index": i,
            "offset": offset,
        })
    return {"key_frames": key_frames}


def write_capture(captures: List[Capture], render_dir: Path) -> None:
    """Encode and write image arrays into the dataset render layout."""
    left_dir = render_dir / "gripper_left_rgb"
    right_dir = render_dir / "gripper_right_rgb"
    depth_dir = render_dir / "gripper_left_depth"
    left_calib_dir = render_dir / "gripper_left_camera_calib_npy"
    base_pose_dir = render_dir / "gripper_base_pose"

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    left_calib_dir.mkdir(parents=True, exist_ok=True)
    base_pose_dir.mkdir(parents=True, exist_ok=True)

    for i, capture in enumerate(captures):
        idx_str = f"{i:04d}"
        cv2.imwrite(str(left_dir / f"rgb_{idx_str}.png"), capture.left_image)
        cv2.imwrite(str(right_dir / f"rgb_{idx_str}.png"), capture.right_image)
        np.save(str(depth_dir / f"distance_to_camera_{idx_str}.npy"), capture.left_depth)
        np.save(left_calib_dir / f"calib_{idx_str}.npy", capture.left_calib)
        np.save(base_pose_dir / f"se3_pose_{idx_str}.npy", capture.robot2base)


def convert_to_dataset(
    captures: List[dict],
    dataset_dir: Path | str,
    scene_num: int,
) -> Path:
    """
    Write capture results into the dataset directory layout expected
    by StereoImageDataset.

    Creates (or appends to) the dataset at dataset_dir:
        dataset_dir/index/index{scene_num:03d}.json
        dataset_dir/render/render{scene_num:03d}/gripper_{left,right}_rgb/rgb_NNNN.png
        dataset_dir/render/render{scene_num:03d}/gripper_left_depth/distance_to_camera_NNNN.npy
    """
    dataset_dir = Path(dataset_dir)
    scene_tag = f"{scene_num:03d}"

    index_dir = dataset_dir / "index"
    render_dir = dataset_dir / "render" / f"render{scene_tag}"
    index_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(captures)
    index_path = index_dir / f"index{scene_tag}.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    write_capture(captures, render_dir)

    return index_path
