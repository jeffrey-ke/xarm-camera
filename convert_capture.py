"""
convert_capture.py
------------------
Convert goto_capture.py result dicts into the dataset format
consumed by StereoImageDataset.

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


def transform_pose_to_offset(pose_6dof: list, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to the [x,y,z] position of a 6-DOF pose."""
    xyz = np.array(pose_6dof[:3], dtype=np.float64)
    homogeneous = np.append(xyz, 1.0)
    transformed = T @ homogeneous
    return transformed[:3]


def build_index(results: List[dict], T: np.ndarray) -> dict:
    """Build a minimal index dict from capture results and a coordinate transform."""
    key_frames = []
    for i, result in enumerate(results):
        offset = transform_pose_to_offset(result["actual_pose"], T)
        key_frames.append({
            "index": i,
            "offset": offset.tolist(),
        })
    return {"key_frames": key_frames}


def write_images(results: List[dict], render_dir: Path) -> None:
    """Encode and write image arrays into the dataset render layout."""
    left_dir = render_dir / "gripper_left_rgb"
    right_dir = render_dir / "gripper_right_rgb"
    depth_dir = render_dir / "gripper_left_depth"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        idx_str = f"{i:04d}"
        cv2.imwrite(str(left_dir / f"rgb_{idx_str}.png"), result["left_data"])
        cv2.imwrite(str(right_dir / f"rgb_{idx_str}.png"), result["right_data"])
        np.save(str(depth_dir / f"distance_to_camera_{idx_str}.npy"), result["depth_data"])


def convert_to_dataset(
    results: List[dict],
    dataset_dir: Path | str,
    scene_num: int,
    T: np.ndarray,
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

    index = build_index(results, T)
    index_path = index_dir / f"index{scene_tag}.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    write_images(results, render_dir)

    return index_path
