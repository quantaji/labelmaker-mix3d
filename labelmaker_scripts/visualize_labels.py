import argparse
from pathlib import Path

import numpy as np

import open3d as o3d
from labelmaker.label_mappings import get_wordnet


def visualize_downsampled_pcd_label(workspace: str, save_path: str):
    workspace = Path(workspace)
    save_path = Path(save_path)

    pcd_path = workspace / "pcd_downsampled.ply"
    label_path = workspace / "labels_downsampled.txt"

    pcd = o3d.io.read_point_cloud(str(pcd_path))

    label = np.loadtxt(str(workspace / "labels.txt"), dtype=np.uint8).reshape(-1, 1)

    color_mapping = {}
    info = get_wordnet()
    for item in info:
        color_mapping[item["id"]] = item["color"]

    label_color = np.asarray(np.vectorize(color_mapping.get)(label)) / 255
    pcd.colors = o3d.utility.Vector3dVector(np.moveaxis(label_color.reshape(3, -1), 0, -1))

    o3d.io.write_point_cloud(str(save_path), pcd)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help='Path to workspace directory. There should be a "color" folder inside.',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    visualize_downsampled_pcd_label(
        workspace=args.workspace,
        save_path=args.save_path,
    )
