import argparse
from pathlib import Path

import numpy as np

import open3d as o3d
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def downsample(
    workspace: str,
    voxel_size: float = 0.02,
):
    workspace = Path(workspace)

    tgt_label_file = workspace / "labels_downsampled.txt"
    tgt_pcd_file = workspace / "pcd_downsampled.ply"

    if tgt_label_file.exists() and tgt_pcd_file.exists():
        return

    rgb_mesh = o3d.io.read_triangle_mesh(str(workspace / "mesh.ply"))

    coords = np.asarray(rgb_mesh.vertices)
    colors = np.asarray(rgb_mesh.vertex_colors)

    rgb_mesh.compute_vertex_normals()
    normals = np.asarray(rgb_mesh.vertex_normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    downsampled_pcd = pcd.voxel_down_sample(
        voxel_size=voxel_size,
    )
    downsampled_coords = np.asarray(downsampled_pcd.points)

    color_interp = LinearNDInterpolator(
        points=coords,
        values=colors,
        fill_value=0.0,
    )
    downsampled_colors = color_interp(downsampled_coords)
    downsampled_colors = np.clip(downsampled_colors, 0, 1)
    downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

    normal_interp = LinearNDInterpolator(
        points=coords,
        values=normals,
        fill_value=0.1,
    )
    downsampled_normals = normal_interp(downsampled_coords)
    downsampled_pcd.normals = o3d.utility.Vector3dVector(downsampled_normals)
    downsampled_pcd.normalize_normals()

    o3d.io.write_point_cloud(str(workspace / "pcd_downsampled.ply"), downsampled_pcd)

    point_lifted_mesh = o3d.io.read_triangle_mesh(str(workspace / "point_lifted_mesh.ply"))
    point_lifted_coords = np.asarray(point_lifted_mesh.vertices)

    wordnet_label = np.loadtxt(str(workspace / "labels.txt"), dtype=np.uint8).reshape(-1, 1)

    label_interp = NearestNDInterpolator(
        x=point_lifted_coords,
        y=wordnet_label,
    )
    downsampled_wordnet_label = label_interp(downsampled_coords).astype(np.uint8)

    np.savetxt(str(workspace / "labels_downsampled.txt"), downsampled_wordnet_label, fmt="%i")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help='Path to workspace directory. There should be a "color" folder inside.',
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default="0.02",
        help="Name of files to save the labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    downsample(
        workspace=args.workspace,
        voxel_size=args.voxel_size,
    )

# python labelmaker_scripts/downsampling.py --workspace /media/hermann/data/labelmaker/ARKitScene_LabelMaker/Training/42897986
