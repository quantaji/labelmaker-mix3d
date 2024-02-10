import os
import re
from hashlib import md5
from pathlib import Path

import numpy as np

import open3d as o3d
import pandas as pd
from fire import Fire
from joblib import Parallel, delayed
from labelmaker import label_mappings
from labelmaker.label_data import get_wordnet
from loguru import logger
from mix3d.datasets.preprocessing.base_preprocessing import BasePreprocessing
from mix3d.datasets.scannet200.scannet200_constants import CLASS_LABELS_200, SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200
from mix3d.utils.point_cloud_utils import load_ply_with_normals
from natsort import natsorted
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def get_wordnet_to_scannet200_mapping():
    table = pd.read_csv(Path(os.path.dirname(os.path.realpath(label_mappings.__file__))) / "mappings" / "label_mapping.csv")
    wordnet = get_wordnet()
    wordnet_keys = [x["name"] for x in wordnet]
    mapping = {}
    for row in table.index:
        if table["wnsynsetkey"][row] not in wordnet_keys:
            continue
        scannet_id = table.loc[row, "id"]
        wordnet199_id = next(x for x in wordnet if x["name"] == table["wnsynsetkey"][row])["id"]

        if scannet_id in VALID_CLASS_IDS_200:
            mapping.setdefault(wordnet199_id, set()).add(scannet_id)

    wn199_size = np.array([x["id"] for x in wordnet]).max() + 1
    mapping_array = np.zeros(shape=(wn199_size,), dtype=np.uint16)
    for wordnet199_id in mapping.keys():
        mapping_array[wordnet199_id] = min(mapping[wordnet199_id])

    return mapping_array


class ARKitScenesLabelMakerPreprocessing(BasePreprocessing):

    mode_mapping = {
        "train": "Training",
        "validation": "Validation",
    }

    def __init__(
        self,
        data_dir: str = "./data/raw/ARKitScenes_LabelMaker",
        save_dir: str = "./data/processed/arkitscenes_labelmaker",
        modes: tuple = {
            "validation",
            "train",
        },
        n_jobs: int = -1,
        voxel_size: float = 0.02,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.create_label_database()

        self.mapping_array = get_wordnet_to_scannet200_mapping()

        self.voxel_size = voxel_size

        # gathering files
        for mode in self.modes:
            split_dir = self.data_dir / self.mode_mapping[mode]
            scene_names = os.listdir(str(split_dir))

            filepaths = []
            for scene in scene_names:
                filepaths.append(split_dir / scene / "mesh.ply")

            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = {}
        for row_id, class_id in enumerate(VALID_CLASS_IDS_200):
            label_database[class_id] = {
                "color": SCANNET_COLOR_MAP_200[class_id],
                "name": CLASS_LABELS_200[row_id],
                "validation": True,
            }
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def _parse_scene(self, filepath: Path):
        return filepath.parent.name

    def process_file(self, filepath, mode):
        video_id = self._parse_scene(filepath=filepath)

        filebase = {
            "filepath": filepath,
            "scene": self._parse_scene(filepath),
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)

        downsampled_pcd = pcd.voxel_down_sample(
            voxel_size=self.voxel_size,
        )
        downsampled_coords = np.asarray(downsampled_pcd.points)
        feat_interp = LinearNDInterpolator(
            points=coords,
            values=features,
            fill_value=0.0,
        )
        downsampled_features = feat_interp(downsampled_coords)

        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((downsampled_coords, downsampled_features))

        # reading label in wordnet format
        point_lifted_filepath = Path(str(filepath).replace("mesh.ply", "point_lifted_mesh.ply"))
        point_lifted_coords, _, _ = load_ply_with_normals(point_lifted_filepath)

        label_file = Path(filepath).parent / "labels.txt"
        wordnet_label = np.loadtxt(str(label_file), dtype=np.uint8).reshape(-1, 1)
        scannet200_label = self.mapping_array[wordnet_label]

        label_interp = NearestNDInterpolator(
            x=point_lifted_coords,
            y=scannet200_label,
        )
        downsampled_scannet200_label = label_interp(downsampled_coords)

        points = np.hstack((points, downsampled_scannet200_label))

        # ## Testing starts
        # test_pcd = o3d.geometry.PointCloud()
        # test_pcd.points = o3d.utility.Vector3dVector(downsampled_coords)

        # rgb_color = downsampled_features[:, :3]
        # test_pcd.colors = o3d.utility.Vector3dVector(rgb_color / 255)

        # o3d.io.write_point_cloud(str(self.save_dir / mode / f"{video_id}_colored.ply"), test_pcd)

        # label_color = np.asarray(np.vectorize(SCANNET_COLOR_MAP_200.get)(downsampled_scannet200_label)) / 255
        # test_pcd.colors = o3d.utility.Vector3dVector(np.moveaxis(label_color.reshape(3, -1), 0, -1))
        # o3d.io.write_point_cloud(str(self.save_dir / mode / f"{video_id}_labeled.ply"), test_pcd)
        # ## Testing ends

        processed_filepath = self.save_dir / mode / f"{video_id}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        filebase["color_mean"] = [
            float((downsampled_features[:, 0] / 255).mean()),
            float((downsampled_features[:, 1] / 255).mean()),
            float((downsampled_features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((downsampled_features[:, 0] / 255) ** 2).mean()),
            float(((downsampled_features[:, 1] / 255) ** 2).mean()),
            float(((downsampled_features[:, 2] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/arkitscenes_labelmaker/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)


if __name__ == "__main__":
    Fire(ARKitScenesLabelMakerPreprocessing)
