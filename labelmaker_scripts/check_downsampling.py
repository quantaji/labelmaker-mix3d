import argparse
from pathlib import Path




def check(
    workspace: str
):
    workspace = Path(workspace)

    tgt_label_file = workspace / "labels_downsampled.txt"
    tgt_pcd_file = workspace / "pcd_downsampled.ply"

    if tgt_label_file.exists() and tgt_pcd_file.exists():
        return
    else:
        print(f"{str(workspace)} is NOT finished!")

    


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
    check(
        workspace=args.workspace,
        
    )

