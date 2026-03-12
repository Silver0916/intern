import subprocess
import argparse
from pathlib import Path
import sys

def run_sift_match(dataset_path, output_root, single_scene=False, img_prtp=0, scene=None):

    Path(output_root).mkdir(parents=True, exist_ok=True)

    sift_dir = Path(__file__).resolve().parent
    sift_match_py = sift_dir / "SIFT_match.py"
    if not sift_match_py.exists():
        raise FileNotFoundError(f"SIFT_match.py not found: {sift_match_py}")

    dataset_path = Path(dataset_path)

    if single_scene:
        assert scene is not None, "Scene name must be provided when single_scene is True."

        cmd = [
            sys.executable, str(sift_match_py),
            "--dataset_dir", str(dataset_path),
            "--scene", str(scene),
            "--output_root", str(output_root),
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate()

        if proc.returncode == 0:
            print("SIFT matching completed successfully for scene:", scene)
        else:
            print("Error occurred during SIFT matching for scene:", scene, "| returncode=", proc.returncode)
        if out:
            print(out)

    else:
        scene_list = sorted([p for p in dataset_path.iterdir() if p.is_dir()])
        if not scene_list:
            raise FileNotFoundError(f"No scene subfolders found under: {dataset_path}")

        for scene_path in scene_list:
            cmd = [
                sys.executable, str(sift_match_py),
                "--dataset_dir", str(dataset_path),
                "--scene", scene_path.name,
                "--output_root", str(output_root),
            ]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = proc.communicate()

            if proc.returncode == 0:
                print("SIFT matching completed successfully for scene:", scene_path.name)
            else:
                print("Error occurred during SIFT matching for scene:", scene_path.name, "| returncode=", proc.returncode)
            if out:
                print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIFT matching on a dataset.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset containing scene subfolders.")
    parser.add_argument("--output_root", type=str, required=True, help="Root directory where results will be saved.")
    parser.add_argument("--single_scene", action="store_true", help="Whether to process only a single scene.")
    parser.add_argument("--img_prtp", type=int, default=0, help="(Unused unless SIFT_match.py supports it).")
    parser.add_argument("--scene", type=str, default=None, help="Scene name to process (only for single scene mode).")

    args = parser.parse_args()

    run_sift_match(args.dataset_path, args.output_root, args.single_scene, args.img_prtp, args.scene)
