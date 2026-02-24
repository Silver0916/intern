from pathlib import Path
import cv2
import numpy as np
import argparse
import csv
import time

# all_files = sorted((DATASET_DIR/SCENE).rglob('*.png'))

def ORB_feature_matching(img1_path, img2_path, output_dir, nfeatures=1500, fastThreshold=20, scaleFactor=1.2, nlevels=8, low_ratio=0.75, ransac_reproj_thr=5.0,show = False, return_points = False):
    img1_bgr = cv2.imread(str(img1_path))
    img2_bgr = cv2.imread(str(img2_path))

    if img1_bgr is None or img2_bgr is None:
        raise FileNotFoundError("One of the images could not be read. Please check the file paths.")

    #   ---grey conversion---
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    #   ---ORB feature detection and description---
    orb = cv2.ORB_create(nfeatures = nfeatures,    fastThreshold=fastThreshold,    scaleFactor=scaleFactor,    nlevels=nlevels)
    kps1, des1 = orb.detectAndCompute(gray1, None)
    kps2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("ORB failed to compute descriptors (des1/des2 is None).")

    print("img1:", img1_bgr.shape, "img2:", img2_bgr.shape)
    print("kp1:", len(kps1), "kp2:", len(kps2))
    print("des1:", None if des1 is None else des1.shape, "des2:", None if des2 is None else des2.shape)

    #   ---Brute-Force matcher with Hamming distance and cross-checking---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # apply ratio test to filter matches
    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        else:
            m, n = pair
            if m.distance < low_ratio * n.distance:
                good_matches.append(m)

    print("Good matches after ratio test:", len(good_matches))

    #   ---RANSAC-based homography estimation---
    if len(good_matches) >= 4:
        src_pts = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thr)
    else:
        raise RuntimeError("Not enough matches to compute homography (need at least 4).")

    if H is None or mask is None:
        raise RuntimeError("findHomography failed (H/mask is None).")
        
    mask = mask.copy().ravel().astype(bool)
    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(good_matches)

    print(f"inliers={num_inliers}/{len(good_matches)}  ratio={inlier_ratio:.3f}")
    print("H=\n", H)

    # ---real matches after RANSAC---
    inlier_matches = [m for m, inlier in zip(good_matches, mask) if inlier]

    # good_matches
    vis_good = cv2.drawMatches(img1_bgr, kps1, img2_bgr, kps2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ok = cv2.imwrite(str(output_dir / "good_matches.png"), vis_good)
    if not ok:
        raise RuntimeError(f"Failed to save good matches visualization to {output_dir / 'good_matches.png'}")

    # inlier_matches(real matches after RANSAC)
    vis_inliers = cv2.drawMatches(img1_bgr, kps1, img2_bgr, kps2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    ok = cv2.imwrite(str(output_dir / "inlier_matches.png"), vis_inliers)
    if not ok:
        raise RuntimeError(f"Failed to save inlier matches visualization to {output_dir / 'inlier_matches.png'}")
    
    # control if show the matches
    if show:
        cv2.imshow("ORB Matches", vis_good)
        cv2.imshow("ORB Inlier Matches", vis_inliers)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return results
    if return_points:
        return {
                "good_pts1": src_pts, 
                "good_pts2": dst_pts,
                "inlier_mask": mask, 
                "H": H
                }
    else:
        return {"kp1": len(kps1),
                "kp2": len(kps2), 
                "good_matches": len(good_matches), 
                "inliers": len(inlier_matches), 
                "inlier_ratio": inlier_ratio, 
                }

def process_scene(scene: str | Path, 
                  dataset_path : Path, 
                  output_root: Path, nfeatures, 
                  fastThreshold, scaleFactor, 
                  nlevels, low_ratio, 
                  ransac_reproj_thr, 
                  show, 
                  img_prtp = 0):

    # make output directory for the scene
    scene_out_dir = output_root / scene
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    # prepare csv file for recording results
    csv_path = scene_out_dir / "results.csv"
    fieldnames = ["scene","pair","kp1","kp2","good_matches","inliers","inlier_ratio","status","reason"]
    
    # read all images in the scene
    img_list = sorted((dataset_path/scene).rglob('*.ppm'))
    
    # protect
    if len(img_list) <= img_prtp:
        print(f"[SKIP] img_prtp={img_prtp} out of range, images={len(img_list)}, scene={scene}")
        return
    img1_path = img_list[img_prtp]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:  # create the file if it doesn't exist
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
        writer.writeheader()

        # process all pairs of images in the scene
        for img2_path in img_list[img_prtp+1:]:
            print(f"Processing {img1_path.name} and {img2_path.name}...")
            pair_out_dir = scene_out_dir / f"{img1_path.stem}_{img2_path.stem}"
            pair_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                results = ORB_feature_matching(img1_path, 
                                    img2_path, 
                                    pair_out_dir, 
                                    nfeatures, 
                                    fastThreshold, 
                                    scaleFactor, 
                                    nlevels, 
                                    low_ratio, 
                                    ransac_reproj_thr, 
                                    show)
                if results['inlier_ratio'] < 0.5 or results['inliers'] < 10 or results['good_matches'] < 20:
                    print(f"Warning: Low inlier ratio ({results['inlier_ratio']:.3f}) or few inliers ({results['inliers']}) for pair {img1_path.name} and {img2_path.name}")
                    writer.writerow(
                        {
                            'scene': str(scene),
                            'pair': f"{img1_path.name} - {img2_path.name}",
                            'kp1': results['kp1'],
                            'kp2': results['kp2'],
                            'good_matches': results['good_matches'],
                            'inliers': results['inliers'],
                            'inlier_ratio': f"{results['inlier_ratio']:.6f}",
                            'status': 'warning',
                            'reason': f"Low inlier ratio ({results['inlier_ratio']:.6f}) or few inliers ({results['inliers']})",
                        }
                    )

                else:
                    writer.writerow(
                        {
                        'scene': str(scene),
                        'pair': f"{img1_path.name} - {img2_path.name}",
                        'kp1': results['kp1'],
                        'kp2': results['kp2'],
                        'good_matches': results['good_matches'],
                        'inliers': results['inliers'],
                        'inlier_ratio': f"{results['inlier_ratio']:.6f}",
                        'status': 'success',
                        'reason':'',
                        }
                    )

            except Exception as e:
                writer.writerow({
                    'scene': str(scene),
                    'pair': f"{img1_path.name} - {img2_path.name}",
                    'kp1': '',
                    'kp2': '',
                    'good_matches': '',
                    'inliers': '',
                    'inlier_ratio': '',
                    'status': 'failure',
                    'reason': str(e),   
                })
                print(f"Error processing pair {img1_path.name} and {img2_path.name}: {e}")
                continue
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ORB feature matching')
    # mode selection
    #parser.add_argument('--dataset_mode', action='store_true', help='Path to the first image')
    # path config
    parser.add_argument('--dataset_dir', type=str, default=r'D:\intern_dataset\hpatches-sequences-release', help='Path to the first image')
    parser.add_argument('--scene', type=str, default=r'v_bark', help='Path to the second image')
    parser.add_argument('--output_root', type=str, default=r'D:\projs\intern\ORB\output', help='Path to the output directory')
    # ORB parameters
    parser.add_argument('--nfeatures', type=int, default=1500, help='Number of features to retain')
    parser.add_argument('--fastThreshold', type=int, default=20, help='FAST threshold for ORB')
    parser.add_argument('--scaleFactor', type=float, default=1.2, help='Pyramid decimation ratio')
    parser.add_argument('--nlevels', type=int, default=8, help='Number of pyramid levels')
    # RANSAC parameters
    parser.add_argument('--low_ratio', type=float, default=0.75, help='Low ratio for filtering matches')
    parser.add_argument('--ransac_reproj_thr', type=float, default=5.0, help='RANSAC reprojection threshold')
    parser.add_argument('--show', action='store_true', help='Whether to display the matches')
    args = parser.parse_args()

    # path config
    DATASET_DIR = Path(args.dataset_dir)
    SCENE = args.scene
    img_list = sorted((DATASET_DIR/SCENE).rglob('*.ppm'))

    OUTPUT_ROOT = Path(args.output_root)
    OUTPUT_ROOT = OUTPUT_ROOT / DATASET_DIR.name

    # ORB feature matching
    process_scene(SCENE, DATASET_DIR, 
                  OUTPUT_ROOT, args.nfeatures, 
                  args.fastThreshold, 
                  args.scaleFactor, 
                  args.nlevels, 
                  args.low_ratio, 
                  args.ransac_reproj_thr, 
                  args.show, 
                  img_prtp = 0)

