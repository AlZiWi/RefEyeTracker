import numpy as np
import cv2
import json
import pickle
import logging

from calibration_utils import triangulate_point, transformation_matrix_from_calib_ext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_gaze_estimation(self, gaze_thread_handler, gaze_imgs, until_frame_id, gaze_max_frame_idxs, cam_keys, experiment_path, output_path_suffix, calib_results):
    gaze_thread_handler["thread_status"] = "running"
    
    if not gaze_imgs:
        status_text = "[Gaze] No frames loaded to run gaze estimation on."
        gaze_thread_handler["status_q"].put(status_text)
        gaze_thread_handler["thread_status"] = "complete"
        return
    
    status_text = "Status: Running Gaze Estimation..."
    gaze_thread_handler["status_q"].put(status_text)

    def fit_sphere(points):
        """
        points: Nx3 array of 3D points
        Returns: center (3,), radius
        """
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        # Build linear system: [x y z 1] [a b c d]^T = -(x^2 + y^2 + z^2)
        A = np.column_stack([X, Y, Z, np.ones_like(X)])
        b = -(X**2 + Y**2 + Z**2)

        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, b_, c, d = params

        center = np.array([-a/2, -b_/2, -c/2])
        radius = np.sqrt((a**2 + b_**2 + c**2)/4 - d)

        return center, radius


    def fit_sphere_robust(points, zscore_thresh=3.0, max_iters=3):
        pts = points.copy()
        for _ in range(max_iters):
            center, radius = fit_sphere(pts)
            d = np.linalg.norm(pts - center[None, :], axis=1)
            z = (d - radius) / (np.std(d) + 1e-6)
            mask = np.abs(z) < zscore_thresh
            if mask.sum() == pts.shape[0]:  # nothing removed
                break
            pts = pts[mask]
        return center, radius
    
    def compute_optical_axis(pupil_3d, C_eye):
        v = pupil_3d - C_eye[None, :]
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        d = v / norms
        return d  # Nx3
    
    pupils_wc_all_frames = {
        "left": {},
        "right": {}
    }
    
    if until_frame_id == -1:
        until_frame_id = max(gaze_max_frame_idxs.values()) + 1
    else:
        until_frame_id = min(until_frame_id, max(gaze_max_frame_idxs.values()) + 1)

    for frame_idx in range(until_frame_id):
        status_text = f"[Gaze] Triangulating pupils for frame {frame_idx}..."
        gaze_thread_handler["status_q"].put(status_text)
        
        pupils_wc = self.triangulate_pupil_for_frame(frame_idx, cam_keys, gaze_imgs, calib_results)
        for eye in ["left", "right"]:
            pupils_wc_all_frames[eye][frame_idx] = pupils_wc.get(eye, None)

    # filter out None values and fit sphere for left and right pupils
    fit_results = {
        "left": None,
        "right": None
    }
    
    for eye in ["left", "right"]:
        # filter dict to only include valid points
        valid_points = {key: p for key, p in pupils_wc_all_frames[eye].items() if p is not None}
        valid_points_keys = list(valid_points.keys())
        valid_points_arr = np.array(list(valid_points.values()))
        valid_points_arr = np.reshape(valid_points_arr, (-1, 3))  # Ensure shape is Nx3
        if valid_points_arr.shape[0] >= 4:  # Need at least 4 points to fit a sphere
            status_text = f"[Gaze] Fitting sphere for {eye} eye..."
            gaze_thread_handler["status_q"].put(status_text)

            C_eye, R_eye = fit_sphere_robust(valid_points_arr)
            optical_dirs = compute_optical_axis(valid_points_arr, C_eye)  # Nx3
            optical_dirs_dict = {valid_points_keys[i]: optical_dirs[i] for i in range(len(valid_points_keys))}
            origins = np.repeat(C_eye[None, :], valid_points_arr.shape[0], axis=0)  # Nx3
            origins_dict = {valid_points_keys[i]: origins[i] for i in range(len(valid_points_keys))}
            fit_results[eye] = {
                "center": C_eye,
                "radius": R_eye,
                "optical_dirs": optical_dirs_dict,
                "origins": origins_dict
            }
            print(f"[Gaze] Fitted sphere for {eye} eye: Center = {C_eye}, Radius = {R_eye}")
        else:
            print(f"[Gaze] Not enough valid points to fit sphere for {eye} eye.")
    
    # Save Results
    results_path = experiment_path / output_path_suffix
    gaze_results_data = {
        "fit_results": fit_results,
        "pupils_wc_all_frames": pupils_wc_all_frames
    }
    # save as pkl
    with open(results_path, "wb") as f:
        pickle.dump(gaze_results_data, f)

    status_text = f"[Gaze] Gaze estimation results saved to {results_path}"
    gaze_thread_handler["status_q"].put(status_text)
    
    gaze_thread_handler["thread_status"] = "complete"


def triangulate_pupil_for_frame(self, frame_idx, cam_keys, gaze_imgs, calib_results):
    seg_data_for_cams = {}

    for cam_key in cam_keys:
        if cam_key in gaze_imgs.keys() and frame_idx in gaze_imgs[cam_key]["img_names_aligned_by_timestamp"].keys() and gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][frame_idx] is not None:
            fname = gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][frame_idx]["filename"]
            # load and display image
            try:                    
                # load json data for segmentation overlay
                segmentation_path = gaze_imgs[cam_key]["segmentation_path"] / fname
                segmentation_path = segmentation_path.with_suffix(".json")
                if segmentation_path.exists():
                    with open(segmentation_path, "r") as f:
                        seg_data = json.load(f)
                        seg_data_for_cams[cam_key] = seg_data
            except Exception as e:
                print(f"[Gaze] Error loading segmentation data for {cam_key} frame {frame_idx}: {e}")

    pupils_wc = self.triangulate_pupils_for_coords(seg_data_for_cams, calib_results)

    return pupils_wc


def triangulate_pupils_for_coords(self, seg_data_for_cams, calib_results):
    pupils_wc = {}

    cam_keys = ("ro", "ri")
    if all(cam_key in seg_data_for_cams for cam_key in cam_keys):
        point_coords = [seg_data_for_cams[cam_key]["center"] for cam_key in cam_keys]
        point_wc, (cam1_point_vector_wc, cam2_point_vector_wc), projection_error = triangulate_point(calib_results["calibration_steps"]["Stereo R-R"]["data"], point_coords)
        if point_wc is not None:
            pupils_wc["right"] = point_wc

    cam_keys = ("lo", "li")
    if all(cam_key in seg_data_for_cams for cam_key in cam_keys):
        point_coords = [seg_data_for_cams[cam_key]["center"] for cam_key in cam_keys]
        point_wc, (cam1_point_vector_wc, cam2_point_vector_wc), projection_error = triangulate_point(calib_results["calibration_steps"]["Stereo L-L"]["data"], point_coords)

        if point_wc is not None:
            T = transformation_matrix_from_calib_ext(calib_results["calibration_steps"]["Stereo R-L"]["data"]["camera_params_1"].extrinsic)
            point_wc = T * np.matrix(np.append(point_wc, 1)).T  # Convert to homogeneous coordinates for transformation
            point_wc = point_wc[:3]  # Convert back to 3D coordinates
            pupils_wc["left"] = point_wc

    return pupils_wc


def rotate_image(self, image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result