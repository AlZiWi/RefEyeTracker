import numpy as np
import cv2
import json
from ellseg import quick_ellseg, DEFAULT_WEIGHTS

def estimate_camera_tilt_angle(self, calib_results, cam_key):
    """Estimate the camera tilt angle based on calibration results."""
    sc_z = calib_results["extrinsics"]["sc"]["z"]
    sc_y = calib_results["extrinsics"]["sc"]["y"]
    cam_y = calib_results["extrinsics"][cam_key]["y"].A1
    cam_y_proj = cam_y - np.dot(cam_y, sc_z) * sc_z  # Project cam_y onto plane orthogonal to sc_z (only catch angle orthogonal to eye/face)
    cam_y_proj /= np.linalg.norm(cam_y_proj)  # Normalize
    cam_tilt_angle_deg = np.degrees(np.arccos(np.clip(np.dot(cam_y_proj, sc_y), -1.0, 1.0)))
    if np.dot(np.cross(sc_y, cam_y_proj), sc_z) < 0:  # If the cross product points in the opposite direction of sc_z, the angle is negative
        cam_tilt_angle_deg = -cam_tilt_angle_deg
        
    return cam_tilt_angle_deg


def detect_pupil_ellseg(self, imgdata, alpha=1.0, beta=0.0, crop_ratio=0, camera_tilt_angle_deg=0):
    """Detect pupil using Ellseg and return the results along with a visualization image."""
    
    # Rotate image to compensate for camera tilt
    if camera_tilt_angle_deg != 0:
        imgdata = self.rotate_image(imgdata, camera_tilt_angle_deg)
    
    # Crop image
    if crop_ratio > 0:
        img_dims = np.array(imgdata.shape)
        
        h, w = img_dims
        x_offset = int(w * (crop_ratio/2))
        y_offset = int(h * (crop_ratio/2))
        w_cropped = int(w * (1 - crop_ratio))
        h_cropped = int(h * (1 - crop_ratio))
        imgdata = imgdata[y_offset:y_offset+h_cropped, x_offset:x_offset+w_cropped]

    # Run Ellseg
    res = quick_ellseg(
        imgdata,
        checkpoint=DEFAULT_WEIGHTS,
        include_image=True,
        use_auto_brightness=True,
        alpha=alpha, beta=beta,
        debug_plot=False,
    )

    center = np.array(tuple(map(int, res["center"])))
    axes = np.array(tuple(map(int, (res["axes"][0] / 2, res["axes"][1] / 2))))  # OpenCV ellipse expects half-axes
    angle_deg = res["angle_deg"]
    
    img_dims_after_crop = np.array(imgdata.shape)
    
    # Rotate ellipse back by -camera_tilt_angle_deg
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), -camera_tilt_angle_deg, 1)
    center_orig = center - img_dims_after_crop[::-1] / 2  # Shift to origin
    center_orig = rotation_matrix[:, :2] @ center_orig  # Apply rotation
    center_orig += img_dims_after_crop[::-1] / 2  # Shift back
    angle_deg_orig = angle_deg + camera_tilt_angle_deg  # Adjust angle by tilt

    imgdata = res["image"]
    imgdata_orig = imgdata.copy()
    imgdata_orig = self.rotate_image(imgdata_orig, -camera_tilt_angle_deg)
    
    # Uncrop the image back to original size
    if crop_ratio > 0:
        h, w = img_dims
        x_offset = int(w * (crop_ratio/2))
        y_offset = int(h * (crop_ratio/2))
        w_cropped = int(w * (1 - crop_ratio))
        h_cropped = int(h * (1 - crop_ratio))

        center_orig = center_orig + np.array([x_offset, y_offset])
        axes_orig = axes  # axes remain the same since cropping doesn't change the size of the ellipse, only its position

        imgdata_orig_uncrop = np.zeros((h, w), dtype=imgdata_orig.dtype)
        imgdata_orig_uncrop[y_offset:y_offset+h_cropped, x_offset:x_offset+w_cropped] = imgdata_orig
        imgdata_orig = imgdata_orig_uncrop
    
    res = {
        "image": imgdata_orig,
        "center": center_orig,
        "axes": axes_orig,
        "angle_deg": angle_deg_orig
    }
    
    res_visualization = {
        "image": imgdata,
        "axes": axes,
        "center": center,
        "angle_deg": angle_deg
    }
    
    return res, res_visualization


def run_segmentation(self, gaze_thread_handler, gaze_imgs, until_frame_id, cam_keys, experiment_path, output_path_suffix, params):
    gaze_thread_handler["thread_status"] = "running"

    status_text = "Status: Running Segmentation..."
    gaze_thread_handler["status_q"].put(status_text)

    for cam_key in cam_keys:
        if cam_key in gaze_imgs:
            print(f"[Gaze] Running segmentation on {len(gaze_imgs[cam_key]['img_names_aligned_by_timestamp']) if until_frame_id < 0 else until_frame_id} frames for camera {cam_key}")
            stats = {
                "success": 0,
                "failure": 0
            }

            for frame_idx, frame_info in gaze_imgs[cam_key]['img_names_aligned_by_timestamp'].items():
                if frame_idx > until_frame_id and until_frame_id != -1:
                    continue
                if frame_info is None:
                    continue
                
                img_path = gaze_imgs[cam_key]["cam_path"] / frame_info["filename"]
                imgdata = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                imgdata = cv2.undistort(imgdata, cameraMatrix=params["camera_matrices"][cam_key], distCoeffs=params["distortion_coeffs"][cam_key])
                
                # save the segmented image for visualization
                save_path = experiment_path / f"{cam_key}{output_path_suffix}" / f"{frame_info['filename']}"
                # create directory if it doesn't exist
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    res, _ = self.detect_pupil_ellseg(
                        imgdata,
                        alpha=params["alpha"], beta=params["beta"],
                        crop_ratio=params["crop_ratio"],
                        camera_tilt_angle_deg=params["camera_tilt_angles_deg"][cam_key]
                    )
                    
                    # Save segmentation data to file
                    segmentation_data = {
                        "frame_idx": frame_idx,
                        "center": res["center"].tolist(),
                        "axes": res["axes"].tolist(),
                        "angle_deg": res["angle_deg"],
                    }
                    
                    with open(save_path.with_suffix(".json"), "w") as f:
                        json.dump(segmentation_data, f)

                    print(f"[Gaze] Segmented image saved for {cam_key} frame {frame_idx}: {save_path}")
                    stats["success"] += 1
                except Exception as e:
                    print(f"[Gaze] Pupil detection error for {cam_key} frame {frame_idx}: {e}")
                    stats["failure"] += 1
                    #  Remove json file if it exists to avoid confusion
                    if save_path.with_suffix(".json").exists():
                        save_path.with_suffix(".json").unlink()

                status_text = f"Status: Running Segmentation...\nCam {cam_key}: {stats['success']} success, {stats['failure']} failure\nFrame {frame_idx}/{len(gaze_imgs[cam_key]['img_names_aligned_by_timestamp']) if until_frame_id < 0 else until_frame_id}\n{img_path}"
                gaze_thread_handler["status_q"].put(status_text)

            print(f"[Gaze] Segmentation stats for {cam_key}: {stats}")
        else:
            print(f"[Gaze] No frames loaded for camera {cam_key}.")
    
    status_text = "Status: Segmentation Complete."
    gaze_thread_handler["status_q"].put(status_text)
    
    gaze_thread_handler["thread_status"] = "complete"