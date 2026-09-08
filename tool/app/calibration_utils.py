from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from scipy.spatial.transform import Rotation
from skspatial.objects import Line, Plane, Sphere

from app.data_structures import CalibrationLabel, CalibrationSummary, CameraCoordinateFrame, CameraParams, CameraParamsExtrinsic, LabelCoordinates, MirrorCalibrationResults, MirrorCalibrationResultsForType, MirrorCalibrationType, SceneCameraExtrinsicsCalibrationResults, StereoCalibrationResults, UIPreset
from app.stereo_calib_opencv import detect_corners, run_opencv_mono_calibration, run_opencv_stereo_calibration


def _calculate_absolute_camera_calibration_results(stereo_calibration_results: StereoCalibrationResults, calib_params_offset: Optional[CameraParams] = None) -> StereoCalibrationResults:
    '''
    Calculate the pairwise camera calibration vectors for a stereo camera setup.
    calib_params_offset = output of this function for the first camera, to be used as absolute reference frame instead of relative
    '''

    if calib_params_offset is not None:
        T_offset = calib_params_offset.extrinsic.absolute.T
        T_R_offset = calib_params_offset.extrinsic.absolute.T_R
        T_t_offset = calib_params_offset.extrinsic.absolute.T_t

        stereo_calibration_results.camera_params_0.extrinsic.absolute = CameraCoordinateFrame(
            T=T_offset,
            T_R=T_R_offset,
            T_t=T_t_offset,
            origin=T_offset @ stereo_calibration_results.camera_params_0.extrinsic.relative.origin,
            x=T_R_offset @ stereo_calibration_results.camera_params_0.extrinsic.relative.x,
            y=T_R_offset @ stereo_calibration_results.camera_params_0.extrinsic.relative.y,
            z=T_R_offset @ stereo_calibration_results.camera_params_0.extrinsic.relative.z
        )

        stereo_calibration_results.camera_params_1.extrinsic.absolute = CameraCoordinateFrame(
            T=T_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.T,
            T_R=T_R_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.T_R,
            T_t=T_t_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.T_t,
            origin=T_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.origin,
            x=T_R_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.x,
            y=T_R_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.y,
            z=T_R_offset @ stereo_calibration_results.camera_params_1.extrinsic.relative.z
        )
    else:
        stereo_calibration_results.camera_params_0.extrinsic.absolute = stereo_calibration_results.camera_params_0.extrinsic.relative
        stereo_calibration_results.camera_params_1.extrinsic.absolute = stereo_calibration_results.camera_params_1.extrinsic.relative


    return stereo_calibration_results


def _change_basis_of_mirror_calibration_results(mirror_calibration_results: MirrorCalibrationResults, T: np.ndarray) -> MirrorCalibrationResults:
    '''
    Change the basis of the mirror calibration results to a new coordinate frame defined by the transformation matrix T.
    '''
    transformed_results: MirrorCalibrationResults = {}
    T_R = np.eye(4)
    T_R[:3, :3] = T[:3, :3]
    
    for calib_type, calib_results in mirror_calibration_results.items():
        transformed_surface_point = _change_of_basis(calib_results.surface_point, T).flatten()
        transformed_surface_normal = T_R @ calib_results.surface_normal
        
        transformed_calibrated_points = {}
        for point_label, point_result in calib_results.calibrated_points.items():
            transformed_point_world = _change_of_basis(point_result.point_world, T).flatten()
            transformed_point_mirrored = _change_of_basis(point_result.point_mirrored, T).flatten()
            transformed_cam_point_vectors = T_R @ point_result.cam_point_vectors  # Rotate the camera point vectors using the rotation part of T
            
            transformed_calibrated_points[point_label] = MirrorCalibrationResultsForType.MirrorCalibrationResultsForPoint(
                point_world=transformed_point_world,
                point_mirrored=transformed_point_mirrored,
                cam_point_vectors=transformed_cam_point_vectors,
                projection_error=point_result.projection_error
            )
        
        transformed_results[calib_type] = MirrorCalibrationResultsForType(
            surface_point=transformed_surface_point,
            surface_normal=transformed_surface_normal,
            img_paths=calib_results.img_paths,
            calibrated_points=transformed_calibrated_points
        )

    return transformed_results


def calculate_calibration_summary(calibration_summary: CalibrationSummary) -> CalibrationSummary:
    '''
    Calculate absolute calibration summary based on what's available in provided calibration results steps.
    '''
    
    if calibration_summary.intermediate_results.STEREO_RR is not None:
        stereo_calibration_results_rr = _calculate_absolute_camera_calibration_results(calibration_summary.intermediate_results.STEREO_RR)
        calibration_summary.intermediate_results.STEREO_RR = stereo_calibration_results_rr
        calibration_summary.CAM_RO = stereo_calibration_results_rr.camera_params_0
        calibration_summary.CAM_RI = stereo_calibration_results_rr.camera_params_1
        
        if calibration_summary.intermediate_results.MIRROR_R is not None:
            # TODO introduce absolute annd relative categories for mirror calibration results
            transformed_mirror_r = _change_basis_of_mirror_calibration_results(calibration_summary.intermediate_results.MIRROR_R, T=calibration_summary.CAM_RO.extrinsic.absolute.T)
            # keys in priority order according to Enum
            _, best_transformed_mirror_r = sorted(transformed_mirror_r.items(), key=lambda item: list(MirrorCalibrationType).index(item[0]))[0]
            calibration_summary.POINTS_R = best_transformed_mirror_r
    
        if calibration_summary.intermediate_results.STEREO_LL is not None and calibration_summary.intermediate_results.STEREO_RL is not None:
            stereo_calibration_results_rl = _calculate_absolute_camera_calibration_results(calibration_summary.intermediate_results.STEREO_RL)
            stereo_calibration_results_ll = _calculate_absolute_camera_calibration_results(calibration_summary.intermediate_results.STEREO_LL, calib_params_offset=stereo_calibration_results_rl.camera_params_1)
            calibration_summary.intermediate_results.STEREO_RL = stereo_calibration_results_rl
            calibration_summary.intermediate_results.STEREO_LL = stereo_calibration_results_ll
            calibration_summary.CAM_LO = stereo_calibration_results_ll.camera_params_0
            calibration_summary.CAM_LI = stereo_calibration_results_ll.camera_params_1
                
            if calibration_summary.intermediate_results.MIRROR_L is not None:
                transformed_mirror_l = _change_basis_of_mirror_calibration_results(calibration_summary.intermediate_results.MIRROR_L, T=calibration_summary.CAM_LO.extrinsic.absolute.T)
                # keys in priority order according to Enum
                _, best_transformed_mirror_l = sorted(transformed_mirror_l.items(), key=lambda item: list(MirrorCalibrationType).index(item[0]))[0]
                calibration_summary.POINTS_L = best_transformed_mirror_l
            
            if calibration_summary.intermediate_results.MIRROR_SC:
                calibration_summary.CAM_SC = calibration_summary.intermediate_results.MIRROR_SC.camera_params

    return calibration_summary
    

def run_stereo_calibration(calibration_summary: CalibrationSummary, step: UIPreset, img_path_pairs: List[tuple[Path, Path]], pattern_size: tuple[int, int], square_size_mm: float, pixel_pitch_mm: float) -> CalibrationSummary:
    '''
    OpenCV stereo calibration
    '''
    
    stereo_calibration_results = run_opencv_stereo_calibration(
        img_path_pairs=img_path_pairs,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        pixel_pitch_mm=pixel_pitch_mm
    )
    
    if step == UIPreset.STEREO_RR:
        calibration_summary.intermediate_results.STEREO_RR = stereo_calibration_results
    elif step == UIPreset.STEREO_LL:
        calibration_summary.intermediate_results.STEREO_LL = stereo_calibration_results
    elif step == UIPreset.STEREO_RL:
        calibration_summary.intermediate_results.STEREO_RL = stereo_calibration_results
    
    calibration_summary = calculate_calibration_summary(calibration_summary)
    
    return calibration_summary



def run_mono_calibration(calibration_summary: CalibrationSummary, step: UIPreset, img_paths: List[Path], pattern_size: tuple[int, int], square_size_mm: float, pixel_pitch_mm: tuple[float, float]) -> CalibrationSummary:
    '''
    OpenCV mono calibration
    '''
    
    mono_calibration_results = run_opencv_mono_calibration(
        img_paths=img_paths,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        pixel_pitch_mm=pixel_pitch_mm,
        verbose=False
    )
    
    if step == UIPreset.MONO_SC:
        calibration_summary.intermediate_results.MONO_SC = mono_calibration_results
        
    calibration_summary = calculate_calibration_summary(calibration_summary)

    return calibration_summary




def run_mirror_points_calibration(calibration_summary: CalibrationSummary, step: UIPreset, img_pair_paths: Tuple[Path], points_label_pairs: LabelCoordinates, camera_label_pairs: LabelCoordinates, pattern_corners: tuple[int, int] | None=None) -> CalibrationSummary:
    '''
    Triangulate mirrored points based on stereo calibration results and labelled points in the images.
    '''
    
    if step not in [UIPreset.MIRROR_R, UIPreset.MIRROR_L]:
        raise ValueError("Invalid step for mirror points calibration. Must be either MIRROR_R or MIRROR_L.")
    
    if step == UIPreset.MIRROR_R:
        if calibration_summary.intermediate_results.STEREO_RR is None:
            raise ValueError("Stereo calibration results for STEREO_RR are required for mirror calibration.")
        stereo_calibration_results = calibration_summary.intermediate_results.STEREO_RR
    if step == UIPreset.MIRROR_L:
        if calibration_summary.intermediate_results.STEREO_LL is None:
            raise ValueError("Stereo calibration results for STEREO_LL are required for mirror calibration.")
        stereo_calibration_results = calibration_summary.intermediate_results.STEREO_LL
    
    try:
        mirror_calibration_results = _triangulate_mirrored_points(stereo_calibration_results, img_pair_paths, points_label_pairs, camera_label_pairs, pattern_corners)
    except Exception as e:
        raise ValueError(f"Error occurred while triangulating mirrored points: {e}")

    if len(mirror_calibration_results) > 0:
        if step == UIPreset.MIRROR_R:
            calibration_summary.intermediate_results.MIRROR_R = mirror_calibration_results
        elif step == UIPreset.MIRROR_L:
            calibration_summary.intermediate_results.MIRROR_L = mirror_calibration_results
            
        calibration_summary = calculate_calibration_summary(calibration_summary)
    
    return calibration_summary



def run_scene_camera_extrinsics_calibration(calibration_summary: CalibrationSummary, img_path: Path, calibration_pattern_size: tuple[float, float], calibration_pattern_square_size_mm: float, point_labels: LabelCoordinates) -> CalibrationSummary:
    '''
    Calibrate the scene camera extrinsics based on the mirror calibration results and labelled points in the image.
    '''
    
    if calibration_summary.intermediate_results.MONO_SC is None:
        raise ValueError("Mono calibration results for MONO_SC are required for scene camera extrinsics calibration.")
    
    sc_extrinsics_calibration_results = _calibrate_scene_camera_extrinsics(calibration_summary, img_path, calibration_pattern_size, calibration_pattern_square_size_mm, point_labels)
    
    calibration_summary.intermediate_results.MIRROR_SC = sc_extrinsics_calibration_results
    
    calibration_summary = calculate_calibration_summary(calibration_summary)
    
    return calibration_summary


def _triangulate_mirrored_points(stereo_calibration_results: StereoCalibrationResults, img_pair_paths: Tuple[Path], points_label_pairs: LabelCoordinates, camera_label_pairs: LabelCoordinates, pattern_corners: tuple[int, int] | None=None) -> MirrorCalibrationResults:
    '''
    Triangulate mirrored points based on stereo calibration results and labelled points in the images.
    '''
    
    imgs = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in img_pair_paths]
    
    mirror_calibration_results: MirrorCalibrationResults = {}
    
    if pattern_corners is not None:
        try:
            mirror_surface_point_wc, mirror_surface_normal_wc = _reconstruct_mirror_surface_from_pattern(stereo_calibration_results, imgs, pattern_corners)
            mirror_calibration_results[MirrorCalibrationType.PATTERN] = MirrorCalibrationResultsForType(
                surface_point=mirror_surface_point_wc,
                surface_normal=mirror_surface_normal_wc,
                img_paths=img_pair_paths
            )
        except Exception as e:
            print(f"[warning] Could not reconstruct mirror surface from pattern: {e}")

    if CalibrationLabel.CAM_0 in camera_label_pairs.keys():
        cam_orig = stereo_calibration_results.camera_params_0.extrinsic.relative.origin
        mirror_surface_point_wc, mirror_surface_normal_wc = _reconstruct_mirror_surface_from_camera_labels(stereo_calibration_results, camera_label_pairs[CalibrationLabel.CAM_0], cam_orig)
        mirror_calibration_results[MirrorCalibrationType.CAM_0] = MirrorCalibrationResultsForType(
            surface_point=mirror_surface_point_wc,
            surface_normal=mirror_surface_normal_wc,
            img_paths=img_pair_paths
        )

    if CalibrationLabel.CAM_1 in camera_label_pairs.keys():
        cam_orig = stereo_calibration_results.camera_params_1.extrinsic.relative.origin
        mirror_surface_point_wc, mirror_surface_normal_wc = _reconstruct_mirror_surface_from_camera_labels(stereo_calibration_results, camera_label_pairs[CalibrationLabel.CAM_1], cam_orig)
        mirror_calibration_results[MirrorCalibrationType.CAM_1] = MirrorCalibrationResultsForType(
            surface_point=mirror_surface_point_wc,
            surface_normal=mirror_surface_normal_wc,
            img_paths=img_pair_paths
        )

    if MirrorCalibrationType.CAM_1 in mirror_calibration_results.keys() and MirrorCalibrationType.CAM_0 in mirror_calibration_results.keys():
        mirror_calibration_results[MirrorCalibrationType.CAM_BOTH] = MirrorCalibrationResultsForType(
            surface_point=np.mean([mirror_calibration_results[MirrorCalibrationType.CAM_0].surface_point, mirror_calibration_results[MirrorCalibrationType.CAM_1].surface_point], axis=0),
            surface_normal=np.mean([mirror_calibration_results[MirrorCalibrationType.CAM_0].surface_normal, mirror_calibration_results[MirrorCalibrationType.CAM_1].surface_normal], axis=0),
            img_paths=img_pair_paths
        )
    
    if len(mirror_calibration_results) > 0:
        # Triangulate points
        for mirror_calib_key, mirror_calib in mirror_calibration_results.items():
            for point_label, point_coords in points_label_pairs.items():
                pt_wc, pt_wc_mirrored, cam_point_vectors, projection_error = _triangulate_mirrored_point(stereo_calibration_results, point_coords, mirror_calib.surface_point, mirror_calib.surface_normal)

                mirror_calibration_results[mirror_calib_key].calibrated_points[point_label] = MirrorCalibrationResultsForType.MirrorCalibrationResultsForPoint(
                    point_world=pt_wc,
                    point_mirrored=pt_wc_mirrored,
                    cam_point_vectors=cam_point_vectors,
                    projection_error=projection_error
                )

    return mirror_calibration_results


def _calibrate_scene_camera_extrinsics(calibration_summary: CalibrationSummary, img_path: Path, calibration_pattern_size: tuple[float, float], calibration_pattern_square_size_mm: float, point_labels: LabelCoordinates) -> SceneCameraExtrinsicsCalibrationResults:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_dimensions = (img.shape[1], img.shape[0])  # (width, height)
    
    # 1 Detect pattern corners in the image
    corners = detect_corners(img, calibration_pattern_size, verbose=True)
    if corners is None:
        raise ValueError("No corners detected in image. Please check the calibration pattern and image quality.")
    image_points_2d_mirror = corners.squeeze()

    # Labelled Data
    image_points_2d_cameras = np.array([
        point_labels[CalibrationLabel.CAM_RO],
        point_labels[CalibrationLabel.CAM_RI],
        point_labels[CalibrationLabel.CAM_LI],
        point_labels[CalibrationLabel.CAM_LO]
    ], dtype=np.float32).squeeze()
    
    # Perpendicular point of mirror (reflection of scene camera)
    image_point_2d_sc = np.array([
        point_labels[CalibrationLabel.CAM_SC]
    ], dtype=np.float32).squeeze()

    # 2 Undistort the image points using the mono calibration data
    
    if calibration_summary.intermediate_results.MONO_SC is None:
        raise ValueError("Mono calibration results for MONO_SC are required for scene camera extrinsics calibration.")
    sc_camera_params_intrinsic = calibration_summary.intermediate_results.MONO_SC
    
    image_points_2d_mirror_undist = cv2.undistortPoints(image_points_2d_mirror, sc_camera_params_intrinsic.K, sc_camera_params_intrinsic.dist, P=sc_camera_params_intrinsic.K).squeeze()
    image_points_2d_cameras_undist = cv2.undistortPoints(image_points_2d_cameras, sc_camera_params_intrinsic.K, sc_camera_params_intrinsic.dist, P=sc_camera_params_intrinsic.K).squeeze()
    image_point_2d_sc_undist = cv2.undistortPoints(image_point_2d_sc, sc_camera_params_intrinsic.K, sc_camera_params_intrinsic.dist, P=sc_camera_params_intrinsic.K).squeeze()

    # Generate world points based on pattern settings. assume z=0 for all points, and x,y based on square size and pattern size
    world_points_3d_mirror = np.zeros((calibration_pattern_size[0] * calibration_pattern_size[1], 3), dtype=np.float32)
    for i in range(calibration_pattern_size[1]):
        for j in range(calibration_pattern_size[0]):
            world_points_3d_mirror[i * calibration_pattern_size[0] + j] = [(j * calibration_pattern_square_size_mm), (-i * calibration_pattern_square_size_mm), 0]

    # 3 Use pattern points to get mirror x/y scaling
    
    ## Place Mirror in 3D space using PnP with the detected corners and known pattern geometry
    _, rvec_mirror, tvec_mirror = cv2.solvePnP(world_points_3d_mirror, image_points_2d_mirror, sc_camera_params_intrinsic.K, sc_camera_params_intrinsic.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    T_mirror = np.eye(4)
    T_mirror[:3, :3] = cv2.Rodrigues(rvec_mirror)[0]
    T_mirror[:3, 3] = tvec_mirror.flatten()
    print(f"Estimated mirror pose (rotation vector): {rvec_mirror.flatten()}, translation vector: {tvec_mirror.flatten()}, Euler angles (degrees): {Rotation.from_matrix(cv2.Rodrigues(rvec_mirror)[0]).as_euler('xyz', degrees=True)}")

    # Project PnP Result from below to get actual scaling factor in x and y direction, otherwise we would assume no tilt and that the pattern is parallel to the camera plane, which is not necessarily the case
    world_points_3d_mirror_projected, _ = cv2.projectPoints(world_points_3d_mirror, rvec_mirror, tvec_mirror, sc_camera_params_intrinsic.K, sc_camera_params_intrinsic.dist)
    world_points_3d_mirror_projected = world_points_3d_mirror_projected.squeeze()
    
    imgpt_length_pattern_corner_to_corner_diag = np.linalg.norm(world_points_3d_mirror_projected[0] - world_points_3d_mirror_projected[-1])
    world_length_pattern_corner_to_corner_diag = np.linalg.norm(world_points_3d_mirror[0] - world_points_3d_mirror[-1])
    scaling_factor = world_length_pattern_corner_to_corner_diag / imgpt_length_pattern_corner_to_corner_diag
    print(f"Calculated scaling factor from projected pattern points: {scaling_factor}")
    
    imgpt_length_pattern_corner_to_corner_diag = np.linalg.norm(image_points_2d_mirror_undist[0] - image_points_2d_mirror_undist[-1])
    world_length_pattern_corner_to_corner_diag = np.linalg.norm(world_points_3d_mirror[0] - world_points_3d_mirror[-1])
    scaling_factor = world_length_pattern_corner_to_corner_diag / imgpt_length_pattern_corner_to_corner_diag
    print(f"Calculated scaling factor from pattern points: {scaling_factor}")

    image_points_2d_cameras_undist_scaled = image_points_2d_cameras_undist * scaling_factor
    image_point_2d_sc_undist_scaled = image_point_2d_sc_undist * scaling_factor

    # 4 Calculate offset between sc and (mirrored) cam labels -> Distance perpendicular from sc to camera intersection on sc plane / 2

    reflected_camera_points_from_sc = image_points_2d_cameras_undist_scaled - image_point_2d_sc_undist_scaled
    
    # 6 Solve resulting triangle with known angle of that line and the calibrated distances between sc and cameras (from eye camera calibration)
    # TODO document in more detail
    
    if calibration_summary.CAM_RO is None or calibration_summary.CAM_RI is None or calibration_summary.CAM_LI is None or calibration_summary.CAM_LO is None:
        raise ValueError("Camera calibration results for CAM_RO, CAM_RI, CAM_LI, and CAM_LO are required for scene camera extrinsics calibration.")
    
    world_points_3d_frames = np.array([
        calibration_summary.CAM_RO.extrinsic.absolute.origin[:3,0],
        calibration_summary.CAM_RI.extrinsic.absolute.origin[:3,0],
        calibration_summary.CAM_LI.extrinsic.absolute.origin[:3,0],
        calibration_summary.CAM_LO.extrinsic.absolute.origin[:3,0]
    ], dtype=np.float32)

    origin_sc = calibration_summary.POINTS_L.calibrated_points[CalibrationLabel.CAM_SC].point_world
    print(f"Scene camera origin: {origin_sc}")
    
    # Get closest distance between mirror and scene camera based on the labelled point for the scene camera and the mirror plane
    plane_mirror = Plane(point=T_mirror[:3, 3], normal=T_mirror[:3, 2])  # Mirror plane defined by its origin and normal, assuming the normal is along the z-axis of the mirror's coordinate system
    line_sc_normal = Line(point=np.array([0, 0, 0]), direction=np.array([(image_point_2d_sc_undist[0] - img_dimensions[0]/2) * sc_camera_params_intrinsic.px_mm, (image_point_2d_sc_undist[1] - img_dimensions[1]/2) * sc_camera_params_intrinsic.py_mm, sc_camera_params_intrinsic.f_mm]))  # Line from the scene camera point in the direction of the camera's view
    #TODO this is not really needed, we can just calculate the distance from the scene camera point to the mirror plane along the normal direction (crossing 0,0,0), but this is a good sanity check to see if the reflected camera points and the mirror plane are consistent with the labelled scene camera point
    intersection = plane_mirror.intersect_line(line_sc_normal)
    distance_mirror_to_sc = np.linalg.norm(intersection - origin_sc[:3])
    print(f"Closest distance between mirror and scene camera: {distance_mirror_to_sc}")

    # Calculate possible orientations for the scene camera based on the reflected camera points and the mirror plane
    distance_cameras_to_sc = np.linalg.norm(world_points_3d_frames - origin_sc[:3], axis=1)  # Vectors from scene camera to each of the camera label points
    origin_sc_scwc = np.array([0, 0, 0])  # Scene camera origin in scene camera world coordinates (SCWC)
    origin_scd_scwc = np.array([0, 0, 2*distance_mirror_to_sc])  # Scene camera origin in scene camera world coordinates (SCWC) but reflected across the mirror plane, assuming the mirror is at z=distance_mirror_to_sc in SCWC
    possible_camera_positions = []
    spheres_possible_camera_positions_from_ext = [Sphere(origin_sc_scwc, distance_cameras_to_sc[i]) for i in range(len(distance_cameras_to_sc))]
    lines_possible_camera_positions_from_sc = [Line(point=origin_scd_scwc, direction=[reflected_camera_points_from_sc[i,0], reflected_camera_points_from_sc[i,1], -distance_mirror_to_sc]) for i in range(len(distance_cameras_to_sc))]
    for i in range(len(distance_cameras_to_sc)):
        try:
            possible_camera_positions_for_cam = spheres_possible_camera_positions_from_ext[i].intersect_line(lines_possible_camera_positions_from_sc[i])
        except Exception as e:
            print(f"Error occurred while intersecting: {e}")
            possible_camera_positions_for_cam = None
        possible_camera_positions.append(possible_camera_positions_for_cam)

    print(f"Possible camera positions based on external calibration: {possible_camera_positions}")
    
    #sol = minimize(lambda x: np.linalg.norm((x - world_points_3d_frames) - reflected_camera_points_from_sc), x0=origin_sc, bounds=[(-1000, 1000), (-1000, 1000), (0, 2000)])
    valid_position = [pos[1] if pos is not None else None for pos in possible_camera_positions]  # Take the first valid position as an initial guess
    
    c1_sc_direction = origin_sc[:3] - world_points_3d_frames[0]
    c2_sc_direction = origin_sc[:3] - world_points_3d_frames[3]

    #print(f"Direction from camera 1 to scene camera: {c1_sc_direction}, direction from camera 2 to scene camera: {c2_sc_direction}")
    
    #print(f"Angle between camera 1 and camera 2 directions: {np.rad2deg(angle_between_vectors(c1_sc_direction, c2_sc_direction))} degrees")
    #print(f"Angle between camera 1 and camera 2 directions: {np.rad2deg(angle_between_vectors(-valid_position[0], -valid_position[3]))} degrees")

    angle_scd_scc1 = _angle_between_vectors(-valid_position[0], np.array([0,0,1]))
    angle_scd_scc2 = _angle_between_vectors(-valid_position[3], np.array([0,0,1]))
    angle_scy_scc1 = _angle_between_vectors(-valid_position[0], np.array([0,1,0]))
    angle_scy_scc2 = _angle_between_vectors(-valid_position[3], np.array([0,1,0]))
    
    #print(f"Valid camera position based on external calibration: {valid_position}")

    #print(f"Angles from valid position 1 to camera 1 direction: {np.rad2deg(angle_scd_scc1)} (to z), {np.rad2deg(angle_scy_scc1)} (to y)")
    #print(f"Angles from valid position 2 to camera 2 direction: {np.rad2deg(angle_scd_scc2)} (to z), {np.rad2deg(angle_scy_scc2)} (to y)")

    # Estimate scene camera direction based on angles
    estimated_sc_z = _find_vector_by_angles(c1_sc_direction, c2_sc_direction, angle_scd_scc1, angle_scd_scc2)
    estimated_sc_y = _find_vector_by_angles(c1_sc_direction, c2_sc_direction, angle_scy_scc1, angle_scy_scc2)
    
    # Choose solution with negative z component (pointing away from the cameras)
    estimated_sc_z = [sol for sol in estimated_sc_z if sol[2] < 0][0]
    # Choose solution with negative y component (eye cameras are upside down)
    estimated_sc_y = [sol for sol in estimated_sc_y if sol[1] < 0][0]
    
    estimated_sc_T = np.eye(3, 4)
    estimated_sc_x = np.cross(estimated_sc_y, estimated_sc_z)
    estimated_sc_T[:3, :3] = np.column_stack((estimated_sc_x, estimated_sc_y, estimated_sc_z)).T  # Need inverse because we want to go from world to camera coordinates
    estimated_sc_T[:3, 3] = origin_sc
    
    #1print(f"Estimated scene camera direction: {estimated_sc_z}, estimated scene camera y direction: {estimated_sc_y}, estimated scene camera transformation matrix: {estimated_sc_T}")
    
    camera_params = CameraParams(intrinsic=sc_camera_params_intrinsic,
        extrinsic=CameraParamsExtrinsic(
        relative=CameraCoordinateFrame(
            T=estimated_sc_T,
            R=estimated_sc_T[:3, :3],
            t=estimated_sc_T[:3, 3],
            origin=origin_sc,
            x=estimated_sc_x,
            y=estimated_sc_y,
            z=estimated_sc_z
        )
    ))
    camera_params.extrinsic.absolute = camera_params.extrinsic.relative  # Scene Camera is already in world coordinates, so absolute and relative are the same

    scene_camera_extrinsics_calibration_results = SceneCameraExtrinsicsCalibrationResults(
        camera_params=camera_params,
        mirror_pose=SceneCameraExtrinsicsCalibrationResults.MirrorPose(
            T_mirror=T_mirror,
            rvec_mirror=rvec_mirror,
            tvec_mirror=tvec_mirror
        ),
        points=SceneCameraExtrinsicsCalibrationResults.LabelPoints(
            image_points_2d_mirror=image_points_2d_mirror,
            image_points_2d_cameras=image_points_2d_cameras,
            image_point_2d_sc=image_point_2d_sc,
            image_points_2d_mirror_undist=image_points_2d_mirror_undist,
            image_points_2d_cameras_undist=image_points_2d_cameras_undist,
            image_point_2d_sc_undist=image_point_2d_sc_undist,
            world_points_3d_mirror=world_points_3d_mirror,
            world_points_3d_frames=world_points_3d_frames
        ),
        scaling_factor=scaling_factor,
        distance_mirror_to_sc=distance_mirror_to_sc,
        possible_camera_positions=possible_camera_positions,
        img_path=img_path
    )
    
    return scene_camera_extrinsics_calibration_results


def _lsq_intersection_of_lines(point_vecs: np.ndarray, normal_vecs: np.ndarray, verbose=False):
    '''
    Find the intersection point of multiple lines in 3D space.
    point_vecs: list of 3D points on each line (N x 3)
    normal_vecs: list of normal vectors for each line (N x 3)
    '''
    # TODO idea: use weighted sum depending on ellipse fit quality etc.?

    vec_len = point_vecs.shape[1]

    S1 = np.zeros((vec_len, vec_len))
    S2 = np.zeros((vec_len, 1))

    dsts = []
    
    for i in range(len(normal_vecs)):
        nv = np.array([normal_vecs[i]]).T
        pv = np.array([point_vecs[i]]).T

        nv = _unit_vector(nv)
        
        s1 = np.identity(vec_len) - (nv @ nv.T)
        s2 = (np.identity(vec_len) - (nv @ nv.T)) @ pv
        
        S1 += s1
        S2 += s2

    c_proj = np.linalg.pinv(S1) @ S2

    dsts = []
    for i in range(len(normal_vecs)):
        nv = normal_vecs[i]
        pv = point_vecs[i]
        nv = _unit_vector(nv)
        dsts.append(np.linalg.norm(np.cross(nv, pv - c_proj)))
    projection_error = np.sum(np.array(dsts))

    if verbose:
        print(f"[dbg] point_vecs: {point_vecs}")
        print(f"[dbg] normal_vecs: {normal_vecs}")
        print(f"[dbg] c_proj: {c_proj}")
        print(f"[dbg] dsts: {dsts}")
        print(f"[dbg] projection_error: {projection_error}")

    return c_proj, projection_error


def _change_of_basis(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    '''
    Rewrites 3D points from the camera coordinate frame to a reference coordinate frame.
    T: 4x4 transformation matrix from the camera to the reference frame
    points: Nx3 or homogenous Nx4 array of points in the camera coordinate frame # TODO most dims are given trannsposed
    '''

    if points.ndim != 2 or points.shape[0] < 3 or points.shape[0] > 4:
        raise ValueError("Points must be a Nx3 or Nx4 array.")
    if points.shape[0] == 3:
        points_homogeneous = np.hstack((points, np.ones((points.shape[1], 1))))  # Convert to homogeneous coordinates
        new_point_homogeneous = T @ points_homogeneous
        new_point = new_point_homogeneous[:3, :]  # Convert back to 3D coordinates
        return new_point
    else:
        new_point_homogeneous = T @ points
        return new_point_homogeneous



def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """ Returns the unit vector of the vector. """
    if vector.shape[0] == 4:
        vector_in = vector[:3].flatten()  # Ignore the homogeneous coordinate if present
        unit_vector = vector_in / np.linalg.norm(vector_in)
        unit_vector = np.array([[unit_vector[0]], [unit_vector[1]], [unit_vector[2]], [1.0]])  # Return as a 4D vector with homogeneous coordinate 1
    else:
        unit_vector = vector / np.linalg.norm(vector)
    
    return unit_vector


def _find_vector_by_angles(a: np.ndarray, b: np.ndarray, angle_a: float, angle_b: float) -> np.ndarray:
    """
    Finds a 3D unit vector that forms angle_a with vector 'a' 
    and angle_b with vector 'b'.
    Angles must be provided in radians.
    """
    # 1. Normalize reference vectors
    a_u = a / np.linalg.norm(a)
    b_u = b / np.linalg.norm(b)
    
    # 2. Build an orthonormal basis (Gram-Schmidt)
    u1 = a_u
    
    # Check if a and b are collinear
    dot_ab = np.dot(u1, b_u)
    if np.abs(dot_ab) > 0.999999:
        raise ValueError("Reference vectors are collinear; infinite solutions or no solution.")
        
    # Second basis vector in the plane of a and b
    u2 = b_u - dot_ab * u1
    u2 = u2 / np.linalg.norm(u2)
    
    # Third basis vector perpendicular to the plane
    u3 = np.cross(u1, u2)
    
    # 3. Solve for coordinates in the new basis (x', y', z')
    # Equation 1: v_prime[0] = cos(angle_a)
    x_prime = np.cos(angle_a)
    
    # Equation 2: v . b_u = cos(angle_b)
    # Expressing b_u in the new basis: b_u = dot_ab * u1 + sin_ab * u2
    # Where sin_ab = sqrt(1 - dot_ab^2)
    sin_ab = np.sqrt(1 - dot_ab**2)
    y_prime = (np.cos(angle_b) - x_prime * dot_ab) / sin_ab
    
    # Equation 3: Unit length condition -> x'^2 + y'^2 + z'^2 = 1
    z_prime_sq = 1.0 - x_prime**2 - y_prime**2
    
    #print(f"Finding vector by angles: angle_a={np.rad2deg(angle_a)}, angle_b={np.rad2deg(angle_b)}, x'={x_prime}, y'={y_prime}, z'^2={z_prime_sq}")
    if z_prime_sq < -1e-7:
        #raise ValueError("No solution exists for these angles with the given vectors. (sum of both angles > angle between a and b)")
        # Patch: Reduce angles slightly to find a solution. Ideally this should not happen and if it does, it indicates that some aspects of the calibration are off.
        angle_a = max(0, angle_a - .01)
        angle_b = max(0, angle_b - .01)
        return _find_vector_by_angles(a, b, angle_a, angle_b)
    elif np.abs(z_prime_sq) < 1e-7:
        # One unique solution (the vector lies exactly in the plane of a and b)
        z_prime_possibilities = [0.0]
    else:
        # Two solutions (mirrored across the plane of a and b)
        z_prime = np.sqrt(z_prime_sq)
        z_prime_possibilities = [z_prime, -z_prime]
        
    # 4. Transform coordinates back to the global system
    solutions = []
    for z_p in z_prime_possibilities:
        v = x_prime * u1 + y_prime * u2 + z_p * u3
        solutions.append(v)
        
    return solutions



def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    '''Returns the angle (in radians) between two 3D vectors.'''
    
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return angle


def _triangulate_point(stereo_calibration_results: StereoCalibrationResults, point_coords: np.ndarray) -> tuple:
    '''
    Triangulates a 3D point from its 2D projections in two camera images.
    points: 2D coordinates of the point in each camera image (list of two tuples/lists) [[x1, y1], [x2, y2]]
    '''

    cam0_f_mm = stereo_calibration_results.camera_params_0.intrinsic.f_mm
    cam0_px_mm = stereo_calibration_results.camera_params_0.intrinsic.px_mm
    cam0_py_mm = stereo_calibration_results.camera_params_0.intrinsic.py_mm
    cam0_nx = stereo_calibration_results.camera_params_0.intrinsic.nx
    cam0_ny = stereo_calibration_results.camera_params_0.intrinsic.ny
    cam0_K = stereo_calibration_results.camera_params_0.intrinsic.K
    cam0_dist = stereo_calibration_results.camera_params_0.intrinsic.dist

    cam1_f_mm = stereo_calibration_results.camera_params_1.intrinsic.f_mm
    cam1_px_mm = stereo_calibration_results.camera_params_1.intrinsic.px_mm
    cam1_py_mm = stereo_calibration_results.camera_params_1.intrinsic.py_mm
    cam1_nx = stereo_calibration_results.camera_params_1.intrinsic.nx
    cam1_ny = stereo_calibration_results.camera_params_1.intrinsic.ny
    cam1_K = stereo_calibration_results.camera_params_1.intrinsic.K
    cam1_dist = stereo_calibration_results.camera_params_1.intrinsic.dist

    T_R = stereo_calibration_results.camera_params_1.extrinsic.relative.T_R

    # Undistort points and convert to camera coordinates (mm)  (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
    P1, roi1 = cv2.getOptimalNewCameraMatrix(cam0_K, cam0_dist, (int(cam0_nx), int(cam0_ny)), 1, (int(cam0_nx), int(cam0_ny)))
    P2, roi2 = cv2.getOptimalNewCameraMatrix(cam1_K, cam1_dist, (int(cam1_nx), int(cam1_ny)), 1, (int(cam1_nx), int(cam1_ny)))

    point_coords_0 = cv2.undistortPoints(np.array([point_coords[0]], dtype=np.float32), cam0_K, cam0_dist, R=None, P=P1)[0][0]
    point_coords_1 = cv2.undistortPoints(np.array([point_coords[1]], dtype=np.float32), cam1_K, cam1_dist, R=None, P=P2)[0][0]
    
    #print(f"[dbg] original point coords: {point_coords[0]}, {point_coords[1]}")
    #print(f"[dbg] undistorted point coords: {point_coords_0}, {point_coords_1}")

    cam0_point_vector_cam_coords = np.array([[ (point_coords_0[0] - cam0_nx/2) * cam0_px_mm ],
                                        [ (point_coords_0[1] - cam0_ny/2) * cam0_py_mm ],
                                        [ cam0_f_mm ],
                                        [ 1.0 ]])
    cam1_point_vector_cam_coords = np.array([[ (point_coords_1[0] - cam1_nx/2) * cam1_px_mm ],
                                        [ (point_coords_1[1] - cam1_ny/2) * cam1_py_mm ],
                                        [ cam1_f_mm ],
                                        [ 1.0 ]])

    cam0_point_vector_relative = _unit_vector(cam0_point_vector_cam_coords)
    cam1_point_vector_relative = _unit_vector(T_R @ cam1_point_vector_cam_coords)
    cam_point_vectors_relative = np.array([cam0_point_vector_relative, cam1_point_vector_relative]).T.reshape(4, 2)
    
    cam0_point_vector_relative_flat = cam0_point_vector_relative[:3].flatten()
    cam1_point_vector_relative_flat = cam1_point_vector_relative[:3].flatten()
    cam_point_vectors_relative_flat = np.array([cam0_point_vector_relative_flat, cam1_point_vector_relative_flat])
    
    point_wc, projection_error = _lsq_intersection_of_lines(np.array([stereo_calibration_results.camera_params_0.extrinsic.relative.origin[:3, 0], stereo_calibration_results.camera_params_1.extrinsic.relative.origin[:3, 0]]), cam_point_vectors_relative_flat)
    
    # convert point to homogenous column vectors
    point_wc = np.array([[point_wc[0,0]], [point_wc[1,0]], [point_wc[2,0]], [1.0]])

    return point_wc, cam_point_vectors_relative, projection_error



def _reflect_point_around_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    '''
    Mirror point along plane defined by normal
    point: 3D point to be mirrored (1x3/1x4)
    plane_point: 3D point on the plane (1x3/1x4)
    plane_normal: 3D normal vector of the plane (1x3/1x4)
    '''
    point_vec = point[:3] - plane_point[:3]
    distance_to_plane = np.dot(point_vec.flatten(), plane_normal[:3].flatten())
    point_ref = point[:3] - (2 * distance_to_plane * plane_normal[:3])
    return np.array([[point_ref[0,0]], [point_ref[1,0]], [point_ref[2,0]], [1.0]])



def _triangulate_mirrored_point(stereo_calibration_results: StereoCalibrationResults, point_coords: np.ndarray, mirror_surface_point_wc: np.ndarray, mirror_surface_normal_wc: np.ndarray):
    point_wc_mirrored, cam_point_vectors, projection_error = _triangulate_point(stereo_calibration_results, point_coords)
    point_wc = _reflect_point_around_plane(point_wc_mirrored, mirror_surface_point_wc, mirror_surface_normal_wc)
    return point_wc, point_wc_mirrored, cam_point_vectors, projection_error


def _reconstruct_mirror_surface_from_pattern(stereo_calibration_results: StereoCalibrationResults, imgs: List[np.ndarray], n_pattern_corners: Tuple[int, int]):
    # Detect corners in calibration pattern
    corners = [detect_corners(img, n_pattern_corners, verbose=True) for img in imgs]

    if np.any([corner is None for corner in corners]):
        raise ValueError("Could not detect corners in one of the images. Please check the calibration pattern and image quality.")

    # Reconstruct surface normal from three corners
    p1, _, _ = _triangulate_point(stereo_calibration_results, (corners[0][0], corners[1][0]))
    p2, _, _ = _triangulate_point(stereo_calibration_results, (corners[0][n_pattern_corners[0]-1], corners[1][n_pattern_corners[0]-1]))
    p3, _, _ = _triangulate_point(stereo_calibration_results, (corners[0][n_pattern_corners[0]*(n_pattern_corners[1]-1)], corners[1][n_pattern_corners[0]*(n_pattern_corners[1]-1)]))

    p1_flat = p1[:3].flatten()
    p2_flat = p2[:3].flatten()
    p3_flat = p3[:3].flatten()

    mirror_surface_point_wc = p1
    mirror_surface_normal_wc = _unit_vector(np.cross(p2_flat - p1_flat, p3_flat - p1_flat))
    mirror_surface_normal_wc = np.array([[mirror_surface_normal_wc[0,0]], [mirror_surface_normal_wc[1,0]], [mirror_surface_normal_wc[2,0]], [1.0]])

    return mirror_surface_point_wc, mirror_surface_normal_wc



def _reconstruct_mirror_surface_from_camera_labels(stereo_calibration_results: StereoCalibrationResults, camera_label_pair: Tuple[np.ndarray, np.ndarray], cam_origin: np.ndarray):
    cam1_proj, _, _ = _triangulate_point(stereo_calibration_results, (camera_label_pair[0], camera_label_pair[1]))

    mirror_surface_point_wc = cam_origin + ((cam1_proj - cam_origin) / 2)
    mirror_surface_normal_wc = _unit_vector(cam1_proj - cam_origin)
    mirror_surface_normal_wc = np.array([[mirror_surface_normal_wc[0,0]], [mirror_surface_normal_wc[1,0]], [mirror_surface_normal_wc[2,0]], [1.0]])
    return mirror_surface_point_wc, mirror_surface_normal_wc


# Prozedere LFI Geocal x Referenzsystem
# Mehrere Views mit externer IDS-Kamera (kalibriert) und Eye-Cameras (kalibriert) mit weißer Target-Ebene mit bekannten Punkten, z.B. Checkerboard.
# Checkerboard -> Mono-Platzierung der Ebene in 3D möglich
# IDS-Kamera: Labeln der Bildpunkte auf der Ebene
# Transformation der Punkte zu Eye-Cameras
# Mehrere Views -> Lösunngslinnien für die Laser, idealerweise Schnittpunkt der Linien in 3D wo die LFI-Sensoren liegen
# Außerdem: Labeln der LFI-Sennsoren via Stereo-Verfahren.

