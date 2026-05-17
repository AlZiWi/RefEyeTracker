from stereo_calib_opencv import *
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, print_func=print):
        self.print_func = print_func

    def msg(self, msg: str, src: str = None, level: str = "info"):
        levels = {"info": 0, "warning": 1, "error": 2}

        if level not in levels:
            raise ValueError(f"Unknown log level: {level}")

        if src is not None:
            self.print_func(f"[{level.upper()}] [{src}] {msg}")
        else:
            self.print_func(f"[{level.upper()}] {msg}")


def lsq_intersection_of_lines(point_vecs, normal_vecs):
    # TODO idea: use weighted sum depending on ellipse fit quality etc.?
    
    vec_len = len(point_vecs[0])
    
    S1 = np.matrix(np.zeros((vec_len, vec_len)))
    S2 = np.matrix(np.zeros((vec_len, 1)))
    
    dsts = []
    
    for i in range(len(normal_vecs)):
        nv = np.matrix(normal_vecs[i]).T
        pv = np.matrix(point_vecs[i]).T
        
        nv = unit_vector(nv)
        
        s1 = np.identity(vec_len) - (nv * nv.T)
        s2 = (np.identity(vec_len) - (nv * nv.T)) * pv
        
        S1 += s1
        S2 += s2

    c_proj = np.linalg.pinv(S1) * S2
    
    dsts = []
    for i in range(len(normal_vecs)):
        nv = np.matrix(normal_vecs[i]).T
        pv = np.matrix(point_vecs[i]).T
        nv = unit_vector(nv)
        
        dsts.append(np.linalg.norm(np.cross(nv.A1, pv.A1 - c_proj.A1)))
        
    print(f"[dbg] dsts: {dsts}")
    projection_error = np.sum(np.array(dsts))
    
    return c_proj.A1, projection_error


def project_points_to_coordinate_frame(points, T):
    # T = transformation matrix from the camera to the reference frame 
    
    projected_points = []

    for point in points:
        point_homogeneous = np.matrix(np.array([[point[0]], [point[1]], [point[2]], [1]]))
        projected_point_homogeneous = T * point_homogeneous
        projected_points.append(projected_point_homogeneous[:3].A1)

    return projected_points


def transformation_matrix_from_calib_ext(calib_data_ext):
    R = calib_data_ext.R_wc
    t = calib_data_ext.t_wc
    T = np.matrix(np.eye(4))
    T[:3, 3] = t
    T[:3, :3] = R
    
    return T


def calculate_pairwise_camera_calib_vecs(calib_data, calib_data_offset=None):
    # calib_ext_in = output of this function for the first camera, to be used as absolute reference frame instead of relative
    T = transformation_matrix_from_calib_ext(calib_data["camera_params_1"].extrinsic)
    R = T[:3, :3]

    cam_origin = np.matrix(np.array([[0],[0],[0],[1]]))  # left camera (from camera POV)
    cam_z_origin = np.matrix(np.array([[0],[0],[1]]))  # Vector pointing towards image plane (OpenCV Z axis)
    cam_y_origin = np.matrix(np.array([[0],[1],[0]]))  # Vector pointing in direction of camera bottom (cable, OpenCV Y axis)

    cam_pair_calib = {
        0: {"relative": {}, "absolute": {}},
        1: {"relative": {}, "absolute": {}},
    }

    cam_pair_calib[0]["relative"] = {'origin': cam_origin,
                'z': cam_z_origin,
                'y': cam_y_origin}
    cam_pair_calib[1]["relative"] = {'origin': T * cam_origin,
                'z': R * cam_z_origin,
                'y': R * cam_y_origin}
        
    if calib_data_offset is not None:
        T_offset = transformation_matrix_from_calib_ext(calib_data_offset["camera_params_1"].extrinsic)
        R_offset = T_offset[:3, :3]
        
        cam_pair_calib[0]["absolute"] = {'origin': T_offset * cam_origin,
                'z': R_offset * cam_z_origin,
                'y': R_offset * cam_y_origin}
        cam_pair_calib[1]["absolute"] = {'origin': T_offset * T * cam_origin,
            'z': R_offset * R * cam_z_origin,
            'y': R_offset * R * cam_y_origin}
    else:
        cam_pair_calib[0]["absolute"] = cam_pair_calib[0]["relative"]
        cam_pair_calib[1]["absolute"] = cam_pair_calib[1]["relative"]

    return cam_pair_calib


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3).A1, (vec2 / np.linalg.norm(vec2)).reshape(3).A1
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)  # no rotation needed
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


def undistort_points(points, K, dist):
    # Undistort points using OpenCV (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
    P, roi = cv2.getOptimalNewCameraMatrix(K, dist, (int(K[0,2]*2), int(K[1,2]*2)), 1, (int(K[0,2]*2), int(K[1,2]*2)))
    undistorted_points = cv2.undistortPoints(np.array(points), K, dist, R=None, P=P)
    return undistorted_points


def triangulate_point(calib_data, point_coords):
    # points: point coordinates in each camera image ((x1, y1), (x2, y2))

    cam1_f_mm = calib_data["camera_params_0"].intrinsic.f_mm
    cam1_px_mm = calib_data["camera_params_0"].intrinsic.px_mm
    cam1_py_mm = calib_data["camera_params_0"].intrinsic.py_mm
    cam1_nx = calib_data["camera_params_0"].intrinsic.nx
    cam1_ny = calib_data["camera_params_0"].intrinsic.ny
    cam1_K = calib_data["camera_params_0"].intrinsic.K
    cam1_dist = calib_data["camera_params_0"].intrinsic.dist

    cam2_f_mm = calib_data["camera_params_1"].intrinsic.f_mm
    cam2_px_mm = calib_data["camera_params_1"].intrinsic.px_mm
    cam2_py_mm = calib_data["camera_params_1"].intrinsic.py_mm
    cam2_nx = calib_data["camera_params_1"].intrinsic.nx
    cam2_ny = calib_data["camera_params_1"].intrinsic.ny
    cam2_K = calib_data["camera_params_1"].intrinsic.K
    cam2_dist = calib_data["camera_params_1"].intrinsic.dist

    R = calib_data["camera_params_1"].extrinsic.R_wc
    t = calib_data["camera_params_1"].extrinsic.t_wc

    cam_pair_calib_vecs = calculate_pairwise_camera_calib_vecs(calib_data)
    
    # Undistort points and convert to camera coordinates (mm)  (https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
    P1, roi1 = cv2.getOptimalNewCameraMatrix(cam1_K, cam1_dist, (int(cam1_nx), int(cam1_ny)), 1, (int(cam1_nx), int(cam1_ny)))
    P2, roi2 = cv2.getOptimalNewCameraMatrix(cam2_K, cam2_dist, (int(cam2_nx), int(cam2_ny)), 1, (int(cam2_nx), int(cam2_ny)))

    point_coords_0 = cv2.undistortPoints(np.array([point_coords[0]]), cam1_K, cam1_dist, R=None, P=P1)[0][0]
    point_coords_1 = cv2.undistortPoints(np.array([point_coords[1]]), cam2_K, cam2_dist, R=None, P=P2)[0][0]
    
    #print(f"[dbg] original point coords: {point_coords[0]}, {point_coords[1]}")
    #print(f"[dbg] undistorted point coords: {point_coords_0}, {point_coords_1}")

    cam1_point_vector_cam_coords = np.matrix(np.array([[ (point_coords_0[0] - cam1_nx/2) * cam1_px_mm ],
                                        [ (point_coords_0[1] - cam1_ny/2) * cam1_py_mm ],
                                        [ cam1_f_mm ]]))
    cam2_point_vector_cam_coords = np.matrix(np.array([[ (point_coords_1[0] - cam2_nx/2) * cam2_px_mm ],
                                        [ (point_coords_1[1] - cam2_ny/2) * cam2_py_mm ],
                                        [ cam2_f_mm ]]))

    cam1_point_vector_wc = unit_vector(cam1_point_vector_cam_coords)
    cam2_point_vector_wc = unit_vector(R * cam2_point_vector_cam_coords)

    point_wc, projection_error = lsq_intersection_of_lines((cam_pair_calib_vecs[0]['relative']['origin'][:3].A1, cam_pair_calib_vecs[1]['relative']['origin'][:3].A1), (cam1_point_vector_wc.A1, cam2_point_vector_wc.A1))

    return point_wc, (cam1_point_vector_wc, cam2_point_vector_wc), projection_error


def reflect_point_around_plane(point, plane_point, plane_normal):
    # mirror point along plane defined by normal
    # Takes vectors as .A1 arrays, not homogeneous coordinates
    point_vec = point - plane_point
    distance_to_plane = np.dot(point_vec, plane_normal)
    point_ref = point - 2 * distance_to_plane * plane_normal
    return point_ref


def triangulate_mirrored_point(calib_data, point_coords, mirror_surface_point_wc, mirror_surface_normal_wc):
    point_wc_mirrored, cam_point_vectors, projection_error = triangulate_point(calib_data, point_coords)
    point_wc = reflect_point_around_plane(point_wc_mirrored, mirror_surface_point_wc, mirror_surface_normal_wc)
    return point_wc, point_wc_mirrored, cam_point_vectors, projection_error


def reconstruct_mirror_surface_from_pattern(calib_data, imgs, pattern_corners):
    # Detect corners in calibration pattern
    corners = [detect_corners(img, pattern_corners, verbose=True) for img in imgs]

    if np.any([corner is None for corner in corners]):
        raise ValueError("Could not detect corners in one of the images. Please check the calibration pattern and image quality.")

    # Reconstruct surface normal from three corners
    p1, _, _ = triangulate_point(calib_data, (corners[0][0], corners[1][0]))
    p2, _, _ = triangulate_point(calib_data, (corners[0][pattern_corners[0]-1], corners[1][pattern_corners[0]-1]))
    p3, _, _ = triangulate_point(calib_data, (corners[0][pattern_corners[0]*(pattern_corners[1]-1)], corners[1][pattern_corners[0]*(pattern_corners[1]-1)]))

    mirror_surface_point_wc = p1
    mirror_surface_normal_wc = unit_vector(np.cross(p2 - p1, p3 - p1))

    return mirror_surface_point_wc, mirror_surface_normal_wc


def reconstruct_mirror_surface_from_camera_labels(calib_data, camera_label_pair, cam_origin):
    cam1_proj, _, projection_error_1 = triangulate_point(calib_data, (camera_label_pair[0], camera_label_pair[1]))
    
    mirror_surface_point_wc = cam_origin + ((cam1_proj - cam_origin) / 2)
    mirror_surface_normal_wc = unit_vector(cam1_proj - cam_origin)

    return mirror_surface_point_wc, mirror_surface_normal_wc

def triangulate_mirrored_points(calib_data, imgs, points_label_pairs={}, camera_label_pairs={}, pattern_corners=None):
    mirror_calibration_results = {}
    
    if pattern_corners is not None:
        try:
            mirror_surface_point_wc, mirror_surface_normal_wc = reconstruct_mirror_surface_from_pattern(calib_data, imgs, pattern_corners)
            mirror_calibration_results["pattern"] = {
                "surface_point": mirror_surface_point_wc,
                "surface_normal": mirror_surface_normal_wc
            }
        except Exception as e:
            print(f"[warning] Could not reconstruct mirror surface from pattern: {e}")

    cam_pair_calib_vecs = calculate_pairwise_camera_calib_vecs(calib_data)

    # TODO can also be called CAM_0 and CAM_1, direction doesn't matter.
    if "CAM_0" in camera_label_pairs.keys():
        cam_orig = cam_pair_calib_vecs[0]['relative']['origin'][:3].A1
        mirror_surface_point_wc, mirror_surface_normal_wc = reconstruct_mirror_surface_from_camera_labels(calib_data, camera_label_pairs["CAM_0"], cam_orig)
        mirror_calibration_results["CAM_0"] = {
            "surface_point": mirror_surface_point_wc,
            "surface_normal": mirror_surface_normal_wc
        }

    if "CAM_1" in camera_label_pairs.keys():
        cam_orig = cam_pair_calib_vecs[1]['relative']['origin'][:3].A1
        mirror_surface_point_wc, mirror_surface_normal_wc = reconstruct_mirror_surface_from_camera_labels(calib_data, camera_label_pairs["CAM_1"], cam_orig)
        mirror_calibration_results["CAM_1"] = {
            "surface_point": mirror_surface_point_wc,
            "surface_normal": mirror_surface_normal_wc
        }

    if "CAM_1" in mirror_calibration_results.keys() and "CAM_0" in mirror_calibration_results.keys():
        mirror_calibration_results["CAM_BOTH"] = {
            "surface_point": np.mean([mirror_calibration_results["CAM_0"]["surface_point"], mirror_calibration_results["CAM_1"]["surface_point"]], axis=0),
            "surface_normal": np.mean([mirror_calibration_results["CAM_0"]["surface_normal"], mirror_calibration_results["CAM_1"]["surface_normal"]], axis=0)
        }

    # Triangulate points
    points_calibration_results_for_mirror_calibs = {}
    for mirror_calib_key, mirror_calib in mirror_calibration_results.items():
        points_calibration_results = {}
        for point_label, point_coords in points_label_pairs.items():
            pt_wc, pt_wc_mirrored, cam_point_vectors, projection_error = triangulate_mirrored_point(calib_data, point_coords, mirror_calib["surface_point"], mirror_calib["surface_normal"])

            points_calibration_results[point_label] = {
                "point_world": pt_wc,
                "point_mirrored": pt_wc_mirrored,  # virtual point behind mirror surface
                "cam_point_vectors": cam_point_vectors,
                "projection_error": projection_error
            }

        points_calibration_results_for_mirror_calibs[mirror_calib_key] = points_calibration_results

    return points_calibration_results_for_mirror_calibs, mirror_calibration_results
