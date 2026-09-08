from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class CameraIndex(Enum):
    RO = "ro"
    RI = "ri"
    LO = "lo"
    LI = "li"
    SC = "sc"

class CameraType(Enum):
    IP = "IP"
    USB = "USB"
    
class CalibrationPatterns(Enum):
    def __init__(self, corners: tuple[int, int], square_size_mm: float):
        self.corners: tuple[int, int] = corners  # inner corners per chessboard row and column
        self.square_size_mm: float = square_size_mm

    SMALL = (10, 7), 3.0 
    LARGE = (10, 7), 6.0
    
class UIPreset(Enum):
    CAPTURE = "CAPTURE"
    STEREO_RR = "STEREO_RR"
    STEREO_LL = "STEREO_LL"
    STEREO_RL = "STEREO_RL"
    MONO_SC = "MONO_SC"
    MIRROR_R = "MIRROR_R"
    MIRROR_L = "MIRROR_L"
    MIRROR_SC = "MIRROR_SC"
    FULL_CALIBRATION = "FULL_CALIBRATION"


class CalibrationLabel(Enum):
    LED_0 = "0"
    LED_1 = "1"
    LED_2 = "2"
    LED_3 = "3"
    CAM_0 = "CAM_0"
    CAM_1 = "CAM_1"
    CAM_RO = "CAM_RO"
    CAM_RI = "CAM_RI"
    CAM_LO = "CAM_LO"
    CAM_LI = "CAM_LI"
    CAM_SC = "CAM_SC"
    

type LabelCoordinates = dict[CalibrationLabel, np.ndarray]  # CalibrationLabel: [camera_index, coord]


@dataclass
class CalibrationLabelInfo:
    label_coordinates: LabelCoordinates


class CalibrationLabelSet(Enum):
    def __init__(self, labels: list[CalibrationLabel], source_camera_labels: list[CalibrationLabel]):
        self.labels: list[CalibrationLabel] = labels
        self.source_camera_labels: list[CalibrationLabel] = source_camera_labels  # Cameras used to reconstruct the mirror position

    MIRROR_STEREO = [CalibrationLabel.LED_0, CalibrationLabel.LED_1, CalibrationLabel.LED_2, CalibrationLabel.LED_3, CalibrationLabel.CAM_0, CalibrationLabel.CAM_1, CalibrationLabel.CAM_SC], [CalibrationLabel.CAM_0, CalibrationLabel.CAM_1]
    MIRROR_SC = [CalibrationLabel.CAM_SC, CalibrationLabel.CAM_RO, CalibrationLabel.CAM_RI, CalibrationLabel.CAM_LO, CalibrationLabel.CAM_LI], []


@dataclass
class UIPresetConfig:
    @dataclass
    class CapturePreset:
        use_threshold: bool
        auto_accept: bool
        auto_contrast: bool

    @dataclass
    class CalibrationPreset:
        calib_labels: CalibrationLabelSet | None
        calib_dependencies: list[UIPreset] | None
        
    description: str
    camera_indices: list[CameraIndex]
    folder_path: str
    use_pattern: CalibrationPatterns | None
    capture_settings: CapturePreset | None
    calib_settings: CalibrationPreset | None
    
RECORDING_SUBFOLDER_FORMAT = "exp__%Y-%m-%d_%H-%M-%S"


CAPTURE_PRESET_CONFIGS = {
    UIPreset.CAPTURE: UIPresetConfig(
        description="Default capture mode",
        camera_indices=[CameraIndex.RO, CameraIndex.RI, CameraIndex.LO, CameraIndex.LI, CameraIndex.SC],
        folder_path="recording",
        use_pattern=None,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=False,
            auto_accept=False,
            auto_contrast=False
        ),
        calib_settings=None
    ),
    UIPreset.STEREO_RR: UIPresetConfig(
        description="Capture stereo calibration images from the right camera pair.",
        camera_indices=[CameraIndex.RO, CameraIndex.RI],
        folder_path="calibration/stereo_ro_ri",
        use_pattern=CalibrationPatterns.SMALL,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=True,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=None,
            calib_dependencies=None
        )
    ),
    UIPreset.STEREO_LL: UIPresetConfig(
        description="Capture stereo calibration images from the left camera pair.",
        camera_indices=[CameraIndex.LO, CameraIndex.LI],
        folder_path="calibration/stereo_lo_li",
        use_pattern=CalibrationPatterns.SMALL,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=True,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=None,
            calib_dependencies=None
        )
    ),
    UIPreset.STEREO_RL: UIPresetConfig(
        description="Capture stereo calibration images from the outer camera pair.",
        camera_indices=[CameraIndex.RO, CameraIndex.LO],
        folder_path="calibration/stereo_ro_lo",
        use_pattern=CalibrationPatterns.LARGE,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=True,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=None,
            calib_dependencies=None
        )
    ),
    UIPreset.MONO_SC: UIPresetConfig(
        description="Capture calibration images from the scene camera.",
        camera_indices=[CameraIndex.SC],
        folder_path="calibration/mono_sc",
        use_pattern=CalibrationPatterns.LARGE,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=True,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=None,
            calib_dependencies=None
        )
    ),
    UIPreset.MIRROR_R: UIPresetConfig(
        description="Capture mirror calibration images with the right camera pair, LEDs and scene camera visible.",
        camera_indices=[CameraIndex.RO, CameraIndex.RI],
        folder_path="calibration/mirror_r",
        use_pattern=None,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=False,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=CalibrationLabelSet.MIRROR_STEREO,
            calib_dependencies=[UIPreset.STEREO_RR]
        )
    ),
    UIPreset.MIRROR_L: UIPresetConfig(
        description="Capture mirror calibration images with the left camera pair, LEDs and scene camera visible.",
        camera_indices=[CameraIndex.LO, CameraIndex.LI],
        folder_path="calibration/mirror_l",
        use_pattern=None,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=False,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=CalibrationLabelSet.MIRROR_STEREO,
            calib_dependencies=[UIPreset.STEREO_LL]
        )
    ),
    UIPreset.MIRROR_SC: UIPresetConfig(
        description="Capture mirror calibration images with the scene camera, LEDs and both camera pairs visible.",
        camera_indices=[CameraIndex.SC],
        folder_path="calibration/mirror_sc",
        use_pattern=CalibrationPatterns.SMALL,
        capture_settings=UIPresetConfig.CapturePreset(
            use_threshold=True,
            auto_accept=False,
            auto_contrast=True
        ),
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=CalibrationLabelSet.MIRROR_SC,
            calib_dependencies=[UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.MONO_SC]
        )
    ),
    UIPreset.FULL_CALIBRATION: UIPresetConfig(
        description="Save all calibration results to a single file for later use.",
        camera_indices=None,
        folder_path="calibration/full",
        use_pattern=None,
        capture_settings=None,
        calib_settings=UIPresetConfig.CalibrationPreset(
            calib_labels=None,
            calib_dependencies=[UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.STEREO_RL, UIPreset.MONO_SC, UIPreset.MIRROR_R, UIPreset.MIRROR_L, UIPreset.MIRROR_SC]
        )
    )
}
    
    
# Stereo


@dataclass
class MonoReprojectionErrors:
    @dataclass
    class PerViewMonoReprojectionError:
        index: int
        mean_px: float
        std_px: float
        max_px: float
        num_pts: int
        
    mean_px: Optional[float] = field(default=None)
    std_px: Optional[float] = field(default=None)
    max_px: Optional[float] = field(default=None)
    per_view: Optional[List[PerViewMonoReprojectionError]] = field(default=None)
    all_errs: Optional[np.ndarray] = field(default=None)
    all_errs_2d: Optional[np.ndarray] = field(default=None)

@dataclass
class MonoCalibrationStatistics:
    errs_mono_reproj: Optional[dict[str, float]] = field(default=None)
    errs_mono_reproj_initial: Optional[dict[str, float]] = field(default=None)
    rms: Optional[float] = field(default=None)
    rvecs: Optional[list[np.ndarray]] = field(default=None)
    tvecs: Optional[list[np.ndarray]] = field(default=None)
    num_images: Optional[int] = field(default=None)

@dataclass
class CameraParamsIntrinsic:
    K: np.ndarray           # 3x3 intrinsic matrix (pixel units)
    dist: np.ndarray        # distortion coefficients (k1, k2, p1, p2, k3)
    map1: np.ndarray        # distortion maxtrix 1 (for cv2.remap)
    map2: np.ndarray        # distortion maxtrix 2 (for cv2.remap)
    f_mm: float             # focal length in mm (average of fx*px_mm and fy*py_mm)
    nx: int                 # number of pixels in x direction
    ny: int                 # number of pixels in y direction
    px_mm: float            # pixel size/pitch in mm (x direction)
    py_mm: float            # pixel size/pitch in mm (y direction)
    W_mm: float             # sensor width in mm (nx * px_mm)
    H_mm: float             # sensor height in mm (ny * py_mm)
    cx: float               # x coordinate of the principal point
    cy: float               # y coordinate of the principal point
    statistics: Optional[MonoCalibrationStatistics] = field(default=None)

@dataclass
class CameraCoordinateFrame:
    T: np.ndarray           # 4x4 transformation matrix world coordinates to camera coordinates
    T_R: np.ndarray         # 4x4 transformation matrix world coordinates to camera coordinates rotational part only (3x3 in upper left)
    T_t: np.ndarray         # 4x4 transformation matrix world coordinates to camera coordinates translational part only (3x1 in upper right)
    # Helper vectors for convenience (redundant with T/R/t)
    origin: Optional[np.ndarray] = field(default=None)      # 3D coordinates of the camera origin in world coordinates. Homogenous coordinates (4x1) with last element 1.0
    x: Optional[np.ndarray] = field(default=None)           # 3D coordinates of the camera x-axis in world coordinates. Homogenous coordinates (4x1) with last element 1.0
    y: Optional[np.ndarray] = field(default=None)           # 3D coordinates of the camera y-axis in world coordinates. Homogenous coordinates (4x1) with last element 1.0
    z: Optional[np.ndarray] = field(default=None)           # 3D coordinates of the camera z-axis in world coordinates. Homogenous coordinates (4x1) with last element 1.0

@dataclass
class CameraParamsExtrinsic:
    relative: CameraCoordinateFrame  # In camera coordinates, relative to the first camera (otherwise default cartesian base)
    absolute: Optional[CameraCoordinateFrame] = field(default=None)  # In world coordinates, requires definition of offset

@dataclass
class CameraParams:
    intrinsic: CameraParamsIntrinsic
    extrinsic: CameraParamsExtrinsic
    
@dataclass
class StereoCalibrationResults:
    camera_params_0: CameraParams
    camera_params_1: CameraParams
    
    R: np.ndarray
    t: np.ndarray
    E: np.ndarray
    F: np.ndarray
    
    
### Calibration


class MirrorCalibrationType(Enum):
    CAM_BOTH = "CAM_BOTH"
    CAM_0 = "CAM_0"
    CAM_1 = "CAM_1"
    PATTERN = "PATTERN"
    
@dataclass
class MirrorCalibrationResultsForType:
    @dataclass
    class MirrorCalibrationResultsForPoint:
        point_world: np.array
        point_mirrored: np.array  # virtual point behind mirror surface
        cam_point_vectors: np.ndarray  # unit vectors from camera origins to the point in world coordinates
        projection_error: float
    
    surface_point: np.array
    surface_normal: np.array
    img_paths: Tuple[Path, Path]
    calibrated_points: dict[CalibrationLabel, MirrorCalibrationResultsForPoint] = field(default_factory=dict)

type MirrorCalibrationResults = dict[MirrorCalibrationType, MirrorCalibrationResultsForType]

@dataclass 
class SceneCameraExtrinsicsCalibrationResults:
    @dataclass
    class MirrorPose:
        T_mirror: np.ndarray
        rvec_mirror: np.ndarray
        tvec_mirror: np.ndarray
    
    @dataclass
    class LabelPoints:
        image_points_2d_mirror: np.ndarray
        image_points_2d_cameras: np.ndarray
        image_point_2d_sc: np.array
        
        image_points_2d_mirror_undist: np.ndarray
        image_points_2d_cameras_undist: np.ndarray
        image_point_2d_sc_undist: np.array
        
        world_points_3d_mirror: np.ndarray
        world_points_3d_frames: np.ndarray
    
    camera_params: CameraParams
    
    mirror_pose: MirrorPose
    points: LabelPoints
    scaling_factor: float
    distance_mirror_to_sc: float
    possible_camera_positions: List[np.ndarray]
    img_path: Path

@dataclass
class CalibrationSummary:
    CAM_RO: Optional[CameraParams] = field(default=None)
    CAM_RI: Optional[CameraParams] = field(default=None)
    CAM_LO: Optional[CameraParams] = field(default=None)
    CAM_LI: Optional[CameraParams] = field(default=None)
    CAM_SC: Optional[CameraParams] = field(default=None)
    POINTS_R: Optional[MirrorCalibrationResultsForType] = field(default=None)
    POINTS_L: Optional[MirrorCalibrationResultsForType] = field(default=None)
    
    @dataclass
    class IntermediateCalibrationResults:
        STEREO_RR: Optional[StereoCalibrationResults] = field(default=None)
        STEREO_LL: Optional[StereoCalibrationResults] = field(default=None)
        STEREO_RL: Optional[StereoCalibrationResults] = field(default=None)
        MIRROR_R: Optional[MirrorCalibrationResults] = field(default=None)
        MIRROR_L: Optional[MirrorCalibrationResults] = field(default=None)
        MONO_SC: Optional[CameraParamsIntrinsic] = field(default=None)
        MIRROR_SC: Optional[SceneCameraExtrinsicsCalibrationResults] = field(default=None)

    intermediate_results: IntermediateCalibrationResults = field(default_factory=IntermediateCalibrationResults)