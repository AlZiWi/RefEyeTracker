# VIVA Reference System

This project contains recording and calibration utilities and the CAD designs for a wearable stereo VOG eye-tracker.

Parts of this project are released as part of an ETRA paper: [Zimmer, Alexander & Abdrabou, Yasmeen & Kasneci, Enkelejda. (2026). An Affordable, Wearable Stereo-Eye-Tracking Platform.](https://arxiv.org/abs/2604.24331)

Note: The documentation is currently still being finalized.

## Hardware

Electronics are mostly covered by the [EyeTrackVR](https://www.eyetrackvr.dev/) project. We use four cameras to achieve a stereo setup.

3D-printable CAD files are provided in the "frames" subfolder.

# Utility

The utility can be run via python app.py. IP-addresses or serial ports need to be configured in the GUI.

For Recording and Calibration, choose the appropriate presets.

## Calibration Workflow:

1. Image acquisition (multiple images assume different poses of the mirror or calibration pattern)
    1. Camera pairs
        1. Capture >32 stereo images with small pattern for right camera pair
        2. Capture >32 stereo images with small pattern for left camera pair
    2. Capture >32 stereo images with large pattern for outer camera pair
    3. Capture >32 mono images with small pattern for scene camera
    4. Mirror calibration
        1. Capture single (multiple) stereo image with mirror for right camera pair, LEDs and scene camera visible
        2. Capture single (multiple) stereo image with mirror for left camera pair, LEDs (and scene camera visible)
    5. Capture single (multiple) mono image with mirror and mirror-mounted pattern for scene camera, right (and left) camera pair visible

2. Labelling
    1. LED labelling
        1. Label LED, camera and scene camera positions and scene camera position in 1.4.1
        2. Label LED, camera and scene camera positions (and scene camera position) in 1.4.2
    2. Label right (and left) camera positions in 1.5

3. Camera calibration (all extrinsics are transformed to a common world coordinate system defined by the right camera pair after each step)
    1. Camera pairs
        1. Calibrate intrinsic and extrinsic parameters of right camera pair using 1.1.1 [bundle adjustment]
        1. Calibrate intrinsic and extrinsic parameters of left camera pair using 1.1.2 [bundle adjustment]
    2. Calibrate extrinsic parameters of outer camera pair using 1.2 [bundle adjustment]
    3. Calibrate intrinsic parameters of scene camera using 1.3 [bundle adjustment]
    4. LED positions
        4. Calibrate LED positions and scene camera position using 2.1.1 and the calibration results from 3.1.1 [optional bundle adjustment]
        4. Calibrate LED positions (and scene camera position) using 2.1.2 and the calibration results from 3.1.2 [optional bundle adjustment]
    5. Calibrate scene camera extrinsic parameters using 2.1.1, (2.1.2), and 2.2 [optional bundle adjustment] [option to include 2.1.2 only if scene camera position is labeled in 2.1.2, otherwise only use 2.1.1]
    6. (Optional) Refine all calibration results using bundle adjustment with all images and all labeled points as input... if that works.