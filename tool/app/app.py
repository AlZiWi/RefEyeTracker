from os import path
from tkinter import *
from tkinter import ttk

import threading
import traceback

from camera import *

import multiprocessing

import cv2

from detect_pupil_ellseg import *
from stereo_calib_opencv import detect_corners, run_calibration
from calibration_utils import *

from datetime import datetime

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from mpl_toolkits.mplot3d import Axes3D

import asyncio
import websockets

import pickle

from eye_crop_and_glint import *



plt.rcParams["font.size"] = 5


class App:
    def __init__(self):
        
        # Defaults
        
        # Calibration Workflow:
        
        # 1) Image acquisition (multiple images assume different poses of the mirror or calibration pattern)
        # 1.1a) Capture >32 stereo images with small pattern for right camera pair
        # 1.1b) Capture >32 stereo images with small pattern for left camera pair
        # 1.2) Capture >32 stereo images with large pattern for outer camera pair
        # 1.3) Capture >32 mono images with small pattern for scene camera
        # 1.4a) Capture single (multiple) stereo image with mirror for right camera pair, LEDs and scene camera visible
        # 1.4b) Capture single (multiple) stereo image with mirror for left camera pair, LEDs (and scene camera visible)
        # 1.5) Capture single (multiple) mono image with mirror and mirror-mounted pattern for scene camera, right (and left) camera pair visible
        
        # 2) Labeling
        # 2.1a) Label LED, camera and scene camera positions and scene camera position in 1.4a
        # 2.1b) Label LED, camera and scene camera positions (and scene camera position) in 1.4b
        # 2.2) Label right (and left) camera positions in 1.5
        
        # 3) Camera calibration (all extrinsics are transformed to a common world coordinate system defined by the right camera pair after each step)
        # 3.1a) Calibrate intrinsic and extrinsic parameters of right camera pair using 1.1a [bundle adjustment]
        # 3.1b) Calibrate intrinsic and extrinsic parameters of left camera pair using 1.1b [bundle adjustment]
        # 3.2) Calibrate extrinsic parameters of outer camera pair using 1.2 [bundle adjustment]
        # 3.3) Calibrate intrinsic parameters of scene camera using 1.3 [bundle adjustment]
        # 3.4a) Calibrate LED positions and scene camera position using 2.1a and the calibration results from 3.1a [optional bundle adjustment]
        # 3.4b) Calibrate LED positions (and scene camera position) using 2.1b and the calibration results from 3.1b [optional bundle adjustment]
        # 3.5) Calibrate scene camera extrinsic parameters using 2.1a, (2.1b), and 2.2 [optional bundle adjustment] [option to include 2.1b only if scene camera position is labeled in 2.1b, otherwise only use 2.1a]
        # 3.6) (Optional) Refine all calibration results using bundle adjustment with all images and all labeled points as input... if that works.
        
        # This is efficient enough to run as one workflow and with caching even live when changing parameters.
        # UI concept:
        # General settings
        # - pattern sizes
        # - main calibration Folder
        # Tab 1 Presets for all image acquisition steps
        # Tab 2 Presets for all labeling steps (mostly implemented)
        # Tab 2b Live visualization of results (per step) -> Drop down with cycle buttons to switch between steps, visualize reprojection errors, etc.
        
        
        self.capture_calibration_presets = {
            "capture_preset_order": ["CAPTURE", "Stereo R-R", "Stereo L-L", "Stereo R-L", "Mono SC", "Mirror R", "Mirror L", "Mirror SC"],
            "calib_preset_order": ["Stereo R-R", "Stereo R-L", "Stereo L-L", "Mono SC", "Mirror R", "Mirror L", "Mirror SC", "Full Calibration"],
            "presets": {
                "CAPTURE": {
                    "description": "Default capture mode",
                    "camera_indices": ["ro", "ri", "lo", "li", "sc"],
                    "folder_path": "recording",
                    "use_pattern": None,
                    "capture_settings": {
                        "use_threshold": False,
                        "auto_accept": False,
                        "auto_contrast": False,
                    },
                    "calib_settings": None
                },
                "Stereo R-R": {
                    "description": "Capture stereo calibration images from the right camera pair.",
                    "camera_indices": ["ro", "ri"],
                    "folder_path": "calib/stereo_ro_ri",
                    "use_pattern": "small",
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": True,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": None,
                        "calib_dependencies": None
                    }
                },
                "Stereo L-L": {
                    "description": "Capture stereo calibration images from the left camera pair.",
                    "camera_indices": ["lo", "li"],
                    "folder_path": "calib/stereo_lo_li",
                    "use_pattern": "small",
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": True,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": None,
                        "calib_dependencies": None
                    }
                },
                "Stereo R-L": {
                    "description": "Capture stereo calibration images from the outer camera pair.",
                    "camera_indices": ["ro", "lo"],
                    "folder_path": "calib/stereo_ro_lo",
                    "use_pattern": "large",
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": True,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": None,
                        "calib_dependencies": None
                    }
                },
                "Mono SC": {
                    "name": "Mono SC",
                    "description": "Capture calibration images from the scene camera.",
                    "camera_indices": ["sc"],
                    "folder_path": "calib/mono_sc",
                    "use_pattern": "large",
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": True,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": None,
                        "calib_dependencies": None
                    }
                },
                "Mirror R": {
                    "description": "Capture mirror calibration images with the right camera pair, LEDs and scene camera visible.",
                    "camera_indices": ["ro", "ri"],
                    "folder_path": "calib/mirror_r",
                    "use_pattern": None,
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": False,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": "MIRROR_STEREO",
                        "calib_dependencies": ["Stereo R-R"]
                    }
                },
                "Mirror L": {
                    "description": "Capture mirror calibration images with the left camera pair, LEDs and scene camera visible.",
                    "camera_indices": ["lo", "li"],
                    "folder_path": "calib/mirror_l",
                    "use_pattern": None,
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": False,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": "MIRROR_STEREO",
                        "calib_dependencies": ["Stereo L-L"]
                    }
                },
                "Mirror SC": {
                    "description": "Capture mirror calibration images with the scene camera, LEDs and both camera pairs visible.",
                    "camera_indices": ["sc"],
                    "folder_path": "calib/mirror_sc",
                    "use_pattern": "small",
                    "capture_settings": {
                        "use_threshold": True,
                        "auto_accept": False,
                        "auto_contrast": True
                    },
                    "calib_settings": {
                        "calib_labels": "MIRROR_SC",
                        "calib_dependencies": ["Stereo R-R", "Stereo L-L", "Mono SC"]
                    }
                },
                "Full Calibration": {
                    "description": "Save all calibration results to a single file for later use.",
                    "camera_indices": None,
                    "folder_path": "calib/full",
                    "use_pattern": None,
                    "capture_settings": None,
                    "calib_settings": {
                        "calib_labels": None,
                        "calib_dependencies": ["Stereo R-R", "Stereo L-L", "Stereo R-L", "Mono SC", "Mirror R", "Mirror L", "Mirror SC"]
                    }
                }
            }
        }
        
        url_cams_ROUTER = ['http://192.168.178.20', 'http://192.168.178.21', 'http://192.168.178.22', 'http://192.168.178.23', 0]  # IP camera URLs and USB camera index (Find camera index via ffmpeg -f avfoundation -list_devices true -i "")
        url_cams_MAC = ['http://192.168.2.8', 'http://192.168.2.9', 'http://192.168.2.6', 'http://192.168.2.7', 0]
        
        adapter_ip_ROUTER = "192.168.178.57"  # Ethernet adapter IP address
        adapter_ip_SILVER = "192.168.178.28"
        adapter_ip_MAC = "10.162.127.98"

        default_adapter_ip = adapter_ip_ROUTER
        default_url_cams = url_cams_ROUTER
        
        available_types_cams = ['IP', 'USB']
        
        self.camera_settings = {
            "ro": {
                "url": default_url_cams[0],
                "type": 'IP',
                "grid_placement": [0,0],  # col, row positions of visualization frames for each camera
                "eye_cam": True
            },
            "ri": {
                "url": default_url_cams[1],
                "type": 'IP',
                "grid_placement": [1,0],
                "eye_cam": True
            },
            "lo": {
                "url": default_url_cams[3],
                "type": 'IP',
                "grid_placement": [0,1],
                "eye_cam": True
            },
            "li": {
                "url": default_url_cams[2],
                "type": 'IP',
                "grid_placement": [1,1],
                "eye_cam": True
            },
            "sc": {
                "url": default_url_cams[4],
                "type": 'USB',
                "grid_placement": [0,2],
                "eye_cam": False
            }
        }
        
        default_auto_accept_img_diff_threshold = 4.0
        
        self.calibration_settings = {
            "patterns": {
                "small": {
                    "corners": (10,7),  # inner corners per chessboard row and column
                    "square_size_mm": 3.0,  # in mm
                },
                "large": {
                    "corners": (10,7),
                    "square_size_mm": 6.0,
                }
            },
            "pixel_pitch_um": (2700/240.0, 2700/240.0),
            "pixel_pitch_sc_um": (3, 3),  # OV9281
            
            "calib_output_directory": "out",
            "calib_output_pkl_filename": "calib_results",  # without extension, .pkl or .json will be added automatically
            "calib_labels_pkl_suffix": "_labels.pkl"
        }

        self.label_types_configs = {
            "MIRROR_STEREO": {
                "available_labels": ["0", "1", "2", "3", "CAM_0", "CAM_1", "CAM_SC"],
                "source_camera_labels": ["CAM_0", "CAM_1"]  # Cameras used to reconstruct the mirror position
            },
            "MIRROR_SC": {
                "available_labels": ["CAM_SC", "CAM_RO", "CAM_RI", "CAM_LO", "CAM_LI"],
            }
        }

        default_calib_experiment_name = "data/calibration_2026_05"
        default_gaze_experiment_name = "data/a_test/saccade33_1m__2026_03_12_17_15_35"
        self.gaze_view_modes = ["Input", "Segmentation", "Gaze Estimation"]
        self.gaze_view_modes_path_suffixes = {
            "Input": "",
            "Segmentation": "_output/segmentation",
            "Gaze Estimation": "_output/gaze_estimation"
        }
        self.gaze_estimation_output_filename = "gaze_estimation_results.pkl"

        ## App Status
        
        # Capture
        self.curr_capture_folder_path = ""
        self.curr_capture_cam_labels = []
        self.curr_capture_cam_urls = []
        self.camera_handler = None
        
        # TODO enums
        self.capture_state = "inactive"  # "stream, capture"
        self.capture_snapshot_state = "idle"  # "idle", "triggered_snapshot", "awaiting_accept"
        self.capture_sync_mode = False
        
        self.cooldown_auto_accept = 1
        self.cooldown_auto_accept_rendering = 0.5  # seconds to wait after rendering accepted snapshot to allow user to see the pattern in the stream
        self.last_timestamp_auto_accept = -1
        
        # Calibration
        self.selected_label_type = None
        self.calibration_status = "idle"

        self.app_running = True

        # Layout
        
        self.root = Tk()
        self.root.title("VIVA Reference System Calibration Utility")
        
        self.tab_control = ttk.Notebook(self.root)

        self.tab_capture = ttk.Frame(self.tab_control)
        self.tab_calibration = ttk.Frame(self.tab_control)
        self.tab_gaze_estimation = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_capture, text ='Capture')
        self.tab_control.add(self.tab_calibration, text ='Calibration')
        self.tab_control.add(self.tab_gaze_estimation, text ='Gaze Estimation')
        self.tab_control.pack(expand = 1, fill ="both")
        
        # - TAB Capture
        
        # -- Cameras

        self.cam_mainframe = ttk.LabelFrame(self.tab_capture, text="Cameras", padding=(5,5))
        self.cam_mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        self.cam_frames = {}

        for label, cam_settings in self.camera_settings.items():
            self.cam_frames[label] = {}
            self.cam_frames[label]["img_store"] = None  # Store PhotoImage to avoid garbage collection
            self.cam_frames[label]["img_store_raw"] = None  # Store image to avoid garbage collection
            self.cam_frames[label]["metadata_store"] = None  # Store PhotoImage to avoid garbage collection

            self.cam_frames[label]["frame"] = ttk.LabelFrame(self.cam_mainframe, text=f"Cam [{label}]", padding=(5,5))
            self.cam_frames[label]["frame"].grid(column=cam_settings['grid_placement'][0], row=cam_settings['grid_placement'][1]+1, sticky=(N, W))

            # https://stackoverflow.com/questions/4310489/how-do-i-remove-the-light-grey-border-around-my-canvas-widget
            self.cam_frames[label]["w_canvas"] = Canvas(self.cam_frames[label]["frame"], width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
            self.cam_frames[label]["w_canvas"].grid(column=0, row=1, sticky=(N, W))
            self.cam_frames[label]["w_canvas"].xview_moveto(5)  # ref highlightthickness
            self.cam_frames[label]["w_canvas"].yview_moveto(5)  # ref highlightthickness

            self.cam_frames[label]["frame_info"] = ttk.Frame(self.cam_frames[label]["frame"])
            self.cam_frames[label]["frame_info"].grid(column=1, row=1, sticky=(N, W))

            ttk.Label(self.cam_frames[label]["frame_info"], text=f"Cam IP").grid(column=1, row=2, sticky=(N,W))
            self.cam_frames[label]["sv_ip"] = StringVar()
            self.cam_frames[label]["sv_ip"].set(cam_settings['url'])
            ttk.Entry(self.cam_frames[label]["frame_info"], textvariable=self.cam_frames[label]["sv_ip"], width=15).grid(column=2, row=2, sticky=(N,W))

            self.cam_frames[label]["sv_active"] = StringVar()
            self.cam_frames[label]["sv_active"].set('active')
            ttk.Checkbutton(self.cam_frames[label]["frame_info"], text="Active", variable=self.cam_frames[label]["sv_active"], onvalue='active', offvalue='inactive').grid(column=1, row=3, sticky=W)

            # Dropdown for camera type (IP/USB)
            self.cam_frames[label]["sv_type"] = StringVar()
            ttk.OptionMenu(self.cam_frames[label]["frame_info"], self.cam_frames[label]["sv_type"], cam_settings['type'], *available_types_cams).grid(column=1, row=4, sticky=W)

            ttk.Label(self.cam_frames[label]["frame_info"], text=f"Cam Info:").grid(column=1, row=5, sticky=(N,W))
            self.cam_frames[label]["sv_info"] = StringVar()
            ttk.Label(self.cam_frames[label]["frame_info"], textvariable=self.cam_frames[label]["sv_info"], font=("Courier", 10)).grid(column=2, row=5, sticky=(N,W))

        # -- Controls

        self.frame_controls = ttk.LabelFrame(self.tab_capture, text="Controls", padding=(5,5))
        self.frame_controls.grid(column=1, row=0, sticky=(N, W, E, S))
        
        # --- Preset

        self.capture_preset_frame = ttk.LabelFrame(self.frame_controls, text="Capture Preset", padding=(5,5))
        self.capture_preset_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.capture_preset_frame, text="Capture Preset:").grid(column=0, row=0, sticky=(N,W))
        self.capture_preset = StringVar()
        self.capture_preset_optionmenu = ttk.OptionMenu(self.capture_preset_frame, self.capture_preset, self.capture_calibration_presets["capture_preset_order"][0], *self.capture_calibration_presets["capture_preset_order"])
        self.capture_preset_optionmenu.grid(column=1, row=0, sticky=W)
        self.capture_preset.trace_add("write", self.load_capture_preset)
        
        # --- General Settings

        self.frame_controls_general = ttk.LabelFrame(self.frame_controls, text="General Settings", padding=(5,5))
        self.frame_controls_general.grid(column=0, row=1, sticky=(N, W, E, S))

        ttk.Label(self.frame_controls_general, text="Adapter IP").grid(column=0, row=0, sticky=(N,W))
        self.sv_adapter_ip = StringVar()
        self.sv_adapter_ip.set(default_adapter_ip)
        ttk.Entry(self.frame_controls_general, textvariable=self.sv_adapter_ip).grid(column=1, row=0, sticky=(N,W))
        
        
        ttk.Label(self.frame_controls_general, text="Gaze Socket Endpoint").grid(column=0, row=1, sticky=(N,W))
        self.sv_gaze_socket_url = StringVar()
        self.sv_gaze_socket_url.set("")
        ttk.Entry(self.frame_controls_general, textvariable=self.sv_gaze_socket_url).grid(column=1, row=1, sticky=(N,W))

        self.sv_sync_recording_with_stimulus = StringVar()
        self.sv_sync_recording_with_stimulus.set('inactive')
        ttk.Checkbutton(self.frame_controls_general, text="Sync with Stimulus", variable=self.sv_sync_recording_with_stimulus, onvalue='active', offvalue='inactive').grid(column=0, row=2, sticky=W)

        self.sv_use_external_trigger = StringVar()
        self.sv_use_external_trigger.set('inactive')
        ttk.Checkbutton(self.frame_controls_general, text="Use External Trigger", variable=self.sv_use_external_trigger, onvalue='active', offvalue='inactive').grid(column=0, row=3, sticky=W)

        self.sv_adjust_contrast = StringVar()
        self.sv_adjust_contrast.set('inactive')
        ttk.Checkbutton(self.frame_controls_general, text="Auto Contrast", variable=self.sv_adjust_contrast, onvalue='active', offvalue='inactive').grid(column=0, row=4, sticky=W)

        # --- Stream and Visualization

        self.frame_controls_stream = ttk.LabelFrame(self.frame_controls, text="Stream and Visualization", padding=(5,5))
        self.frame_controls_stream.grid(column=0, row=2, sticky=(N, W, E, S))
        
        self.sv_btn_start_stream = StringVar()
        self.sv_btn_start_stream.set("Start Stream")
        ttk.Button(self.frame_controls_stream, textvariable=self.sv_btn_start_stream, command=lambda: self.toggle_capture("stream")).grid(column=0, row=0, sticky=W)

        self.sv_vis_calibration_pattern = StringVar()
        self.sv_vis_calibration_pattern.set('inactive')
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Calibration Pattern", variable=self.sv_vis_calibration_pattern, onvalue='active', offvalue='inactive').grid(column=0, row=1, sticky=W)
        
        self.sv_vis_img_diff_thr = StringVar()
        self.sv_vis_img_diff_thr.set('active')
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Img Diff Threshold", variable=self.sv_vis_img_diff_thr, onvalue='active', offvalue='inactive').grid(column=0, row=2, sticky=W)
        
        self.sv_auto_accept_img_diff_threshold = StringVar()
        self.sv_auto_accept_img_diff_threshold.set(default_auto_accept_img_diff_threshold)
        ttk.Entry(self.frame_controls_stream, textvariable=self.sv_auto_accept_img_diff_threshold, width=10).grid(column=1, row=2, sticky=(N,W))
        
        self.sv_vis_pupil_detection = StringVar()
        self.sv_vis_pupil_detection.set('inactive')
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Pupil Detection", variable=self.sv_vis_pupil_detection, onvalue='active', offvalue='inactive').grid(column=0, row=3, sticky=W)

        # --- Capture

        self.frame_controls_capture = ttk.LabelFrame(self.frame_controls, text="Capture", padding=(5,5))
        self.frame_controls_capture.grid(column=0, row=3, sticky=(N, W, E, S))
        
        # ---- Export Settings

        self.frame_controls_capture_export = ttk.LabelFrame(self.frame_controls_capture, text="Export Settings", padding=(5,5))
        self.frame_controls_capture_export.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.frame_controls_capture_export, text="Path").grid(column=0, row=0, sticky=(N,W))
        self.sv_experiment_name = StringVar()
        self.sv_experiment_name.set(default_calib_experiment_name)
        ttk.Entry(self.frame_controls_capture_export, textvariable=self.sv_experiment_name, width=50).grid(column=1, row=0, sticky=(N,W))

        # ---- Synchronized Capture

        self.frame_controls_capture_record = ttk.LabelFrame(self.frame_controls_capture, text="Synchronized Capture", padding=(5,5))
        self.frame_controls_capture_record.grid(column=0, row=1, sticky=(N, W, E, S))
        
        self.sv_recording = StringVar()
        self.sv_recording.set("Start Capture")
        ttk.Button(self.frame_controls_capture_record, textvariable=self.sv_recording, command=lambda: self.toggle_capture("capture")).grid(column=0, row=0, sticky=W)

        self.sv_capture_with_stream = StringVar()
        self.sv_capture_with_stream.set('inactive')
        ttk.Checkbutton(self.frame_controls_capture_record, text="Show Stream", variable=self.sv_capture_with_stream, onvalue='active', offvalue='inactive').grid(column=0, row=1, sticky=W)

        # ---- Pattern-Based Capture
        
        self.frame_controls_capture_pattern = ttk.LabelFrame(self.frame_controls_capture, text="Pattern-Based Capture", padding=(5,5))
        self.frame_controls_capture_pattern.grid(column=0, row=2, sticky=(N, W, E, S))

        ttk.Button(self.frame_controls_capture_pattern, text="Triggered Snapshot", command=self.onclick_triggered_snapshot).grid(column=0, row=0, sticky=W)
        ttk.Button(self.frame_controls_capture_pattern, text="Accept", command=self.accept_triggered_snapshot).grid(column=0, row=1, sticky=W)
        ttk.Button(self.frame_controls_capture_pattern, text="Stop", command=self.stop_triggered_snapshot).grid(column=0, row=2, sticky=W)

        self.sv_auto_accept = StringVar()
        self.sv_auto_accept.set('inactive')
        ttk.Checkbutton(self.frame_controls_capture_pattern, text="Auto-Accept", variable=self.sv_auto_accept, onvalue='active', offvalue='inactive').grid(column=0, row=3, sticky=W)

        self.sv_capture_status = StringVar()
        self.sv_capture_status.set("Idle")
        ttk.Label(self.frame_controls_capture_pattern, textvariable=self.sv_capture_status).grid(column=0, row=5, sticky=W)
        
        # - TAB Calibration

        # -- Calibration Workflow

        self.calibration_mainframe = ttk.LabelFrame(self.tab_calibration, text="Calibration Workflow", padding=(5,5))
        self.calibration_mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        # --- Calibration Configuration

        self.calib_config_frame = ttk.LabelFrame(self.calibration_mainframe, text="Calibration Configuration", padding=(5,5))
        self.calib_config_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.calib_config_frame, text="Calibration Preset:").grid(column=0, row=0, sticky=(N, W, E, S))
        self.calib_preset = StringVar()
        self.calib_preset_optionmenu = ttk.OptionMenu(self.calib_config_frame, self.calib_preset, self.capture_calibration_presets["calib_preset_order"][0], *self.capture_calibration_presets["calib_preset_order"])
        self.calib_preset_optionmenu.grid(column=1, row=0, sticky=W)
        self.calib_preset.trace_add("write", self.load_calib_preset)
        
        ttk.Button(self.calib_config_frame, text="Run Calibration Step", command=self.run_calibration_step).grid(column=0, row=1, sticky=W)
        ttk.Button(self.calib_config_frame, text="Run All Until Step", command=self.run_calibration_steps_until).grid(column=0, row=2, sticky=W)
        ttk.Button(self.calib_config_frame, text="Visualize Calibration Results", command=self.visualize_calibration_results).grid(column=0, row=3, sticky=W)

        # --- Labeling Frames

        self.label_mainframe = ttk.LabelFrame(self.calibration_mainframe, text="Calibration Labeling", padding=(5,5))
        self.label_mainframe.grid(column=0, row=1, sticky=(N, W, E, S))
        
        self.label_file_frame = {}
        
        self.label_file_frame["frame"] = ttk.LabelFrame(self.label_mainframe, text=f"Image and Label selection", padding=(5,5))
        self.label_file_frame["frame"].grid(column=0, row=0, sticky=(N, W))
        
        ## Images

        self.label_file_frame["available_label_imgs_store"] = []  # Store list of available label images
        self.label_file_frame["selected_label_img_id"] = -1

        self.label_file_frame["sv_available_label_imgs"] = StringVar(value=[])
        self.label_file_frame["listbox_available_label_imgs"] = Listbox(self.label_file_frame["frame"], listvariable=self.label_file_frame["sv_available_label_imgs"], height=7, width=28, selectmode=SINGLE)
        self.label_file_frame["listbox_available_label_imgs"].grid(column=0, row=0, sticky=W)
        self.label_file_frame["listbox_available_label_imgs"].bind("<<ListboxSelect>>", self.onclick_listbox_available_label_imgs)

        ttk.Button(self.label_file_frame["frame"], text="Save Labels", command=self.save_label_coordinates).grid(column=0, row=1, sticky=W)
        
        ## Labels
        
        ttk.Separator(self.label_file_frame["frame"], orient=HORIZONTAL).grid(column=0, row=2, sticky=(E, W), pady=5)
        
        self.label_file_frame["sv_available_label_names"] = StringVar(value=[])
        self.label_file_frame["listbox_available_label_names"] = Listbox(self.label_file_frame["frame"], listvariable=self.label_file_frame["sv_available_label_names"], height=7, width=16, selectmode=SINGLE)
        self.label_file_frame["listbox_available_label_names"].grid(column=0, row=3, sticky=W)

        ttk.Label(self.label_file_frame["frame"], text=f"Click image to label").grid(column=0, row=4, sticky=(N,W))

        ## Label Frames

        self.label_frames = []
        
        for i in range(2):        
            self.label_frames.append({})
            
            # Variables

            self.label_frames[i]["img_store"] = None  # Store PhotoImage to avoid garbage collection
            self.label_frames[i]["img_store_raw"] = None  # Store imported raw image
            self.label_frames[i]["img_folder_path_store"] = None  # Store folder path of imported images
            self.label_frames[i]["img_path_full_store"] = None  # Store full path of imported images
            self.label_frames[i]["label_coordinates_store"] = {}
            self.label_frames[i]["pattern_corners_store"] = None  # Store detected pattern corners for visualization

            # Frame
            
            self.label_frames[i]["frame"] = ttk.LabelFrame(self.label_mainframe, text=f"CAM [{i+1}] Labels", padding=(5,5))
            self.label_frames[i]["frame"].grid(column=i+1, row=0, sticky=(N, W))

            self.label_frames[i]["w_canvas"] = Canvas(self.label_frames[i]["frame"], width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
            self.label_frames[i]["w_canvas"].grid(column=0, row=0, sticky=(N, W), padx=0, pady=0)
            self.label_frames[i]["w_canvas"].bind("<Button-1>", lambda event, idx=i: self.onclick_set_label_coordinates(event, idx))
            self.label_frames[i]["w_canvas"].xview_moveto(5)  # ref highlightthickness
            self.label_frames[i]["w_canvas"].yview_moveto(5)  # ref highlightthickness
            
            ttk.Button(self.label_frames[i]["frame"], text="Reset Label", command=lambda idx=i: self.reset_label_coordinates(idx)).grid(column=0, row=1, sticky=W)
            ttk.Button(self.label_frames[i]["frame"], text="Reset All Labels", command=lambda idx=i: self.reset_all_label_coordinates(idx)).grid(column=0, row=2, sticky=W)
            
            self.label_frames[i]["sv_label_coordinates"] = StringVar()
            ttk.Label(self.label_frames[i]["frame"], textvariable=self.label_frames[i]["sv_label_coordinates"], font=("Courier", 12)).grid(column=0, row=3, sticky=(N,W))


        # -- Calibration Configuration

        self.frame_calib_conf = ttk.LabelFrame(self.tab_calibration, text="Output", padding=(5,5))
        self.frame_calib_conf.grid(column=1, row=0, sticky=(N, W, E, S))

        # --- Visualization

        self.frame_calib_conf_vis = ttk.LabelFrame(self.frame_calib_conf, text="Visualization", padding=(5,5))
        self.frame_calib_conf_vis.grid(column=0, row=10, sticky=(N, W, E, S))

        self.canvas_calib_1 = FigureCanvasTkAgg(Figure(figsize=(6, 4)), master=self.frame_calib_conf_vis)
        self.canvas_calib_1.get_tk_widget().grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Separator(self.frame_calib_conf_vis, orient=HORIZONTAL).grid(column=0, row=1, sticky=(E, W), pady=5)

        self.canvas_calib_2 = FigureCanvasTkAgg(Figure(figsize=(6, 3)), master=self.frame_calib_conf_vis)
        self.canvas_calib_2.get_tk_widget().grid(column=0, row=2, sticky=(N, W, E, S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # - TAB Gaze Estimation

        self.frame_gaze_estimation = ttk.LabelFrame(self.tab_gaze_estimation, text="Gaze Estimation", padding=(5,5))
        self.frame_gaze_estimation.grid(column=0, row=0, sticky=(N, W, E, S))

        # --- Gaze Estimation Visualization Layout
        self.gaze_vis_frame = ttk.LabelFrame(self.frame_gaze_estimation, text="Visualization", padding=(5,5))
        self.gaze_vis_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        # Small camera canvases (2x2)
        self.gaze_cam_frames = {}
        cam_order = ['ro', 'ri', 'lo', 'li']
        for i, cam_label in enumerate(cam_order):
            frm = ttk.Frame(self.gaze_vis_frame)
            frm.grid(column=i % 2, row=(i // 2) * 2, padx=5, pady=5, sticky=(N, W))
            
            ttk.Label(frm, text=f"Cam [{cam_label}]").grid(column=0, row=0, sticky=(N, W))
            
            subframe = ttk.Frame(frm)
            subframe.grid(column=0, row=1, sticky=(N, W))
            
            c = Canvas(subframe, width=120, height=120, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
            c.grid(column=0, row=0, sticky=(N, W))
            c.xview_moveto(5)  # ref highlightthickness
            c.yview_moveto(5)  # ref highlightthickness
            
            c2 = Canvas(subframe, width=120, height=120, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
            c2.grid(column=1, row=0, sticky=(N, W))
            c2.xview_moveto(5)  # ref highlightthickness
            c2.yview_moveto(5)  # ref highlightthickness
            
            info = ttk.Label(frm, text="--", font=("Courier", 9))
            info.grid(column=0, row=2, sticky=(N, W))

            self.gaze_cam_frames[cam_label] = {"frame": frm, "subframe": subframe, "canvas": c, "canvas2": c2, "info": info}

        # Scene camera and gaze-result visualization (side by side)
        self.gaze_scene_wrap = ttk.Frame(self.gaze_vis_frame)
        self.gaze_scene_wrap.grid(column=0, row=4, columnspan=2, sticky=(N, W), pady=(8,0))

        # Scene camera
        sc_frame = ttk.Frame(self.gaze_scene_wrap)
        sc_frame.grid(column=0, row=0, padx=5, sticky=(N, W))
        
        ttk.Label(sc_frame, text="Scene Camera").grid(column=0, row=0, sticky=(N, W))
        
        canvas_scene = Canvas(sc_frame, width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
        canvas_scene.grid(column=0, row=1, sticky=(N, W))
        canvas_scene.xview_moveto(5)  # ref highlightthickness
        canvas_scene.yview_moveto(5)  # ref highlightthickness

        scene_info = ttk.Label(sc_frame, text="--", font=("Courier", 10))
        scene_info.grid(column=0, row=2, sticky=(N, W))

        self.gaze_cam_frames['sc'] = {"frame": sc_frame, "canvas": canvas_scene, "info": scene_info}

        # Gaze results
        gr_frame = ttk.Frame(self.gaze_scene_wrap)
        gr_frame.grid(column=1, row=0, padx=5, sticky=(N, W))
        
        ttk.Label(gr_frame, text="Gaze Results").grid(column=0, row=0, sticky=(N, W))
        
        self.canvas_gaze_result = FigureCanvasTkAgg(Figure(figsize=(3, 3)), master=gr_frame)
        self.canvas_gaze_result.get_tk_widget().grid(column=0, row=1, sticky=(N, W))

        self.gaze_info = ttk.Label(gr_frame, text="--", font=("Courier", 10))
        self.gaze_info.grid(column=0, row=2, sticky=(N, W))

        # --- Gaze Estimation Controls
        self.gaze_ctrl_frame = ttk.LabelFrame(self.frame_gaze_estimation, text="Controls", padding=(5,5))
        self.gaze_ctrl_frame.grid(column=1, row=0, sticky=(N, W, E, S))

        ttk.Label(self.gaze_ctrl_frame, text="Experiment Path:").grid(column=0, row=0, sticky=(N, W))
        self.sv_gaze_experiment_path = StringVar()
        self.sv_gaze_experiment_path.set(default_gaze_experiment_name)
        ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_experiment_path, width=40).grid(column=1, row=0, sticky=(N, W))

        ttk.Label(self.gaze_ctrl_frame, text="Calibration Path:").grid(column=0, row=1, sticky=(N, W))
        self.sv_gaze_calib_path = StringVar()
        self.sv_gaze_calib_path.set(default_calib_experiment_name)
        ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_calib_path, width=40).grid(column=1, row=1, sticky=(N, W))
        
        ttk.Label(self.gaze_ctrl_frame, text="View:").grid(column=0, row=2, sticky=(N, W, E, S))
        self.sv_gaze_view_mode = StringVar()
        self.gaze_view_mode_optionmenu = ttk.OptionMenu(self.gaze_ctrl_frame, self.sv_gaze_view_mode, self.gaze_view_modes[0], *self.gaze_view_modes)
        self.gaze_view_mode_optionmenu.grid(column=1, row=2, sticky=W)
        self.sv_gaze_view_mode.trace_add("write", self.display_current_gaze_frame)

        ttk.Button(self.gaze_ctrl_frame, text="Load", command=self.load_gaze_paths).grid(column=0, row=3, sticky=W)
        ttk.Button(self.gaze_ctrl_frame, text="Run Segmentation", command=self.start_segmentation).grid(column=0, row=4, sticky=W)
        ttk.Button(self.gaze_ctrl_frame, text="Run Gaze Estimation", command=self.start_gaze_estimation).grid(column=0, row=5, sticky=W)
        
        ttk.Label(self.gaze_ctrl_frame, text="Until Frame").grid(column=0, row=6, sticky=(N, W))
        self.sv_gaze_until_frame_id = StringVar()
        self.sv_gaze_until_frame_id.set("-1")
        ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_until_frame_id, width=8).grid(column=1, row=6, sticky=(N, W))
        
        ttk.Label(self.gaze_ctrl_frame, text="alpha").grid(column=0, row=7, sticky=(N, W))
        self.sv_gaze_alpha = StringVar()
        self.sv_gaze_alpha.set("1.0")
        ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_alpha, width=8).grid(column=1, row=7, sticky=(N, W))

        ttk.Label(self.gaze_ctrl_frame, text="beta").grid(column=0, row=8, sticky=(N, W))
        self.sv_gaze_beta = StringVar()
        self.sv_gaze_beta.set("0.0")
        ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_beta, width=8).grid(column=1, row=8, sticky=(N, W))

        self.label_gaze_status = ttk.Label(self.gaze_ctrl_frame, text="Status: Idle", font=("Courier", 10))
        self.label_gaze_status.grid(column=0, row=10, columnspan=2, sticky=(N, W), pady=(8,0))

        # Frame navigation and animation
        nav_wrap = ttk.Frame(self.gaze_ctrl_frame)
        nav_wrap.grid(column=0, row=9, columnspan=2, pady=(8,0), sticky=(W))

        ttk.Button(nav_wrap, text="<", width=3, command=self.goto_frame_prev).grid(column=0, row=0, padx=(0,4))
        self.sv_gaze_frame_id = StringVar()
        self.sv_gaze_frame_id.set("0")
        ttk.Entry(nav_wrap, textvariable=self.sv_gaze_frame_id, width=8).grid(column=1, row=0)
        self.sv_gaze_frame_id.trace_add("write", self.onchange_gaze_frame_id)
        ttk.Button(nav_wrap, text=">", width=3, command=self.goto_frame_next).grid(column=2, row=0, padx=(4,6))

        self.btn_animate_gaze = ttk.Button(nav_wrap, text="Animate", command=self.toggle_animate_gaze)
        self.btn_animate_gaze.grid(column=3, row=0)

        # Internal gaze state
        self.gaze_imgs = {}  # list of available frame filenames (filled by load_gaze_paths)
        self.gaze_current_frame_idx = 0
        self.gaze_max_frame_idxs = {cam_label: 0 for cam_label in self.camera_settings.keys()}
        self.gaze_animating = False
        self.gaze_anim_after_id = None
        
        self.gaze_thread_handler = {
            "status_q": multiprocessing.Queue(),
            "thread_status": "idle",
            "thread": None
        }

        ##### RUN

        self.init_logic()


    def init_logic(self):
        self.load_capture_preset()
        self.load_calib_preset()


    # -------------------- Gaze estimation helpers (UI handlers) --------------------
    
    def load_gaze_paths(self):
        """Load gaze experiment frames from the experiment path entry."""
        
        exp_path = Path(self.sv_gaze_experiment_path.get())
        cam_subfolders = self.camera_settings.keys()  # Assuming camera labels correspond to subfolder names in the experiment path, e.g. "ro", "ri", "lo", "li", "sc".
        self.gaze_imgs = {cam_label: {} for cam_label in cam_subfolders}
        self.gaze_max_frame_idxs = {cam_label: 0 for cam_label in cam_subfolders}
        
        gaze_view_mode = self.sv_gaze_view_mode.get()
        
        for cam_label in cam_subfolders:
            cam_path = exp_path / f"{cam_label}{self.gaze_view_modes_path_suffixes['Input']}"
            if not (cam_path.exists() and cam_path.is_dir()):
                print(f"[Gaze] Input camera path not found for {cam_label}: {cam_path}.")
                continue

            img_names = [p.name for p in cam_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            if not img_names:
                print(f"[Gaze] No image files found in {cam_path} for camera {cam_label}.")
                continue
            
            def get_frame_metadata_from_filename(filename):
                match = re.search(r'frame_(\d+)_timestamp_[\d\.]+', filename)
                if match:
                    frame_idx = int(match.group(1))
                    frame_timestamp = float(match.group(0).split('_timestamp_')[1].split('.png')[0][:-1])
                    return frame_idx, frame_timestamp
                else:
                    return float('inf'), float('inf')  # If no match, place at the end

            # Sort img_names by frame index extracted from filename
            img_names.sort(key=lambda x: get_frame_metadata_from_filename(x)[0])

            img_names_aligned_by_frame_idx = {}
            img_names_aligned_by_timestamp = {}
            # Parse frame index using regex. Pattern: frame_0_timestamp_0.000.png
            for img_name in img_names:
                frame_idx, frame_timestamp = get_frame_metadata_from_filename(img_name)
                if frame_idx != float('inf'):
                    img_names_aligned_by_frame_idx[frame_idx] = {"filename": img_name, "timestamp": frame_timestamp}
                    # Create bins matching framerate of 44.5fps (~22.47ms per frame)
                    base_framerate = 44.5
                    bin_idx = int((frame_timestamp + .5*(1/base_framerate)) // (1/base_framerate))
                    if bin_idx not in img_names_aligned_by_timestamp.keys():
                        img_names_aligned_by_timestamp[bin_idx] = {"filename": img_name, "frame_idx": frame_idx}  # IF support for multiple images per bin is needed, change to list
                    else:
                        pass
                        #print(f"[Gaze] Warning: Multiple images found for timestamp bin {bin_idx} in camera {cam_label}. Using the first one.")

            curr_highest_idx = max(img_names_aligned_by_timestamp.keys())
            self.gaze_max_frame_idxs[cam_label] = curr_highest_idx

            self.gaze_imgs[cam_label] = {
                "cam_path": cam_path,
                "segmentation_path": exp_path / f"{cam_label}{self.gaze_view_modes_path_suffixes['Segmentation']}",
                "gaze_estimation_path": exp_path / f"{cam_label}{self.gaze_view_modes_path_suffixes['Gaze Estimation']}",
                "img_names": img_names,
                "img_names_aligned_by_frame_idx": img_names_aligned_by_frame_idx,
                "img_names_aligned_by_timestamp": img_names_aligned_by_timestamp
            }
            print(f"[Gaze] Loaded {len(img_names)} frames for camera {cam_label} from {cam_path}")
            
        # fill all missing frame indices with None to ensure continuous indexing
        for cam_label in self.gaze_imgs.keys():
            img_names_aligned_by_frame_idx = self.gaze_imgs[cam_label]["img_names_aligned_by_frame_idx"]
            if img_names_aligned_by_frame_idx:
                min_frame_idx = min(img_names_aligned_by_frame_idx.keys())
                max_frame_idx_cam = max(img_names_aligned_by_frame_idx.keys())
                for idx in range(min_frame_idx, max_frame_idx_cam + 1):
                    if idx not in img_names_aligned_by_frame_idx:
                        img_names_aligned_by_frame_idx[idx] = None
        
        self.gaze_current_frame_idx = 0
        self.sv_gaze_frame_id.set('0')  # Side effect: triggers onchange event to display first frame

    
    def start_segmentation(self):
        self.label_gaze_status.config(text="Status: Starting Segmentation...")
        
        until_frame_id = int(self.sv_gaze_until_frame_id.get())
        experiment_path = Path(self.sv_gaze_experiment_path.get())
        cam_keys = [key for key, value in self.camera_settings.items() if value["eye_cam"]]
        output_path_suffix = self.gaze_view_modes_path_suffixes['Segmentation']
        
        calib_results = self.load_calibration_results(Path(self.sv_gaze_calib_path.get()))
        if calib_results is None:
            camera_tilt_angles_deg = {cam_key: 0 for cam_key in cam_keys}
            camera_matrices = {cam_key: None for cam_key in cam_keys}
            distortion_coeffs = {cam_key: None for cam_key in cam_keys}
        else:
            camera_tilt_angles_deg = {cam_key: self.estimate_camera_tilt_angle(calib_results, cam_key) for cam_key in cam_keys}
            camera_matrices = {cam_key: np.array(calib_results["intrinsics"][cam_key]["K"]) for cam_key in cam_keys}
            distortion_coeffs = {cam_key: np.array(calib_results["intrinsics"][cam_key]["dist"]) for cam_key in cam_keys}
        
        params = {
            "alpha": float(self.sv_gaze_alpha.get()),
            "beta": float(self.sv_gaze_beta.get()),
            "crop_ratio": 0.5,
            "camera_tilt_angles_deg": camera_tilt_angles_deg,
            "camera_matrices": camera_matrices,
            "distortion_coeffs": distortion_coeffs
        }

        self.gaze_thread_handler["thread_status"] = "starting"
        self.gaze_thread_handler["status_q"] = multiprocessing.Queue()  # Reset the queue for new segmentation run
        self.gaze_thread_handler["thread"] = threading.Thread(target=self.run_segmentation, args=(self.gaze_thread_handler, self.gaze_imgs.copy(), until_frame_id, cam_keys, experiment_path, output_path_suffix, params))
        self.gaze_thread_handler["thread"].start()


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
    
    
    def start_gaze_estimation(self):
        self.label_gaze_status.config(text="Status: Starting Gaze Estimation...")
        
        until_frame_id = int(self.sv_gaze_until_frame_id.get())
        experiment_path = Path(self.sv_gaze_experiment_path.get())
        calib_root_path = Path(self.sv_gaze_calib_path.get())
        output_path_suffix = self.gaze_estimation_output_filename
        gaze_max_frame_idxs = self.gaze_max_frame_idxs
        cam_keys = [key for key, value in self.camera_settings.items() if value["eye_cam"]]

        _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration", root_path=calib_root_path)
        if calib_pkl_path.exists():
            calib_results = self.load_pkl(calib_pkl_path)

        self.gaze_thread_handler["thread_status"] = "starting"
        self.gaze_thread_handler["status_q"] = multiprocessing.Queue()  # Reset the queue for new segmentation run
        self.gaze_thread_handler["thread"] = threading.Thread(target=self.run_gaze_estimation, args=(self.gaze_thread_handler, self.gaze_imgs.copy(), until_frame_id, gaze_max_frame_idxs.copy(), cam_keys, experiment_path, output_path_suffix, calib_results))

        self.gaze_thread_handler["thread"].start()


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
    
    
    def load_calibration_results(self, calib_root_path):
        _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration", root_path=calib_root_path)
        if calib_pkl_path.exists():
            calib_results = self.load_pkl(calib_pkl_path)
            return calib_results
        else:
            print(f"[Gaze] Calibration results file not found: {calib_pkl_path}")
            return None

    # alpha 3 beta -100
    def display_current_gaze_frame(self, *args):
        """Render a simple placeholder visualization for the current gaze frame into the canvases."""
        idx = int(self.sv_gaze_frame_id.get())
        gaze_view_mode = self.sv_gaze_view_mode.get()
        
        vis_data = {}
        
        calib_results = self.load_calibration_results(Path(self.sv_gaze_calib_path.get()))
        if calib_results is None:
            self.canvas_gaze_result.draw()
            return

        for cam_key in self.gaze_cam_frames.keys():
            if cam_key in self.gaze_imgs.keys() and idx in self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"].keys() and self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][idx] is not None:
                try:
                    # Load and display image
                    fname = self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][idx]["filename"]
                    img_path = self.gaze_imgs[cam_key]["cam_path"] / fname
                    imgdata = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    imgdata = cv2.undistort(imgdata, cameraMatrix=calib_results["intrinsics"][cam_key]["K"], distCoeffs=calib_results["intrinsics"][cam_key]["dist"])
                    
                    if cam_key != 'sc':  #  Don't run segmentation on scene camera
                        imgdata2 = np.zeros_like(imgdata)
                        if gaze_view_mode == 'Segmentation':

                            cam_tilt_angle_deg = self.estimate_camera_tilt_angle(calib_results, cam_key)

                            try:
                                res, res_visualization = self.detect_pupil_ellseg(imgdata, float(self.sv_gaze_alpha.get()), float(self.sv_gaze_beta.get()), .5, cam_tilt_angle_deg)
                                
                                imgdata = res["image"]
                                imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)  # TODO appears to operate in place
                                imgdata = cv2.ellipse(imgdata, (int(res["center"][0]), int(res["center"][1])), (int(res["axes"][0]), int(res["axes"][1])), res["angle_deg"], 0, 360, (0, 0, 255), 1)
                                
                                imgdata2 = res_visualization["image"]
                                imgdata2 = cv2.cvtColor(imgdata2, cv2.COLOR_GRAY2RGB)
                                imgdata2 = cv2.ellipse(imgdata2, (int(res_visualization["center"][0]), int(res_visualization["center"][1])), (int(res_visualization["axes"][0]), int(res_visualization["axes"][1])), res_visualization["angle_deg"], 0, 360, (0, 0, 255), 1)
                            except Exception as e:
                                print(f"[Gaze] Pupil detection error for {cam_key} frame {idx}: {e}")
                                imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
                                imgdata = cv2.putText(imgdata, "Segmentation Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                imgdata2 = imgdata.copy()
                        
                        elif gaze_view_mode == 'Gaze Estimation':
                            imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)  # Convert to RGB for visualization if not done before
                        
                        if gaze_view_mode in ['Segmentation', 'Gaze Estimation']:                            
                            # load json data for segmentation overlay
                            segmentation_path = self.gaze_imgs[cam_key]["segmentation_path"] / fname
                            segmentation_path = segmentation_path.with_suffix(".json")
                            if segmentation_path.exists():
                                with open(segmentation_path, "r") as f:
                                    seg_data = json.load(f)
                                    vis_data[cam_key] = seg_data
                                imgdata = cv2.ellipse(imgdata, (int(seg_data["center"][0]), int(seg_data["center"][1])), (int(seg_data["axes"][0]), int(seg_data["axes"][1])), seg_data["angle_deg"], 0, 360, (0, 255, 0), 1)
                    else:
                        imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
                    
                    self.gaze_cam_frames[cam_key]["img_store_raw"] = imgdata.copy()  # Store raw image
                    
                    if cam_key != 'sc':  # scene camera gets larger preview
                        imgdata = self.resize_and_pad_image(imgdata, 120, 120)
                        imgdata2 = self.resize_and_pad_image(imgdata2, 120, 120)

                        imgdata2 = cv2.imencode(".png", imgdata2)[1].tobytes()
                        img_tk2 = PhotoImage(data=imgdata2)
                        self.gaze_cam_frames[cam_key]["img_store2"] = img_tk2  # prevent garbage collection
                        self.gaze_cam_frames[cam_key]["canvas2"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames[cam_key]["img_store2"])
                    else:
                        imgdata = self.resize_and_pad_image(imgdata, 240, 240)
                    
                    imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
                    img_tk = PhotoImage(data=imgdata)
                    self.gaze_cam_frames[cam_key]["img_store"] = img_tk  # prevent garbage collection
                    self.gaze_cam_frames[cam_key]["canvas"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames[cam_key]["img_store"])

                    print(f"[Gaze] Displayed image for {cam_key} frame {idx}: {img_path}")
                except Exception as e:
                    print(f"[Gaze] Error loading image for {cam_key} frame {idx}: {e}")
                    self.gaze_cam_frames[cam_key]["canvas"].delete("all")
                    self.gaze_cam_frames[cam_key]["canvas"].create_rectangle(2, 2, 118, 118, outline='red')
                    self.gaze_cam_frames[cam_key]["canvas"].create_text(60, 60, text="Error", fill='red', font=("Courier", 10), anchor='center')
            else:
                fname = "N/A"
                self.gaze_cam_frames[cam_key]["canvas"].delete("all")
                self.gaze_cam_frames[cam_key]["canvas"].create_rectangle(2, 2, 118, 118, outline='white')
                self.gaze_cam_frames[cam_key]["canvas"].create_text(60, 60, text="N/A", fill='white', font=("Courier", 10), anchor='center')
            
            self.gaze_cam_frames[cam_key]["info"].config(text=f"{cam_key.upper()}: {fname}\nFrame idx: {idx}\nMax idx: {self.gaze_max_frame_idxs[cam_key]}")
        
        
        if gaze_view_mode == 'Gaze Estimation':
            
            fig = self.canvas_gaze_result.figure
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            
            imgdata = self.gaze_cam_frames["sc"]["img_store_raw"].copy()
            img_dims = imgdata.shape
            sc_distortion = calib_results["intrinsics"]["sc"]["dist"]
            sc_K = calib_results["intrinsics"]["sc"]["K"]
            imgdata = cv2.undistort(imgdata, cameraMatrix=sc_K, distCoeffs=sc_distortion)
            
            # Load gaze estimation json results for the current frame
            gaze_estimation_path = Path(self.sv_gaze_experiment_path.get()) / self.gaze_estimation_output_filename
            if gaze_estimation_path.exists():
                gaze_results_data = self.load_pkl(gaze_estimation_path)
                fit_results = gaze_results_data.get("fit_results", {})
            
                for eye in ["left", "right"]:
                    if fit_results[eye] is not None:
                        C_eye = fit_results[eye]["center"]
                        R_eye = fit_results[eye]["radius"]
                        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                        x = C_eye[0] + R_eye * np.cos(u) * np.sin(v)
                        y = C_eye[1] + R_eye * np.sin(u) * np.sin(v)
                        z = C_eye[2] + R_eye * np.cos(v)
                        ax.plot_wireframe(x, y, z, color='k', alpha=0.3)
                        
                        # Plot optical axes
                        if idx in fit_results[eye]["optical_dirs"] and idx in fit_results[eye]["origins"]:
                            origin = fit_results[eye]["origins"][idx]
                            optical_dir = fit_results[eye]["optical_dirs"][idx]
                            ax.quiver(origin[0], origin[1], origin[2], optical_dir[0], optical_dir[1], optical_dir[2], length=30.0, color='r' if eye == 'left' else 'b', alpha=0.5)
                            
                            if "sc" in self.camera_settings.keys() and "sc" in calib_results["extrinsics"]:
                                print(optical_dir)
                                optical_dir_from_sc = calib_results["extrinsics"]["sc"]["origin"] + optical_dir*1000
                                optical_dir_homog = np.append(optical_dir_from_sc, 1)  # Convert to homogeneous coordinates
                                sc_T = calib_results["extrinsics"]["sc"]["T"]
                                sc_M = sc_K @ sc_T
                                optical_dir_proj = sc_M @ optical_dir_homog
                                optical_dir_proj /= optical_dir_proj[2]  # Normalize to get pixel coordinates
                                optical_dir_proj = optical_dir_proj[:2]# + (np.array(img_dims[:2]) / 2)  # Get x, y pixel coordinates
                                print(f"[Gaze] Projected optical direction into scene camera image plane: {eye} {optical_dir_proj}")
                                optical_dir_proj[0] = np.clip(optical_dir_proj[0], 0, img_dims[1] - 1)
                                optical_dir_proj[1] = np.clip(optical_dir_proj[1], 0, img_dims[0] - 1)
                                if eye == "left":
                                    color = (0, 0, 255)  # Red for left eye
                                else:
                                    color = (255, 0, 0)  # Blue for right eye
                                imgdata = cv2.circle(imgdata, (int(optical_dir_proj[0]), int(optical_dir_proj[1])), 10, color, -1)

            else:
                print(f"[Gaze] Gaze estimation results file not found: {gaze_estimation_path}")
                
            # Update canvas with scene camera image and projected optical axes
            imgdata = self.resize_and_pad_image(imgdata, 240, 240)
            imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
            img_tk = PhotoImage(data=imgdata)
            self.gaze_cam_frames["sc"]["img_store"] = img_tk  # prevent garbage collection
            self.gaze_cam_frames["sc"]["canvas"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames["sc"]["img_store"])

            # plot full extrinsics from calibration results
            for cam_key in self.camera_settings.keys():
                if cam_key in calib_results["extrinsics"]:
                    extrinsics = calib_results["extrinsics"][cam_key]
                    ax.quiver(extrinsics["origin"][0],
                              extrinsics["origin"][1],
                              extrinsics["origin"][2],
                              extrinsics["z"][0],
                              extrinsics["z"][1],
                              extrinsics["z"][2],
                              length=10, color='b', arrow_length_ratio=0.1)
                    ax.quiver(extrinsics["origin"][0],
                              extrinsics["origin"][1],
                              extrinsics["origin"][2],
                              extrinsics["y"][0],
                              extrinsics["y"][1],
                              extrinsics["y"][2],
                              length=5, color='g', arrow_length_ratio=0.1)
                        
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_zlabel('Z [mm]')
            ax.set_title('Fitted Eye Spheres and Optical Axes')
            ax.set_aspect('equal')
            self.canvas_gaze_result.draw()
            
        elif gaze_view_mode == 'Segmentation':
            # Load full calibration results for gaze estimation visualization
            fig = self.canvas_gaze_result.figure
            fig.clf()  # Clear the figure to avoid overlapping plots
            
            _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration")
            if calib_pkl_path.exists():
                calib_results = self.load_pkl(calib_pkl_path)
                
                cam_key_pairs = [("ro", "ri"), ("lo", "li")]
                triang_calib_data = [calib_results["calibration_steps"]["Stereo R-R"]["data"], calib_results["calibration_steps"]["Stereo L-L"]["data"]]
                subplot_idxs = [211, 212]

                for cam_pair, calib_data, subplot_idx in zip(cam_key_pairs, triang_calib_data, subplot_idxs):
                    if all(cam_key in vis_data for cam_key in cam_pair):
                        point_coords = [vis_data[cam_key]["center"] for cam_key in cam_pair]
                        point_wc, (cam1_point_vector_wc, cam2_point_vector_wc), projection_error = triangulate_point(calib_data, point_coords)

                        if point_wc is not None:
                            
                            if cam_pair == ("lo", "li"):
                                T = transformation_matrix_from_calib_ext(calib_results["calibration_steps"]["Stereo R-L"]["data"]["camera_params_1"].extrinsic)
                                point_wc = T * np.matrix(np.append(point_wc, 1)).T  # Convert to homogeneous coordinates for transformation
                                point_wc = point_wc[:3]  # Convert back to 3D coordinates

                            ax = fig.add_subplot(subplot_idx, projection='3d')

                            # Triangulated point
                            ax.scatter(point_wc[0], point_wc[1], point_wc[2], c='r', marker='o')
                            
                            extrinsics_cam1 = calib_results["extrinsics"][cam_pair[0]]
                            extrinsics_cam2 = calib_results["extrinsics"][cam_pair[1]]

                            # Cameras
                            ax.quiver(extrinsics_cam1["origin"][0],
                                    extrinsics_cam1["origin"][1],
                                    extrinsics_cam1["origin"][2],
                                    extrinsics_cam1["z"][0],
                                    extrinsics_cam1["z"][1],
                                    extrinsics_cam1["z"][2],
                                    length=10, color='b', arrow_length_ratio=0.1)
                            ax.quiver(extrinsics_cam1["origin"][0],
                                    extrinsics_cam1["origin"][1],
                                    extrinsics_cam1["origin"][2],
                                    extrinsics_cam1["y"][0],
                                    extrinsics_cam1["y"][1],
                                    extrinsics_cam1["y"][2],
                                    length=5, color='g', arrow_length_ratio=0.1)

                            ax.quiver(extrinsics_cam2["origin"][0],
                                    extrinsics_cam2["origin"][1],
                                    extrinsics_cam2["origin"][2],
                                    extrinsics_cam2["z"][0],
                                    extrinsics_cam2["z"][1],
                                    extrinsics_cam2["z"][2],
                                    length=10, color='b', arrow_length_ratio=0.1)
                            ax.quiver(extrinsics_cam2["origin"][0],
                                    extrinsics_cam2["origin"][1],
                                    extrinsics_cam2["origin"][2],
                                    extrinsics_cam2["y"][0],
                                    extrinsics_cam2["y"][1],
                                    extrinsics_cam2["y"][2],
                                    length=5, color='g', arrow_length_ratio=0.1)

                            ax.set_xlabel('X [mm]')
                            ax.set_ylabel('Y [mm]')
                            ax.set_zlabel('Z [mm]')
                            ax.set_title(f'3D Point (Frame {idx}, Pair: {cam_pair}, Projection Error: {projection_error:.2f} mm)')
                            ax.set_aspect('equal')

                            fig.tight_layout()
                        else:
                            ax = fig.add_subplot(subplot_idx)
                            ax.text(0.5, 0.5, f"Triangulation Failed for {cam_pair}", ha='center', va='center', fontsize=8, color='red')
                            ax.axis('off')
                    else:
                        ax = fig.add_subplot(subplot_idx)
                        ax.text(0.5, 0.5, f"Insufficient Data for Triangulation for {cam_pair}", ha='center', va='center', fontsize=8, color='red')
                        ax.axis('off')
            else:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "Calibration Results Not Found", ha='center', va='center', fontsize=12, color='red')
                ax.axis('off')
            
            self.canvas_gaze_result.draw()
            
        self.gaze_info.config(text=f"Gaze: idx={idx}")
            
            
    def onchange_gaze_frame_id(self, *args):
        idx_str = self.sv_gaze_frame_id.get()
        idx = int(idx_str)
        if 0 <= idx <= max(self.gaze_max_frame_idxs.values()):
            self.gaze_current_frame_idx = idx
            self.display_current_gaze_frame()
        else:
            print(f"[Gaze] Frame index out of range: {idx}")


    def goto_frame_prev(self):
        self.gaze_current_frame_idx = max(0, self.gaze_current_frame_idx - 1)
        self.sv_gaze_frame_id.set(str(self.gaze_current_frame_idx))  # Side effect: triggers onchange event to display first frame


    def goto_frame_next(self):
        self.gaze_current_frame_idx = min(max(self.gaze_max_frame_idxs.values()), self.gaze_current_frame_idx + 1)
        self.sv_gaze_frame_id.set(str(self.gaze_current_frame_idx))  # Side effect: triggers onchange event to display first frame


    def _gaze_anim_step(self):
        # increment and schedule next
        new_gaze_frame_idx = (self.gaze_current_frame_idx + 1) % max(self.gaze_max_frame_idxs.values())
        self.sv_gaze_frame_id.set(str(new_gaze_frame_idx))  # Side effect: triggers onchange event to display first frame
        # schedule next step
        self.gaze_anim_after_id = self.root.after(int(1000/44.5), self._gaze_anim_step)


    def toggle_animate_gaze(self):
        if getattr(self, 'gaze_animating', False):
            # stop
            if getattr(self, 'gaze_anim_after_id', None) is not None:
                try:
                    self.root.after_cancel(self.gaze_anim_after_id)
                except Exception:
                    pass
                self.gaze_anim_after_id = None
            self.gaze_animating = False
            try:
                self.btn_animate_gaze.config(text='Animate')
            except Exception:
                pass
        else:
            # start
            self.gaze_animating = True
            try:
                self.btn_animate_gaze.config(text='Stop')
            except Exception:
                pass
            self.gaze_anim_after_id = self.root.after(200, self._gaze_anim_step)
    
    
    def get_current_capture_preset_name(self):
        return self.capture_preset.get()
    
    
    def get_current_calibration_preset_name(self):
        return self.calib_preset.get()
    

    def get_preset_for_step(self, step):
        return self.capture_calibration_presets["presets"][step]


    def get_current_calibration_preset(self):
        return self.get_preset_for_step(self.get_current_calibration_preset_name())


    def get_root_path(self, step=None, idx=None):
        path = Path(self.sv_experiment_name.get())
        if step is not None:
            path /= self.get_preset_for_step(step)["folder_path"]
        if idx is not None:
            path /= self.get_preset_for_step(step)["camera_indices"][idx]
        return path
    
    
    def get_calibration_output_pkl_path(self, step, root_path=None):
        if root_path is None:
            root_path = self.get_root_path(step)
        else:
            root_path = Path(root_path)
            root_path /= self.get_preset_for_step(step)["folder_path"]
        output_dir = root_path / self.calibration_settings["calib_output_directory"]
        output_file_path = output_dir / (self.calibration_settings["calib_output_pkl_filename"])
        output_file_path = output_file_path.with_suffix(".pkl")
        return output_dir, output_file_path
    
    
    def load_pkl(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'rb') as handle:
            data = pickle.load(handle)
            return data
        
        
    def load_json(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as handle:
            data = json.load(handle)
            return data


    def get_calibration_input_paths(self, step):
        path_root = self.get_root_path(step)
        paths_cams = [path_root / camera_idc for camera_idc in self.get_preset_for_step(step)["camera_indices"]]
        return paths_cams
    
    
    def get_pattern_settings_for_step(self, step):
        pattern_type = self.get_preset_for_step(step)["use_pattern"]
        if step in ["Mono SC", "Mirror SC"]:
            pixel_pitch_um = self.calibration_settings["pixel_pitch_sc_um"]
        else:
            pixel_pitch_um = self.calibration_settings["pixel_pitch_um"]
        if pattern_type is not None:
            pattern_settings = self.calibration_settings["patterns"][pattern_type]
            return pattern_type, pattern_settings["corners"], pattern_settings["square_size_mm"], pixel_pitch_um
        else:
            return pattern_type, None, None, pixel_pitch_um


    def run_calibration_step(self, step=None):
        if step is None:
            step = self.get_current_calibration_preset_name()
        
        print(f"Running calibration step: {step}")
        
        # Run Calibration

        #try:
        if step in ["Stereo R-R", "Stereo L-L", "Stereo R-L"]:
            self.run_stereo_calibration(step=step)
            
        elif step == "Mono SC":
            self.run_mono_calibration(step=step)

        elif step in ["Mirror R", "Mirror L"]:
            self.calib_triangulate_points_for_step_and_file(step=step, save_output=True)

        elif step == "Mirror SC":
            self.calibrate_scene_camera_extrinsics(step=step, save_output=True)

        elif step == "Full Calibration":
            self.save_full_calibration()
        
        else:
            raise ValueError(f"Unknown calibration step: {step}")
            
        # except Exception as e:
        #     print(f"Error running calibration step {step}: {e}")
        #     return False
        
        # Visualize results
        self.plot_calibration_step_results(step)
        

    def calculate_and_plot_extrinsic_calibration(self, until_step=None):
        data_calib_steps, _, _ = self.calculate_full_calibration(until_step=until_step)
        self.plot_extrinsic_calibration_results(data_calib_steps)


    def set_calibration_step_visualization_empty(self):
        fig = self.canvas_calib_1.figure
        fig.clf()
        self.canvas_calib_1.draw()
        

    def plot_calibration_step_results(self, step):
        if step in ["Stereo R-R", "Stereo L-L", "Stereo R-L"]:
            self.plot_stereo_calibration_results(step)
        elif step == "Mono SC":
            self.plot_mono_calibration_results(step)
        elif step in ["Mirror R", "Mirror L"]:
            self.plot_stereo_points_calibration_results(step)
        elif step == "Mirror SC":
            #TODO self.plot_scene_camera_extrinsics_calibration_results(step) (carve out logic)
            pass
        else:
            print(f"No visualization implemented for step {step}")
            self.set_calibration_step_visualization_empty()

        self.calculate_and_plot_extrinsic_calibration(until_step=step)
    
    
    def visualize_calibration_results(self):
        step = self.get_current_calibration_preset_name()
        self.plot_calibration_step_results(step)

    
    def run_calibration_steps_until(self):
        goal_step = self.get_current_calibration_preset_name()
        for step in self.capture_calibration_presets["calib_preset_order"]:
            self.run_calibration_step(step)
            if step == goal_step:
                break


    def run_stereo_calibration(self, step=None, verbose=True):
        if step is None:
            step = self.get_current_calibration_preset_name()

        cam_paths = self.get_calibration_input_paths(step)
        output_dir, _ = self.get_calibration_output_pkl_path(step)
        output_filename = self.calibration_settings["calib_output_pkl_filename"]
        _, pattern_size, square_size_mm, pixel_pitch_um = self.get_pattern_settings_for_step(step)
        
        summary = run_calibration(
            left_dir=cam_paths[0],
            right_dir=cam_paths[1],
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            pixel_pitch_um=pixel_pitch_um,
            out_dir=output_dir,
            out_filename=output_filename,
            left_dir_mono=None,
            right_dir_mono=None,
            verbose=verbose,
        )
        
        self.calibration_status = "finished"


    def run_mono_calibration(self, step=None, verbose=True):
        if step is None:
            step = self.get_current_calibration_preset_name()

        cam_paths = self.get_calibration_input_paths(step)
        output_dir, output_file = self.get_calibration_output_pkl_path(step)
        _, pattern_size, square_size_mm, pixel_pitch_um = self.get_pattern_settings_for_step(step)
        
        rms, K, dist, rvecs, tvecs, map1, map2, errs_mono_reproj_initial, errs_mono_reproj = run_mono_calibration(
            dir=cam_paths[0],
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            pixel_pitch_um=pixel_pitch_um,
            verbose=verbose,
        )
        
        mono_calibration_data = {
            "rms": rms,
            "K": K,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "map1": map1,
            "map2": map2,
            "errs_mono_reproj_initial": errs_mono_reproj_initial,
            "errs_mono_reproj": errs_mono_reproj,
        }

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'wb') as handle:
            pickle.dump(mono_calibration_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Mono calibration data saved to {output_file}")


    def calibrate_scene_camera_extrinsics(self, step=None, input_filename=None, live=False, save_output=False):
        if step is None:
            step = self.get_current_calibration_preset_name()
            
        if step != "Mirror SC":
            print(f"Scene camera extrinsics calibration only implemented for Mirror SC step, current step: {step}")
            return None

        cam_path = self.get_calibration_input_paths(step)[0]
        output_dir, output_file = self.get_calibration_output_pkl_path(step)
        _, pattern_size, square_size_mm, _ = self.get_pattern_settings_for_step(step)

        img_dir = self.get_img_dirs(step)[0]

        # Load from file
        if input_filename is None:
            # recursively run calibration for all files in folders
            img_files_per_folder = self.get_files_in_dirs([img_dir])[0]
            scene_camera_extrinsics_list = [self.calibrate_scene_camera_extrinsics(step=step, input_filename=img_file, live=False) for img_file in img_files_per_folder]

            # TODO: Combine results from all files
            sc_extrinsics_calibration_data = scene_camera_extrinsics_list[0] # Placeholder: return first result for now, implement combination logic as needed

        else:
            if live:
                # Collect points positions
                label_coordinates = self.label_frames[0]['label_coordinates_store']
                img = self.label_frames[0]["img_store_raw"]
            else:
                try:
                    label_coordinates = self.load_pkl(self.get_file_path_with_suffix(cam_path / input_filename, self.calibration_settings["calib_labels_pkl_suffix"]))["label_coordinates"]
                except Exception as e:
                    print(f"[CSCE] Error loading label coordinates from {input_filename}: {e}")
                    return None

                img_path = img_dir / input_filename
                img = cv2.imread(img_path)
                #img = self.resize_and_pad_image(img, 240, 240)

            point_labels = {}
            for key in label_coordinates.keys():
                key_coordinates = label_coordinates[key]
                if np.all([coordinate_i >= 0 for coordinate_i in key_coordinates]):
                    if key in self.label_types_configs["MIRROR_SC"]["available_labels"]:
                        point_labels[key] = key_coordinates
                    else:
                        print(f"Label {key} not in available labels for MIRROR_SC")
                else:
                    print(f"Label {key} has invalid coordinates: {key_coordinates[0]}")
            
            # This seems like a good use case for PnP, but we can't use that here because all cameras are roughly in line.
            
            # Load the mono calibration data
            mono_calibration_file = self.get_calibration_output_pkl_path("Mono SC")[1]
            mono_calibration_data = self.load_pkl(mono_calibration_file)
            pixel_pitch_sc_mm = (self.calibration_settings["pixel_pitch_sc_um"][0] / 1000)
            focal_length_mm = mono_calibration_data["K"][0, 0] * pixel_pitch_sc_mm  # Convert pixel focal length to mm using pixel pitch
            img_dimensions = (img.shape[1], img.shape[0])  # (width, height)
            print(f"Mono calibration data loaded from {mono_calibration_file}, focal length in mm: {focal_length_mm}, image dimensions: {img_dimensions}")
            
            # 1 Detect pattern corners in the image
            corners = detect_corners(img, pattern_size, verbose=True)
            if corners is None:
                print(f"No corners detected in image {input_filename}")
                return None
            image_points_2d_mirror = corners.squeeze()
            
            # Labelled Data
            image_points_2d_cameras = np.array([
                point_labels["CAM_RO"],
                point_labels["CAM_RI"],
                point_labels["CAM_LI"],
                point_labels["CAM_LO"]
            ], dtype=np.float32).squeeze()
            
            # Perpendicular point of mirror (reflection of scene camera)
            image_point_2d_sc = np.array([
                point_labels["CAM_SC"]
            ], dtype=np.float32).squeeze()

            # 2 Undistort the image points using the mono calibration data
            image_points_2d_mirror_undist = cv2.undistortPoints(image_points_2d_mirror, mono_calibration_data["K"], mono_calibration_data["dist"], P=mono_calibration_data["K"]).squeeze()
            image_points_2d_cameras_undist = cv2.undistortPoints(image_points_2d_cameras, mono_calibration_data["K"], mono_calibration_data["dist"], P=mono_calibration_data["K"]).squeeze()
            image_point_2d_sc_undist = cv2.undistortPoints(image_point_2d_sc, mono_calibration_data["K"], mono_calibration_data["dist"], P=mono_calibration_data["K"]).squeeze()

            # Generate world points based on pattern settings. assume z=0 for all points, and x,y based on square size and pattern size
            world_points_3d_mirror = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
            for i in range(pattern_size[1]):
                for j in range(pattern_size[0]):
                    world_points_3d_mirror[i * pattern_size[0] + j] = [(j * square_size_mm), (-i * square_size_mm), 0]

            # 3 Use pattern points to get mirror x/y scaling
            
            ## Place Mirror in 3D space using PnP with the detected corners and known pattern geometry
            _, rvec_mirror, tvec_mirror = cv2.solvePnP(world_points_3d_mirror, image_points_2d_mirror, mono_calibration_data["K"], mono_calibration_data["dist"], flags=cv2.SOLVEPNP_ITERATIVE)
            T_mirror = np.eye(4)
            T_mirror[:3, :3] = cv2.Rodrigues(rvec_mirror)[0]
            T_mirror[:3, 3] = tvec_mirror.flatten()
            from scipy.spatial.transform import Rotation as R
            print(f"Estimated mirror pose (rotation vector): {rvec_mirror.flatten()}, translation vector: {tvec_mirror.flatten()}, Euler angles (degrees): {R.from_matrix(cv2.Rodrigues(rvec_mirror)[0]).as_euler('xyz', degrees=True)}")

            # Project PnP Result from below to get actual scaling factor in x and y direction, otherwise we would assume no tilt and that the pattern is parallel to the camera plane, which is not necessarily the case
            world_points_3d_mirror_projected, _ = cv2.projectPoints(world_points_3d_mirror, rvec_mirror, tvec_mirror, mono_calibration_data["K"], mono_calibration_data["dist"])
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
            
            _, ext_calib_summary, _ = self.calculate_full_calibration(until_step="Mirror SC")
            world_points_3d_frames = np.array([
                ext_calib_summary["ro"]["origin"].A1[:3],
                ext_calib_summary["ri"]["origin"].A1[:3],
                ext_calib_summary["li"]["origin"].A1[:3],
                ext_calib_summary["lo"]["origin"].A1[:3]
            ], dtype=np.float32)
            
            origin_sc = ext_calib_summary["mr"]["CAM_SC"][:3]
            print(f"Scene camera origin: {origin_sc}")
            
            # Get closest distance between mirror and scene camera based on the labelled point for the scene camera and the mirror plane
            from skspatial.objects import Line, Plane, Sphere
            plane_mirror = Plane(point=T_mirror[:3, 3], normal=T_mirror[:3, 2])  # Mirror plane defined by its origin and normal, assuming the normal is along the z-axis of the mirror's coordinate system
            line_sc_normal = Line(point=np.array([0, 0, 0]), direction=np.array([(image_point_2d_sc_undist[0] - img_dimensions[0]/2) * pixel_pitch_sc_mm, (image_point_2d_sc_undist[1] - img_dimensions[1]/2) * pixel_pitch_sc_mm, focal_length_mm]))  # Line from the scene camera point in the direction of the camera's view
            #TODO this is not really needed, we can just calculate the distance from the scene camera point to the mirror plane along the normal direction (crossing 0,0,0), but this is a good sanity check to see if the reflected camera points and the mirror plane are consistent with the labelled scene camera point
            intersection = plane_mirror.intersect_line(line_sc_normal)
            distance_mirror_to_sc = np.linalg.norm(intersection - origin_sc[:3])
            print(f"Closest distance between mirror and scene camera: {distance_mirror_to_sc}")

            # Calculate possible orientations for the scene camera based on the reflected camera points and the mirror plane
            distance_cameras_to_sc = np.linalg.norm(world_points_3d_frames - origin_sc, axis=1)  # Vectors from scene camera to each of the camera label points
            origin_sc_scwc = np.array([0, 0, 0])
            origin_scd_scwc = np.array([0, 0, 2*distance_mirror_to_sc])
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
            
            c1_sc_direction = origin_sc - world_points_3d_frames[0]
            c2_sc_direction = origin_sc - world_points_3d_frames[3]

            #print(f"Direction from camera 1 to scene camera: {c1_sc_direction}, direction from camera 2 to scene camera: {c2_sc_direction}")
            
            #print(f"Angle between camera 1 and camera 2 directions: {np.rad2deg(angle_between_vectors(c1_sc_direction, c2_sc_direction))} degrees")
            #print(f"Angle between camera 1 and camera 2 directions: {np.rad2deg(angle_between_vectors(-valid_position[0], -valid_position[3]))} degrees")

            angle_scd_scc1 = angle_between_vectors(-valid_position[0], np.array([0,0,1]))
            angle_scd_scc2 = angle_between_vectors(-valid_position[3], np.array([0,0,1]))
            angle_scy_scc1 = angle_between_vectors(-valid_position[0], np.array([0,1,0]))
            angle_scy_scc2 = angle_between_vectors(-valid_position[3], np.array([0,1,0]))
            
            #print(f"Valid camera position based on external calibration: {valid_position}")

            #print(f"Angles from valid position 1 to camera 1 direction: {np.rad2deg(angle_scd_scc1)} (to z), {np.rad2deg(angle_scy_scc1)} (to y)")
            #print(f"Angles from valid position 2 to camera 2 direction: {np.rad2deg(angle_scd_scc2)} (to z), {np.rad2deg(angle_scy_scc2)} (to y)")

            # Estimate scene camera direction based on angles
            estimated_sc_z = find_vector_by_angles(c1_sc_direction, c2_sc_direction, angle_scd_scc1, angle_scd_scc2)
            estimated_sc_y = find_vector_by_angles(c1_sc_direction, c2_sc_direction, angle_scy_scc1, angle_scy_scc2)
            
            # Choose solution with negative z component (pointing away from the cameras)
            estimated_sc_z = [sol for sol in estimated_sc_z if sol[2] < 0][0]
            # Choose solution with negative y component (eye cameras are upside down)
            estimated_sc_y = [sol for sol in estimated_sc_y if sol[1] < 0][0]
            
            estimated_sc_T = np.eye(3, 4)
            estimated_sc_x = np.cross(estimated_sc_y, estimated_sc_z)
            estimated_sc_T[:3, :3] = np.column_stack((estimated_sc_x, estimated_sc_y, estimated_sc_z)).T  # Need inverse because we want to go from world to camera coordinates
            estimated_sc_T[:3, 3] = origin_sc

            print(f"Estimated scene camera direction: {estimated_sc_z}, estimated scene camera y direction: {estimated_sc_y}, estimated scene camera transformation matrix: {estimated_sc_T}")

            # Plot results
            # TODO own function
            gs0 = plt.GridSpec(2,2, height_ratios=[1,2])
            fig = self.canvas_calib_1.figure
            fig.clf()
            
            
            # Plot undistorted points
            ax = fig.add_subplot(gs0[0,0])
            ax.scatter(image_points_2d_mirror_undist[:, 0], -image_points_2d_mirror_undist[:, 1], c='r', marker='o', label='Mirror Pattern')
            ax.scatter(image_points_2d_cameras_undist[:, 0], -image_points_2d_cameras_undist[:, 1], c='g', marker='x', label='Camera Labels')
            ax.scatter(image_point_2d_sc_undist[0], -image_point_2d_sc_undist[1], c='b', marker='s', label='Scene Camera Label')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title('Undistorted Points')
            ax.legend()
            # TODO Would be neat to overlay undistorted image, but cv2 undistortion changes image dimensions, so we would need to remap the image first and then overlay the points on top of it, but for now we just plot the points

            # Plot undistorted image
            ax = fig.add_subplot(gs0[0,1])
            # plot img remapped according to distortion map
            img_undistorted = cv2.remap(img, mono_calibration_data["map1"], mono_calibration_data["map2"], interpolation=cv2.INTER_LINEAR)
            ax.imshow(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title("Undistorted Image")

            # 3D plot of the scene camera, mirror, and possible camera positions
            ax = fig.add_subplot(gs0[1,0], projection='3d')

            mirror_size = 100  # Size of the mirror plane for visualization
            mirror_corners = np.array([
                [-mirror_size, -mirror_size, 0],
                [mirror_size, -mirror_size, 0],
                [mirror_size, mirror_size, 0],
                [-mirror_size, mirror_size, 0]
            ])
            mirror_corners_world = (T_mirror[:3, :3] @ mirror_corners.T).T + T_mirror[:3, 3]
            ax.plot_trisurf(mirror_corners_world[:, 0], mirror_corners_world[:, 1], mirror_corners_world[:, 2], color='c', alpha=0.5, label='Mirror')
            
            # sc normal
            ax.quiver(line_sc_normal.point[0], line_sc_normal.point[1], line_sc_normal.point[2], line_sc_normal.direction[0], line_sc_normal.direction[1], line_sc_normal.direction[2], length=20, color='m')

            # Sphere of possible camera positions based on external calibration
            # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            # x_sphere = sphere_possible_camera_positions_from_ext.point[0] + sphere_possible_camera_positions_from_ext.radius * np.cos(u) * np.sin(v)
            # y_sphere = sphere_possible_camera_positions_from_ext.point[1] + sphere_possible_camera_positions_from_ext.radius * np.sin(u) * np.sin(v)
            # z_sphere = sphere_possible_camera_positions_from_ext.point[2] + sphere_possible_camera_positions_from_ext.radius * np.cos(v)
            # ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='r', alpha=0.5, label='Possible Camera Positions (External Calib)')
            
            ax.quiver(0, 0, 0, 0, 0, 1, length=100, color='m', label='SC (Z)')
            ax.quiver(0, 0, 0, 0, 1, 0, length=50, color='y', label='SC (Y)')
            
            # TODO wrong, needs to be the inverse orientation.
            #ax.quiver(0, 0, 0, estimated_sc_z[0], estimated_sc_z[1], estimated_sc_z[2], length=100, color='g', label='RO (Z)')
            #ax.quiver(0, 0, 0, estimated_sc_y[0], estimated_sc_y[1], estimated_sc_y[2], length=50, color='c', label='RO (Y)')

            # origin scd scwc
            ax.scatter(origin_scd_scwc[0], origin_scd_scwc[1], origin_scd_scwc[2], c='g', marker='o', label="SC'")

            # Line of possible camera positions based on the scene camera label point
            for i in range(len(lines_possible_camera_positions_from_sc)):
                line_points = np.array([lines_possible_camera_positions_from_sc[i].point + t * lines_possible_camera_positions_from_sc[i].direction for t in np.linspace(0, 3, 2)])
                ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], color='b')
                if possible_camera_positions[i] is not None:
                    ax.scatter(possible_camera_positions[i][0][0], possible_camera_positions[i][0][1], possible_camera_positions[i][0][2], c='r', marker='^')
                    ax.scatter(possible_camera_positions[i][1][0], possible_camera_positions[i][1][1], possible_camera_positions[i][1][2], c='r', marker='v')

            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Scene camera coordinate system')
            ax.set_box_aspect([1,1,1])  # Equal aspect ratio
            ax.legend()

            # 3D plot of the camera label points and their directions in world coordinates
            ax = fig.add_subplot(gs0[1,1], projection='3d')

            cam_zs = np.array([
                ext_calib_summary["ro"]["z"].A1[:3],
                ext_calib_summary["ri"]["z"].A1[:3],
                ext_calib_summary["li"]["z"].A1[:3],
                ext_calib_summary["lo"]["z"].A1[:3]
            ], dtype=np.float32)
            
            cam_ys = np.array([
                ext_calib_summary["ro"]["y"].A1[:3],
                ext_calib_summary["ri"]["y"].A1[:3],
                ext_calib_summary["li"]["y"].A1[:3],
                ext_calib_summary["lo"]["y"].A1[:3]
            ], dtype=np.float32)

            ax.scatter(world_points_3d_frames[:, 0], world_points_3d_frames[:, 1], world_points_3d_frames[:, 2], c='r', marker='o', label='Cams')
            ax.quiver(world_points_3d_frames[:, 0], world_points_3d_frames[:, 1], world_points_3d_frames[:, 2], cam_zs[:, 0], cam_zs[:, 1], cam_zs[:, 2], length=20, color='g', label='Cams (Z)')
            ax.quiver(world_points_3d_frames[:, 0], world_points_3d_frames[:, 1], world_points_3d_frames[:, 2], cam_ys[:, 0], cam_ys[:, 1], cam_ys[:, 2], length=10, color='b', label='Cams (Y)')
            ax.scatter(origin_sc[0], origin_sc[1], origin_sc[2], c='m', marker='^', label='SC')
            ax.quiver(origin_sc[0], origin_sc[1], origin_sc[2], estimated_sc_z[0], estimated_sc_z[1], estimated_sc_z[2], length=20, color='c', label='SC (Z)')
            ax.quiver(origin_sc[0], origin_sc[1], origin_sc[2], estimated_sc_y[0], estimated_sc_y[1], estimated_sc_y[2], length=10, color='y', label='SC (Y)')

            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Frame Extrinsics')
            ax.set_aspect('equal', 'box')
            ax.legend()

            fig.tight_layout()

            self.canvas_calib_1.draw_idle()


            sc_extrinsics_calibration_data = {
                "mono_calibration": mono_calibration_data,
                "sc_calibration_result": {
                    "absolute": {
                        "origin": origin_sc,
                        "z": estimated_sc_z,
                        "y": estimated_sc_y,
                        "T": estimated_sc_T,
                    },
                    "distance_mirror_to_sc": distance_mirror_to_sc,
                    "possible_camera_positions": possible_camera_positions,
                },
                "mirror_pose": {
                    "T_mirror": T_mirror,
                    "rvec_mirror": rvec_mirror,
                    "tvec_mirror": tvec_mirror,
                },
                "label_points": {
                    "image_points_2d_mirror": image_points_2d_mirror,
                    "image_points_2d_cameras": image_points_2d_cameras,
                    "image_point_2d_sc": image_point_2d_sc,
                },
                "undistorted_points": {
                    "image_points_2d_mirror_undist": image_points_2d_mirror_undist,
                    "image_points_2d_cameras_undist": image_points_2d_cameras_undist,
                    "image_point_2d_sc_undist": image_point_2d_sc_undist,
                },
                "scaling_factor": scaling_factor,
                "world_points_3d_mirror": world_points_3d_mirror,
                "world_points_3d_frames": world_points_3d_frames,
            }

        if save_output:
            output_dir, output_file = self.get_calibration_output_pkl_path(step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, 'wb') as handle:
                pickle.dump(sc_extrinsics_calibration_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Scene camera extrinsics calibration data saved to {output_file}")
        
        return sc_extrinsics_calibration_data


    def calib_triangulate_points_for_step_and_file(self, step=None, input_filename=None, live=False, save_output=False):
        # filename == None -> for all in folder, else for specific file (for debugging and visualization purposes). Filename without path.
        if step is None:
            step = self.get_current_calibration_preset_name()

        cam_paths = self.get_calibration_input_paths(step)
        output_dir, output_file = self.get_calibration_output_pkl_path(step)
        _, pattern_size, _, _ = self.get_pattern_settings_for_step(step)
        
        img_dirs = self.get_img_dirs(step)
        
        # Load from file
        if input_filename is None:
            # recursively run triangulation for all files in folders
            img_files_per_folder = self.get_files_in_dirs(img_dirs)
            joined_img_files = self.get_common_files(img_files_per_folder)

            stereo_points_calibration_data_list = [self.calib_triangulate_points_for_step_and_file(step=step, input_filename=img_file, live=False) for img_file in joined_img_files]

            # TODO: Combine results from all files
            stereo_points_calibration_data = stereo_points_calibration_data_list[0] # Placeholder: return first result for now, implement combination logic as needed

        else:
            if live:
                # Collect points positions
                label_coordinates = [self.label_frames[i]['label_coordinates_store'] for i in range(2)]
                imgs = [self.label_frames[i]["img_store_raw"] for i in range(2)]
            else:
                try:
                    label_coordinates = [self.load_pkl(self.get_file_path_with_suffix(cam_path / input_filename, self.calibration_settings["calib_labels_pkl_suffix"]))["label_coordinates"] for cam_path in cam_paths]
                except Exception as e:
                    print(f"[TPM] Error loading label coordinates from {input_filename}: {e}")
                    return None

                img_paths = [img_dir / input_filename for img_dir in img_dirs]
                imgs = [cv2.imread(img_path) for img_path in img_paths]
                imgs = [self.resize_and_pad_image(img, 240, 240) for img in imgs]

            points_label_pairs = {}
            camera_label_pairs = {}
            for key in [key for key in label_coordinates[0].keys() if key in label_coordinates[1].keys()]:
                key_coordinates = [label_coordinates[i][key] for i in range(2)]
                if np.all([coordinate_i >= 0 for coordinate in key_coordinates for coordinate_i in coordinate]):
                    if key in self.label_types_configs["MIRROR_STEREO"]["available_labels"]:
                        if key in self.label_types_configs["MIRROR_STEREO"]["source_camera_labels"]:
                            camera_label_pairs[key] = key_coordinates
                        else:
                            points_label_pairs[key] = key_coordinates
                    else:
                        print(f"Label {key} not in available labels for MIRROR_STEREO")
                else:
                    print(f"Label {key} has invalid coordinates: {key_coordinates[0]}, {key_coordinates[1]}")
            
            if step == "Mirror R":
                stereo_calibration_file = self.get_calibration_output_pkl_path("Stereo R-R")[1]
            elif step == "Mirror L":
                stereo_calibration_file = self.get_calibration_output_pkl_path("Stereo L-L")[1]
            else:
                print(f"Unknown step for triangulation: {step}")
                return None

            stereo_calibration_data = self.load_pkl(stereo_calibration_file)

            points_calibration_results_for_mirror_calibs, mirror_calibration_results = triangulate_mirrored_points(stereo_calibration_data, imgs, points_label_pairs, camera_label_pairs, pattern_size)
            cam_pair_calib_vecs = calculate_pairwise_camera_calib_vecs(stereo_calibration_data)

            stereo_points_calibration_data = {
                "stereo_calibration": stereo_calibration_data,
                "cam_pair_calib_vecs": cam_pair_calib_vecs,
                "calibrated_points": points_calibration_results_for_mirror_calibs,
                "mirror_calibration": mirror_calibration_results,
            }

            print(f"Stereo Points Calibration Results: {points_calibration_results_for_mirror_calibs}")

        if save_output:
            output_dir, output_file = self.get_calibration_output_pkl_path(step)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, 'wb') as handle:
                pickle.dump(stereo_points_calibration_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Stereo points calibration data saved to {output_file}")
        
        return stereo_points_calibration_data


    def plot_stereo_points_calibration_results(self, step=None, stereo_points_calibration_data=None):
        if stereo_points_calibration_data is None:
            if step is not None:
                try:
                    stereo_points_calibration_data = self.load_pkl(self.get_calibration_output_pkl_path(step)[1])
                except Exception as e:
                    print(f"Error loading stereo points calibration data for step {step}: {e}")
                    self.set_calibration_step_visualization_empty()
                    return None
            else:
                raise ValueError("Either stereo_points_calibration_data or step must be provided for plotting.")

        fig = self.canvas_calib_1.figure
        fig.clf()
        
        for idx, (mirror_label, mirror_result) in enumerate(stereo_points_calibration_data["mirror_calibration"].items()):
            mirror_surface_point_wc = mirror_result["surface_point"]
            mirror_surface_normal_wc = mirror_result["surface_normal"]

            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{mirror_label}")

            self.plot_camera_pair_calib(stereo_points_calibration_data["cam_pair_calib_vecs"], ax)

            for point_id, result in stereo_points_calibration_data["calibrated_points"][mirror_label].items():
                # TODO use point_id for better labeling and visualization of correspondences between points and cameras
                point_mirrored = result["point_mirrored"]
                point_wc = result["point_world"]
                cam_point_vectors = result["cam_point_vectors"]

                ax.scatter(point_mirrored[0], point_mirrored[1], point_mirrored[2], c='b', marker='x')
                ax.scatter(point_wc[0], point_wc[1], point_wc[2], c='g', marker='o')

                for i in range(2):
                    curr_cam_origin = stereo_points_calibration_data["cam_pair_calib_vecs"][i]['relative']['origin']
                    curr_dst = np.linalg.norm(point_mirrored - curr_cam_origin[:3].A1)
                    curr_cpv = cam_point_vectors[i] * (curr_dst * 1.1)  # scale cpv for better visualization

                    ax.plot([curr_cam_origin[0], curr_cam_origin[0] + curr_cpv[0]], [curr_cam_origin[1], curr_cam_origin[1] + curr_cpv[1]], [curr_cam_origin[2], curr_cam_origin[2] + curr_cpv[2]], c='c', linestyle='--')

            ax.quiver(mirror_surface_point_wc[0], mirror_surface_point_wc[1], mirror_surface_point_wc[2],
                    mirror_surface_normal_wc[0], mirror_surface_normal_wc[1], mirror_surface_normal_wc[2],
                    length=20, color='m')
            
            ax.set_aspect('equal')
        
        fig.tight_layout()

        self.canvas_calib_1.draw()

        return


    def save_full_calibration(self):
        data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary = self.calculate_full_calibration()
        full_calibration_data = {
            "calibration_steps": data_calib_steps,
            "extrinsics": extrinsic_calib_summary,
            "intrinsics": intrinsic_calib_summary
        }
        
        output_dir, output_file = self.get_calibration_output_pkl_path("Full Calibration")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_file, 'wb') as handle:
            pickle.dump(full_calibration_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Full calibration results saved to {output_file}")


    def calculate_full_calibration(self, until_step=None):
        data_calib_steps = {}
        extrinsic_calib_summary = {}
        intrinsic_calib_summary = {}
        
        ## Stereo R-R
        if until_step == "Stereo R-R":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Stereo R-R")[1]
        try:
            data_calib_steps["Stereo R-R"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Stereo R-R calibration data: {e}")
            return None
        data_calib_steps["Stereo R-R"]["pairwise"] = calculate_pairwise_camera_calib_vecs(data_calib_steps["Stereo R-R"]["data"])
        extrinsic_calib_summary["ro"] = data_calib_steps["Stereo R-R"]["pairwise"][0]['absolute']
        extrinsic_calib_summary["ri"] = data_calib_steps["Stereo R-R"]["pairwise"][1]['absolute']
        intrinsic_calib_summary["ro"] = asdict(data_calib_steps["Stereo R-R"]["data"]["camera_params_0"])["intrinsic"]
        intrinsic_calib_summary["ri"] = asdict(data_calib_steps["Stereo R-R"]["data"]["camera_params_1"])["intrinsic"]

        ## Stereo R-L
        if until_step == "Stereo R-L":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Stereo R-L")[1]
        try:
            data_calib_steps["Stereo R-L"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Stereo R-L calibration data: {e}")
            return None
        data_calib_steps["Stereo R-L"]["pairwise"] = calculate_pairwise_camera_calib_vecs(data_calib_steps["Stereo R-L"]["data"])

        ## Stereo L-L
        if until_step == "Stereo L-L":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Stereo L-L")[1]
        try:
            data_calib_steps["Stereo L-L"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Stereo L-L calibration data: {e}")
            return None
        data_calib_steps["Stereo L-L"]["pairwise"] = calculate_pairwise_camera_calib_vecs(data_calib_steps["Stereo L-L"]["data"], calib_data_offset=data_calib_steps["Stereo R-L"]["data"])
        extrinsic_calib_summary["lo"] = data_calib_steps["Stereo L-L"]["pairwise"][0]['absolute']
        extrinsic_calib_summary["li"] = data_calib_steps["Stereo L-L"]["pairwise"][1]['absolute']
        intrinsic_calib_summary["lo"] = asdict(data_calib_steps["Stereo L-L"]["data"]["camera_params_0"])["intrinsic"]
        intrinsic_calib_summary["li"] = asdict(data_calib_steps["Stereo L-L"]["data"]["camera_params_1"])["intrinsic"]

        ## Mono SC
        if until_step == "Mono SC":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Mono SC")[1]
        try:
            data_calib_steps["Mono SC"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Mono SC calibration data: {e}")
            return None
        intrinsic_calib_summary["sc"] = data_calib_steps["Mono SC"]["data"]

        ## Mirror R
        if until_step == "Mirror R":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary
        
        calibration_file = self.get_calibration_output_pkl_path("Mirror R")[1]
        try:
            data_calib_steps["Mirror R"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Mirror R calibration data: {e}")
            return None
        data_calib_steps["Mirror R"]["points_wc"] = {}
        points = data_calib_steps["Mirror R"]["data"]["calibrated_points"]["CAM_BOTH"]
        for key in points.keys():
            data_calib_steps["Mirror R"]["points_wc"][key] = points[key]["point_world"]
        extrinsic_calib_summary["mr"] = data_calib_steps["Mirror R"]["points_wc"]

        ## Mirror L
        if until_step == "Mirror L":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Mirror L")[1]
        try:
            data_calib_steps["Mirror L"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Mirror L calibration data: {e}")
            return None
        data_calib_steps["Mirror L"]["points_wc"] = {}
        points = data_calib_steps["Mirror L"]["data"]["calibrated_points"]["CAM_BOTH"]
        for key in points.keys():
            data_calib_steps["Mirror L"]["points_wc"][key] = project_points_to_coordinate_frame([points[key]["point_world"]], T=transformation_matrix_from_calib_ext(data_calib_steps["Stereo R-L"]["data"]["camera_params_1"].extrinsic))[0]
        extrinsic_calib_summary["ml"] = data_calib_steps["Mirror L"]["points_wc"]

        ## Mirror SC
        if until_step == "Mirror SC":
            return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary

        calibration_file = self.get_calibration_output_pkl_path("Mirror SC")[1]
        try:
            data_calib_steps["Mirror SC"] = {"data": self.load_pkl(calibration_file)}
        except Exception as e:
            print(f"Error loading Mirror SC calibration data: {e}")
            return None
        data_calib_steps["Mirror SC"]["absolute"] = data_calib_steps["Mirror SC"]["data"]["sc_calibration_result"]["absolute"]
        extrinsic_calib_summary["sc"] = data_calib_steps["Mirror SC"]["absolute"]

        return data_calib_steps, extrinsic_calib_summary, intrinsic_calib_summary


    def plot_extrinsic_calibration_results(self, data_calib_steps):
        fig = self.canvas_calib_2.figure
        fig.clf()
        
        print(f"Plotting extrinsic calibration results for steps: {list(data_calib_steps.keys())}")
        
        if data_calib_steps is not None:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Full Extrinsic Calibration Results")
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_zlabel('Z [mm]')

            if "Stereo R-R" in data_calib_steps:
                self.plot_camera_pair_calib(data_calib_steps["Stereo R-R"]["pairwise"], ax, key="absolute")
            if "Stereo L-L" in data_calib_steps:
                self.plot_camera_pair_calib(data_calib_steps["Stereo L-L"]["pairwise"], ax, key="absolute")

            # plot points in world coordinates for mirror R and mirror L
            if "Mirror R" in data_calib_steps:
                points_wc = data_calib_steps["Mirror R"]["points_wc"]
                for point_id, point_wc in points_wc.items():
                    ax.scatter(point_wc[0], point_wc[1], point_wc[2], c='g', marker='o')
                    if point_id == "CAM_SC":
                        if "Mirror SC" in data_calib_steps:
                            sc_dir = data_calib_steps["Mirror SC"]["absolute"]["z"]
                            sc_y = data_calib_steps["Mirror SC"]["absolute"]["y"]
                            ax.quiver(point_wc[0], point_wc[1], point_wc[2], sc_dir[0], sc_dir[1], sc_dir[2], length=20, color='m')
                            ax.quiver(point_wc[0], point_wc[1], point_wc[2], sc_y[0], sc_y[1], sc_y[2], length=20/2, color='m')

            if "Mirror L" in data_calib_steps:
                points_wc = data_calib_steps["Mirror L"]["points_wc"]
                for point_id, point_wc in points_wc.items():
                    ax.scatter(point_wc[0], point_wc[1], point_wc[2], c='g', marker='o')

            fig.tight_layout()

        self.canvas_calib_2.draw()
        
        
    def plot_camera_pair_calib(self, cam_pair_calib, ax, key="relative", length=25):
        ax.quiver(cam_pair_calib[0][key]['origin'][0], cam_pair_calib[0][key]['origin'][1], cam_pair_calib[0][key]['origin'][2],
                   cam_pair_calib[0][key]['z'][0], cam_pair_calib[0][key]['z'][1], cam_pair_calib[0][key]['z'][2],
                   length=length, color='r')
        ax.quiver(cam_pair_calib[1][key]['origin'][0], cam_pair_calib[1][key]['origin'][1], cam_pair_calib[1][key]['origin'][2],
                   cam_pair_calib[1][key]['z'][0], cam_pair_calib[1][key]['z'][1], cam_pair_calib[1][key]['z'][2],
                   length=length, color='b')
        ax.quiver(cam_pair_calib[0][key]['origin'][0], cam_pair_calib[0][key]['origin'][1], cam_pair_calib[0][key]['origin'][2],
                   cam_pair_calib[0][key]['y'][0], cam_pair_calib[0][key]['y'][1], cam_pair_calib[0][key]['y'][2],
                   length=length/2, color='r')
        ax.quiver(cam_pair_calib[1][key]['origin'][0], cam_pair_calib[1][key]['origin'][1], cam_pair_calib[1][key]['origin'][2],
                   cam_pair_calib[1][key]['y'][0], cam_pair_calib[1][key]['y'][1], cam_pair_calib[1][key]['y'][2],
                   length=length/2, color='b')
        
        
    def plot_mono_calibration_results(self, step=None):
        if step is None:
            step = self.get_current_calibration_preset_name()
        
        path = self.get_calibration_output_pkl_path(step)[1]
        
        try:
            calib_data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading mono calibration data from {path}: {e}")
            self.set_calibration_step_visualization_empty()
            return

        # plot cameras and their viewing directions in 3D
        
        plt_dim = (1,3)
        
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        # Mono Distortion results
        
        def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
            h, w, _ = img.shape
            rows, cols = grid_shape
            dy, dx = h / rows, w / cols

            # draw vertical lines
            for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
                x = int(round(x))
                cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

            # draw horizontal lines
            for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
                y = int(round(y))
                cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

            return img
        
        # Generate image with grid and apply maps to undistort it according to mono calibration results
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 1)
        ax.set_title("Mono Undistortion")
        
        map1 = calib_data["map1"]
        map2 = calib_data["map2"]
        
        img_grid = np.zeros((map1.shape[0], map1.shape[1], 3), np.uint8)
        img_grid = draw_grid(img_grid, (20, 20))
        
        img_undistorted = cv2.remap(img_grid, map1, map2, interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_undistorted)
        ax.axis('off')
        ax.set_aspect('equal')

        # Scatter plots for errors
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 2)
        ax.set_title("Reproj Errors")

        errs = calib_data["errs_mono_reproj"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="blue")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 3)
        ax.set_title("Ini Reproj Errors")

        errs = calib_data["errs_mono_reproj_initial"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="red")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
        
        fig.tight_layout()

        self.canvas_calib_1.draw()
    
    
    def plot_stereo_calibration_results(self, step=None):
        if step is None:
            step = self.get_current_calibration_preset_name()

        pkl_path = self.get_calibration_output_pkl_path(step)[1]
        try:
            calib_data = np.load(pkl_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading stereo calibration data from {pkl_path}: {e}")
            self.set_calibration_step_visualization_empty()
            return

        cam_pair_calib_vecs = calculate_pairwise_camera_calib_vecs(calib_data)

        # plot cameras and their viewing directions in 3D
        
        plt_dim = (2,4)
        
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 1, projection='3d')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title("Camera Pair Extrinsics")

        self.plot_camera_pair_calib(cam_pair_calib_vecs, ax, length=10)

        ax.set_aspect('equal')
        
        # Mono Distortion results
        
        def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
            h, w, _ = img.shape
            rows, cols = grid_shape
            dy, dx = h / rows, w / cols

            # draw vertical lines
            for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
                x = int(round(x))
                cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

            # draw horizontal lines
            for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
                y = int(round(y))
                cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

            return img
        
        # Generate image with grid and apply maps to undistort it according to mono calibration results
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 2)
        ax.set_title("Mono Undistortion L")
        
        map1 = calib_data["camera_params_0"].intrinsic.map1
        map2 = calib_data["camera_params_0"].intrinsic.map2
        
        img_grid = np.zeros((map1.shape[0], map1.shape[1], 3), np.uint8)
        img_grid = draw_grid(img_grid, (20, 20))
        
        img_undistorted = cv2.remap(img_grid, map1, map2, interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_undistorted)
        ax.axis('off')
        ax.set_aspect('equal')
        
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 6)
        ax.set_title("Mono Undistortion R")

        map1 = calib_data["camera_params_1"].intrinsic.map1
        map2 = calib_data["camera_params_1"].intrinsic.map2
        
        img_grid = np.zeros((map1.shape[0], map1.shape[1], 3), np.uint8)
        img_grid = draw_grid(img_grid, (20, 20))
        
        img_undistorted = cv2.remap(img_grid, map1, map2, interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_undistorted)
        ax.axis('off')
        ax.set_aspect('equal')

        # Scatter plots for errors
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 3)
        ax.set_title("Reproj Errors L")
        
        errs = calib_data["errs_mono_reproj_l"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="blue")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 4)
        ax.set_title("Ini Reproj Errors L")
        
        errs = calib_data["errs_mono_reproj_initial_l"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="red")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 7)
        ax.set_title("Reproj Errors R")
        
        errs = calib_data["errs_mono_reproj_r"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="blue")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 8)
        ax.set_title("Ini Reproj Errors R")
        
        errs = calib_data["errs_mono_reproj_initial_r"]["all_errs_2d"]
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="red")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')

        fig.tight_layout()

        self.canvas_calib_1.draw()


    # TODO unused
    def calibrate_cameras_async(self, *args):
        if self.calibration_status == "running":
            print("Calibration already running.")
            return
        
        print("Calibrating cameras...")
        
        path_root = self.get_root_path(self.get_current_calibration_preset_name())
        step = self.get_current_calibration_preset_name()
        paths_cams = [f"{path_root}/{camera_idc}" for camera_idc in self.get_preset_for_step(step)["camera_indices"]]
        output_dir, output_file = self.get_calibration_output_pkl_path(step)

        self.calibration_status = "running"
        t_calib = threading.Thread(target=self.run_stereo_calibration, args=(paths_cams[0], paths_cams[1], None, None, output_file)).start()

        return
    
    
    def stop_all_capturing(self):
        if self.capture_state == "streaming":
            self.toggle_capture("stream")
        elif self.capture_state == "capturing":
            self.toggle_capture("capture")
    
    
    def load_capture_preset(self, *args):
        selected_capture_preset = self.get_current_capture_preset_name()

        # Stop capture and streaming before changing settings
        self.stop_all_capturing()

        for key, frame in self.cam_frames.items():
            frame["sv_active"].set('active' if key in self.get_preset_for_step(selected_capture_preset)["camera_indices"] else 'inactive')

        self.sv_vis_calibration_pattern.set('active' if self.get_preset_for_step(selected_capture_preset)["use_pattern"] is not None else 'inactive')
        self.sv_vis_img_diff_thr.set('active' if self.get_preset_for_step(selected_capture_preset)["capture_settings"]["use_threshold"] else 'inactive')
        self.sv_auto_accept.set('active' if self.get_preset_for_step(selected_capture_preset)["capture_settings"]["auto_accept"] else 'inactive')
        self.sv_adjust_contrast.set('active' if self.get_preset_for_step(selected_capture_preset)["capture_settings"]["auto_contrast"] else 'inactive')


    def load_calib_preset(self, *args):
        if self.get_current_calibration_preset()["calib_settings"]["calib_labels"] is not None:
            # Update available labels in listbox
            self.selected_label_type = self.get_current_calibration_preset()["calib_settings"]["calib_labels"]
            available_labels = self.label_types_configs[self.selected_label_type]["available_labels"]
            self.label_file_frame["sv_available_label_names"].set(available_labels) # Reset label coordinates for new label type
            for i in range(2):
                self.label_frames[i]["label_coordinates_store"] = dict(zip(available_labels, [[-1,-1]] * len(available_labels))) 
        else:
            self.selected_label_type = None
            self.label_file_frame["sv_available_label_names"].set([])  # Clear available labels if none specified in preset
            for i in range(2):
                self.label_frames[i]["label_coordinates_store"] = dict([])  # Clear label coordinates if none specified in preset

        for i in range(2):
            self.label_frames[i]["img_folder_path_store"] = None  # Reset folder path store before loading new images
            self.label_frames[i]["img_path_full_store"] = None
            self.set_empty_label_image(i)
            
            if self.get_current_calibration_preset()["camera_indices"] is not None and i < len(self.get_current_calibration_preset()["camera_indices"]):
                camera_name = self.get_current_calibration_preset()["camera_indices"][i]
                self.label_frames[i]["frame"].configure(text=f"Cam [{camera_name}] Labels")
                # Re-add frame to grid in case it was removed before
                self.label_frames[i]["frame"].grid()
            else:
                self.label_frames[i]["label_coordinates_store"] = {}  # Clear label coordinates if no labels available
                self.label_frames[i]["frame"].configure(text=f"Cam [None] Labels")
                # Hide label frames for cameras not defined in preset and reset their image and label coordinates stores
                self.label_frames[i]["frame"].grid_remove()

        self.load_label_path()  # Reload image to reset label coordinates visualization
        
        self.visualize_calibration_results()  # Visualize step results if already executed


    def get_img_dirs(self, step=None):
        if step is None:
            step = self.get_current_calibration_preset_name()
        if self.get_current_calibration_preset()["camera_indices"] is None:
            print(f"No camera indices defined for calibration step {step}. Cannot get image directories.")
            return None
        return [
            self.get_root_path(step, frame_idx)
            for frame_idx in range(len(self.get_current_calibration_preset()["camera_indices"]))
        ]
        
        
    def get_files_in_dirs(self, dirs):
        files_per_dir = []
        for dir in dirs:
            if os.path.exists(dir) and os.path.isdir(dir):
                files = [f for f in os.listdir(dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files_per_dir.append(files)
            else:
                print(f"Directory path {dir} does not exist or is not a directory.")
                files_per_dir.append([])
                
        return files_per_dir


    def get_common_files(self, list_of_file_lists):
        return sorted(list(set.intersection(*map(set, list_of_file_lists))))


    def load_label_path(self):
        # Enumerate available images in path
        img_dirs = self.get_img_dirs()
        if img_dirs is None:
            print("No image directories found for the current calibration preset.")
            self.label_file_frame["sv_available_label_imgs"].set([])  # Clear available images for cameras not defined in preset
            self.label_file_frame["available_label_imgs_store"] = []
            self.label_file_frame["selected_label_img_id"] = -1
            for i in range(2):
                self.set_empty_label_image(i)
            return
        
        img_files_per_folder = self.get_files_in_dirs(img_dirs)
        joined_img_files = self.get_common_files(img_files_per_folder)
        
        self.label_file_frame["sv_available_label_imgs"].set(joined_img_files)  # Clear available images for cameras not defined in preset
        self.label_file_frame["available_label_imgs_store"] = joined_img_files
        
        # select first image by default
        self.label_file_frame["selected_label_img_id"] = 0
        self.label_file_frame["listbox_available_label_imgs"].selection_clear(0, "end")
        self.label_file_frame["listbox_available_label_imgs"].selection_set(self.label_file_frame["selected_label_img_id"])

        for frame_idx in range(len(self.label_frames)):
            if frame_idx < len(self.get_current_calibration_preset()["camera_indices"]):
                self.label_frames[frame_idx]["img_folder_path_store"] = img_dirs[frame_idx]  # Store folder path to load images later when clicking on listbox
                self.load_label_image(frame_idx)


    def onclick_listbox_available_label_imgs(self, event):
        for i, _ in enumerate(self.get_current_calibration_preset()["camera_indices"]):
            self.load_label_image(i)


    def load_label_image(self, frame_idx):
        idxs = self.label_file_frame["listbox_available_label_imgs"].curselection()
        if len(idxs) == 1:
            img_idx = idxs[0]
            self.label_file_frame["selected_label_img_id"] = img_idx  # Store selected image index to load it again when switching label types
        else:
            img_idx = self.label_file_frame["selected_label_img_id"]  # Load previously selected image if no new selection, otherwise load first image by default
            
        self.label_frames[frame_idx]["img_full_path_store"] = None  # Reset full image path store before loading new image
        
        if img_idx >= len(self.label_file_frame["available_label_imgs_store"]):
            print(f"Selected image index {img_idx} is out of bounds for available images.")
            self.set_empty_label_image(frame_idx)
            return

        img_path = os.path.join(self.label_frames[frame_idx]["img_folder_path_store"], self.label_file_frame["available_label_imgs_store"][img_idx])
        if os.path.exists(img_path):
            self.label_frames[frame_idx]["img_full_path_store"] = img_path

            imgdata = cv2.imread(img_path)
            #imgdata = self.resize_and_pad_image(imgdata, 240, 240)
            
            # resize canvas to image size if needed
            img_h, img_w = imgdata.shape[:2]
            if self.label_frames[frame_idx]["w_canvas"].winfo_width() != img_w or self.label_frames[frame_idx]["w_canvas"].winfo_height() != img_h:
                self.label_frames[frame_idx]["w_canvas"].config(width=img_w, height=img_h)

            self.label_frames[frame_idx]["img_store_raw"] = imgdata.copy()  # Store raw image

            imgdata, corners = self.detect_and_visualize_corners(imgdata, self.get_current_calibration_preset()["use_pattern"])
            self.label_frames[frame_idx]["pattern_corners_store"] = corners
            imgdata = cv2.imencode(".png", imgdata)[1].tobytes()

            self.label_frames[frame_idx]["img_store"] = PhotoImage(data=imgdata)  # Store to self to avoid issues with garbage collector
            for key, _ in self.label_frames[frame_idx]["label_coordinates_store"].items():
                self.label_frames[frame_idx]["label_coordinates_store"][key] = [-1,-1]  # Reset led coordinates when loading new image
                
            # select first listbox entry id by default
            self.label_file_frame["listbox_available_label_names"].selection_clear(0, "end")
            self.label_file_frame["listbox_available_label_names"].selection_set(0)
            
            label_path = img_path.rsplit(".", 1)[0] + self.calibration_settings["calib_labels_pkl_suffix"]
            try:
                self.label_frames[frame_idx]["label_coordinates_store"] = self.load_pkl(label_path)["label_coordinates"]
                print(f"Frame {frame_idx+1}: Loaded existing label coordinates from {label_path}")
            except Exception as e:
                print(f"Frame {frame_idx+1}: Error loading label coordinates from {label_path}: {e}")

            self.update_label_coordinates_display(frame_idx)
            print(f"Frame {frame_idx+1}: Loaded image {img_path}")
        else:
            self.set_empty_label_image(frame_idx)
            print(f"Frame {frame_idx+1}: Image path {img_path} does not exist.")
            
    
    def set_empty_label_image(self, frame_idx):
        empty_img = np.zeros((240, 240, 3), dtype=np.uint8)
        self.label_frames[frame_idx]["img_store_raw"] = empty_img.copy()  # Store raw image
        imgdata = cv2.imencode(".png", empty_img)[1].tobytes()
        self.label_frames[frame_idx]["img_store"] = PhotoImage(data=imgdata)  # Store to self to avoid issues with garbage collector

        self.reset_all_label_coordinates(frame_idx)


    def reset_label_coordinates(self, frame_idx):
        label_ids = self.label_file_frame["listbox_available_label_names"].curselection()
        if len(label_ids) == 1:
            label_id = label_ids[0]
            label_name = self.label_types_configs[self.selected_label_type]["available_labels"][label_id]
            self.label_frames[frame_idx]["label_coordinates_store"][label_name] = [-1,-1]  # Reset label coordinates

            self.update_label_coordinates_display(frame_idx)


    def reset_all_label_coordinates(self, frame_idx):
        for key in self.label_frames[frame_idx]["label_coordinates_store"].keys():
            self.label_frames[frame_idx]["label_coordinates_store"][key] = [-1,-1]  # Reset label coordinates

        self.update_label_coordinates_display(frame_idx)
    
    
    def get_file_path_with_suffix(self, file_path, suffix):
        return str(file_path).rsplit(".", 1)[0] + suffix


    def save_label_coordinates_for_frame(self, frame_idx):
        img_path = self.label_frames[frame_idx]["img_full_path_store"]
        if img_path:
            save_path = self.get_file_path_with_suffix(img_path, self.calibration_settings["calib_labels_pkl_suffix"])

            save_dict = {
                "label_coordinates": self.label_frames[frame_idx]["label_coordinates_store"],
                "pattern_corners": self.label_frames[frame_idx]["pattern_corners_store"],
            }

            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved label coordinates to {save_path}")
        else:
            print(f"No image selected.")


    def save_label_coordinates(self):
        for frame_idx in range(len(self.label_frames)):
            self.save_label_coordinates_for_frame(frame_idx)
    

    def update_label_coordinates_display(self, frame_idx):
        self.label_frames[frame_idx]["w_canvas"].create_image(0, 0, image=self.label_frames[frame_idx]["img_store"], anchor='nw')

        # Draw Edge points to visualize canvas borders (for debugging and to avoid confusion about coordinate system)
        self.label_frames[frame_idx]["w_canvas"].create_rectangle(0, 0, 1, 1, fill='green', width=0)
        self.label_frames[frame_idx]["w_canvas"].create_rectangle(239, 0, 240, 1, fill='green', width=0)
        self.label_frames[frame_idx]["w_canvas"].create_rectangle(0, 239, 1, 240, fill='green', width=0)
        self.label_frames[frame_idx]["w_canvas"].create_rectangle(239, 239, 240, 240, fill='green', width=0)

        for key, val in self.label_frames[frame_idx]['label_coordinates_store'].items():
            if val[0] >= 0 and val[1] >= 0:
                # Draw circle at clicked position
                self.label_frames[frame_idx]["w_canvas"].create_oval(val[0]-2, val[1]-2, val[0]+1, val[1]+1, outline='red', width=2)  # See https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/create_oval.html, really counterintuitive
                self.label_frames[frame_idx]["w_canvas"].create_text(val[0], val[1]-10, text=f"{key}", fill='red', font=('Arial', 10, 'bold'))
        
        positions_str = "\n".join([f"{key}: ({val[0]}, {val[1]})" for key, val in self.label_frames[frame_idx]['label_coordinates_store'].items()])
        positions_str += "\nPattern Detected: " + ("True" if self.label_frames[frame_idx]["pattern_corners_store"] is not None else "False")
        self.label_frames[frame_idx]["sv_label_coordinates"].set(positions_str)

        # Don't run this when image is empty
        if self.label_file_frame["selected_label_img_id"] >= 0:
            step = self.get_current_calibration_preset_name()
            if step in ["Mirror R", "Mirror L"]:
                stereo_points_calibration_data = self.calib_triangulate_points_for_step_and_file(live=True, step=step, input_filename=self.label_file_frame["available_label_imgs_store"][self.label_file_frame["selected_label_img_id"]])
                if stereo_points_calibration_data is not None:
                    self.plot_stereo_points_calibration_results(stereo_points_calibration_data=stereo_points_calibration_data)
            elif step == "Mirror SC":
                _ = self.calibrate_scene_camera_extrinsics(live=True, step=step, input_filename=self.label_file_frame["available_label_imgs_store"][self.label_file_frame["selected_label_img_id"]])
                # TODO separate plot function


    def onclick_set_label_coordinates(self, event, frame_idx):
        if frame_idx >= len(self.get_current_calibration_preset()["camera_indices"]):
            print(f"Frame index {frame_idx} is out of bounds for current calibration preset.")
            return
        
        label_ids = self.label_file_frame["listbox_available_label_names"].curselection()
        if len(label_ids) != 1:
            return  # Ensure only one label is selected

        label_id = label_ids[0]
        label_name = self.label_types_configs[self.selected_label_type]["available_labels"][label_id]
        x, y = event.widget.canvasx(event.x), event.widget.canvasy(event.y)
        self.label_frames[frame_idx]["label_coordinates_store"][label_name] = [x, y]

        # Update display
        self.update_label_coordinates_display(frame_idx)

        # cycle dropdown to next led id
        next_label_id = (label_id + 1) % len(self.label_types_configs[self.selected_label_type]["available_labels"])
        self.label_file_frame["listbox_available_label_names"].selection_clear(0, "end")
        self.label_file_frame["listbox_available_label_names"].selection_set(next_label_id)


    ########## CAPTURE FUNCTIONS ##########

    def onclick_triggered_snapshot(self):
        if self.capture_state != "stream":
            print("Stream not running, cannot capture.")
            return

        self.capture_snapshot_state = "triggered_snapshot"
        self.sv_capture_status.set("Waiting for trigger...")


    def accept_triggered_snapshot(self, *args):
        if self.capture_state != "stream":
            print("Stream not running, cannot capture.")
            return

        if self.capture_snapshot_state == "awaiting_accept":
            path = self.get_root_path(self.get_current_capture_preset_name())
            self.save_snapshot(
                folder_path=Path(path),
                capture_buffer=self.capture_buffer,
                append=True
            )
            
            self.capture_snapshot_state = "idle"
            self.sv_capture_status.set("Snapshot saved.")


    def stop_triggered_snapshot(self, *args):
        self.capture_snapshot_state = "idle"
        self.sv_capture_status.set("Idle")


    def create_capture_folders(self):
        #create folder if it does not exist
        if not os.path.exists(self.curr_capture_folder_path):
            os.makedirs(self.curr_capture_folder_path)

            subfolder_paths = [os.path.join(self.curr_capture_folder_path, self.curr_capture_cam_labels[cam_i]) for cam_i in range(len(self.curr_capture_cam_labels))]
            for subfolder_path in subfolder_paths:
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

            print(f"Created folder {self.curr_capture_folder_path} for capture.")

            return self.curr_capture_folder_path
        else:
            print(f"Folder {self.curr_capture_folder_path} already exists. Not creating new folder.")
            return None


    def save_frame(self, frame, frame_id, timestamp, cam_label):
        subfolder_path = os.path.join(self.curr_capture_folder_path, cam_label)
        fname = os.path.join(subfolder_path, f"frame_{frame_id}_timestamp_{timestamp:.3f}.png")
        print(f"Saving frame to {fname}")
        cv2.imwrite(fname, frame)
        
    
    def save_logs(self, logs):
        log_file_path = os.path.join(self.curr_capture_folder_path, "logs.json")
        with open(log_file_path, "w") as log_file:
            json.dump(logs, log_file)
        print(f"Saved logs to {log_file_path}")
    
    
    async def websocket_sync_handler(self, camera_handler, url):
        ws_messages = []
        
        try:
            async with websockets.connect(url) as websocket:
                await websocket.send("subscriber")
                print("WebSocket connected as subscriber.")

                while not camera_handler.ev_request_terminate.is_set():
                    try:
                        message = await websocket.recv()
                        system_ts = time.time_ns() / 1e9
                        log_entry = {
                            "system_unix_ts": system_ts,
                            "ws_message": message
                        }

                        parsed = json.loads(message)

                        if "timestamp" in parsed:
                            log_entry["message_unix_ts"] = parsed["timestamp"]
                            print(f"WS Message: {json.dumps(log_entry)}")
                        
                        eventType = parsed.get("eventType")
                        
                        if eventType == "TaskStart":
                            print("Received TaskStart event.")
                            
                            # Start camera recording
                            camera_handler.ev_start_capture.set()
                            
                        if eventType == "TaskEnd":
                            print("Received TaskEnd event.")
                            
                            # Stop camera recording
                            camera_handler.ev_websocket_request_terminate.set()
                            break

                        ws_messages.append(log_entry)

                    except Exception as e:
                        print(f"WS receive error: {e}")
                        camera_handler.ev_websocket_request_terminate.set()
                        break
                    
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            print("Terminating")
            camera_handler.ev_websocket_request_terminate.set()

        camera_handler.ws_message_q.put(ws_messages)

        return


    def websocket_sync_handler_thread(self, camera_handler, url="wss://hctlsrvc.edu.sot.tum.de/eventdetectionwsmarker2/"):
        asyncio.run(self.websocket_sync_handler(camera_handler, url))
        return


    def toggle_capture(self, mode, *args):
        print("Capture triggered")

        if self.capture_state == "inactive":
            self.curr_capture_cam_urls = []
            self.curr_capture_cam_labels = []
            
            for key, cam_frame in self.cam_frames.items():
                if cam_frame["sv_active"].get() == 'active':
                    if cam_frame["sv_type"].get() == 'USB':
                        self.curr_capture_cam_urls.append(int(cam_frame["sv_ip"].get()))
                    else:
                        self.curr_capture_cam_urls.append(cam_frame["sv_ip"].get())

                    self.curr_capture_cam_labels.append(key)
                    
                    cam_frame["img_store"] = None
                    cam_frame["img_store_raw"] = None
                    cam_frame["metadata_store"] = None

            self.camera_handler = CameraHandler(
                urls=self.curr_capture_cam_urls,
                adapter_ip=self.sv_adapter_ip.get()
            )
            
            if mode == "capture":
                stream_enabled = self.sv_capture_with_stream.get() == 'active'
                recording_enabled = True
                self.sv_recording.set("Stop Capture")
                print("Capture activated")
                
                # Add datetime string to folder path
                now = datetime.now()
                dt_string = now.strftime("__%Y_%m_%d_%H_%M_%S")
                self.curr_capture_folder_path = self.sv_experiment_name.get() + dt_string

                self.create_capture_folders()
            else:  # mode == "stream"
                stream_enabled = True
                recording_enabled = False
                self.sv_btn_start_stream.set("Stop Stream")
                print("Stream activated")
                
            if self.sv_sync_recording_with_stimulus.get() == 'active':
                print("Syncing capture with stimulus...")
                self.capture_sync_mode = True
                
                gaze_socket_url = self.sv_gaze_socket_url.get()

                self.t_websocket_sync_handler = threading.Thread(target=self.websocket_sync_handler_thread, args=(self.camera_handler, gaze_socket_url))
            else:
                self.capture_sync_mode = False

            use_external_trigger = self.sv_use_external_trigger.get() == 'active'

            self.t_rec = threading.Thread(target=record_frames_multithreaded, args=(self.camera_handler, None, stream_enabled, recording_enabled, self.capture_sync_mode, use_external_trigger))
            self.t_rec.start()
            
            if self.capture_sync_mode:
                self.t_websocket_sync_handler.start()

            print("Camera threads started")

            self.capture_state = mode

            self.vtimer = VTimer()

        elif mode == self.capture_state:
            print(f"{mode} deactivated")
            self.camera_handler.ev_request_terminate.set()

            if mode == "stream" and self.capture_state == "stream":
                # Relabel button
                self.sv_btn_start_stream.set("Start Stream")
            
            elif mode == "capture" and self.capture_state == "capture":
                # Relabel button
                self.sv_recording.set("Start Capture")
                
                # Empty queues and gather remaining frames for saving
                for stream_idx, cam_label in enumerate(self.curr_capture_cam_labels):
                    print(f"Gathering remaining frames for camera {cam_label} from stream index {stream_idx} queue...")
                    while True:
                        try:
                            frame_q_data = self.camera_handler.recording_qs[stream_idx].get_nowait()
                            self.save_frame(frame_q_data["frame"], frame_q_data["metadata"][-1]["frame_idx"], frame_q_data["metadata"][-1]["x_timestamp_from_start"], cam_label)
                        except queue.Empty:
                            print(f"No more frames in queue for camera {cam_label}.")
                            break

                logs={
                    "url_cams": self.curr_capture_cam_urls,
                    "url_labels": self.curr_capture_cam_labels,
                    "folder_path": self.curr_capture_folder_path
                }
                
                if self.capture_sync_mode:
                    try:
                        ws_messages = self.camera_handler.ws_message_q.get_nowait()
                    except queue.Empty:
                        ws_messages = []

                    logs["websocket_messages"] = ws_messages
                    logs["recording_begin_trigger"] = "Websocket TaskStart"
                    
                    print("Waiting for WebSocket sync handler to finish...")
                    self.t_websocket_sync_handler.join()
                    print("WebSocket sync handler joined")
                else:
                    logs["recording_begin_trigger"] = "Manual Capture"
                
                self.save_logs(logs)
            
            self.t_rec.join()
            print("Camera threads joined")
            
            for cam_frame_label in self.curr_capture_cam_labels:
                self.cam_frames[cam_frame_label]["w_canvas"].create_rectangle(1, 1, 239, 239, outline='red', width=2)
                
            self.capture_state = "inactive"
            self.camera_handler = None
        
        else:
            print("Action not possible, CameraHandler is already running. Please stop current capture/stream before starting a new one.")
            return


    def save_snapshot(self, folder_path, capture_buffer, append=False):
        # TODO add frame ID and/or x_timestamp to file name?
        
        # add datetime string to file name path
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        
        #create folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        elif not append:
            print(f"Folder {folder_path} already exists. Not saving frames.")
            return

        for cam_label, buf in capture_buffer.items():
            subfolder_path = os.path.join(folder_path, cam_label)            
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            fname = os.path.join(subfolder_path, f"snapshot__{dt_string}.png")
            print(fname)
            cv2.imwrite(fname, buf["frame"])
            
            print(f"Saved snapshot for camera {cam_label} to {fname}")


    def resize_and_pad_image(self, img, target_width, target_height):
        # get aspect ratio for resize
        h, w = img.shape[:2]
        aspect_ratio = w / h
        if aspect_ratio >= 1:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        img = cv2.resize(img, (new_width, new_height))
        return img


    def detect_and_visualize_corners(self, frame, pattern_size_key=None):
        if pattern_size_key is None:
            pattern_size_key = "small"  # Default pattern size key if none specified in preset
        
        corners = detect_corners(frame, self.calibration_settings["patterns"][pattern_size_key]["corners"])  # cols, rows
        corners_detected = corners is not None

        if corners_detected:
            #print(f"Detected {len(corners)} corners.")
            imgdata = frame.copy()
            if len(imgdata.shape) == 2:
                imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
            for idx, corner in enumerate(corners):
                color = (0, 255, 0) if idx < 2 else (255, 0, 0)
                imgdata = cv2.circle(imgdata, (int(corner[0][0]), int(corner[0][1])), 3, color, -1)
        else:
            #print("No corners detected.")
            imgdata = frame.copy()

        return imgdata, corners

    
    def update_stream_ui(self):
        if self.capture_state != "inactive":
            capture_info = {
                cam_label: {
                "pattern_detected": False,
                "img_diff": -1,
                "frame": None
                } for cam_label in self.curr_capture_cam_labels
            }

            try:
                img_diff_threshold = float(self.sv_auto_accept_img_diff_threshold.get())
            except ValueError:
                img_diff_threshold = 0

            for stream_idx, cam_label in enumerate(self.curr_capture_cam_labels):
                if self.capture_state != "inactive" and not self.camera_handler.stream_qs[stream_idx].empty():  # TODO thread safety
                    stream_entry = self.camera_handler.stream_qs[stream_idx].get()
                    self.vtimer.time_log()
                    
                    self.cam_frames[cam_label]["metadata_store"] = {
                        "x_timestamp": stream_entry["metadata"][-1]["x_timestamp"],
                        "x_timestamp_from_start": stream_entry["metadata"][-1]["x_timestamp_from_start"],
                        "frame_idx": stream_entry["metadata"][-1]["frame_idx"],
                        "fps": stream_entry['fps']
                    }

                    capture_info[cam_label]["frame"] = stream_entry["frame"]

                    if capture_info[cam_label]["frame"] is not None:
                        imgdata = capture_info[cam_label]["frame"].copy()

                        if self.cam_frames[cam_label]["img_store_raw"] is not None:
                            capture_info[cam_label]["img_diff"] = np.mean(cv2.absdiff(self.cam_frames[cam_label]["img_store_raw"], imgdata))

                        #print(f"Stream {stream_idx} Frame received at {self.vtimer.get()['times'][-1]:.3f} s, Stream time: {stream_entry['time_steps']['times'][-1]:.3f} s, Difference: {self.vtimer.get()['times'][-1] - stream_entry['time_steps']['times'][-1]:.3f} s, Metadata timestamp: {stream_entry['metadata'][-1]['x_timestamp']} s, Difference with offset: {self.vtimer.get()['times'][-1] - (stream_entry['metadata'][-1]['x_timestamp'] + self.offset_time):.3f} s")
                        
                        self.cam_frames[cam_label]["img_store_raw"] = imgdata.copy()
                        
                        if self.sv_adjust_contrast.get() == 'active':
                            imgdata, _, _ = automatic_brightness_and_contrast(imgdata, clip_hist_percent=0.5)

                        if self.sv_vis_calibration_pattern.get() == 'active':
                            imgdata, corners = self.detect_and_visualize_corners(imgdata, self.get_preset_for_step(self.get_current_capture_preset_name())["use_pattern"])

                            if corners is not None:
                                capture_info[cam_label]["pattern_detected"] = True
                                
                        elif self.sv_vis_pupil_detection.get() == 'active':
                            try:
                                res = quick_ellseg(
                                    imgdata,
                                    checkpoint=DEFAULT_WEIGHTS,
                                    include_image=True,
                                    use_auto_brightness=True,
                                    alpha=1.0, beta=0.0,
                                    debug_plot=False,
                                )

                                imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
                                imgdata = cv2.ellipse(imgdata, (int(res["center"][0]), int(res["center"][1])), (int(res["axes"][0]/2), int(res["axes"][1]/2)), res["angle_deg"], 0, 360, (0, 255, 0), 1)
                            except Exception as e:
                                print(f"Pupil detection error: {e}")
                                imgdata = capture_info[cam_label]["frames"].copy()

                        if (not self.capture_snapshot_state == "awaiting_accept") and (self.last_timestamp_auto_accept == -1 or (time.time() - self.last_timestamp_auto_accept) > (self.cooldown_auto_accept_rendering / 2)):
                            self.cam_frames[cam_label]["sv_info"].set(f"FPS: {self.cam_frames[cam_label]['metadata_store']['fps']:.1f}\nFrame: {self.cam_frames[cam_label]['metadata_store']['frame_idx']}\nX_t: {self.cam_frames[cam_label]['metadata_store']['x_timestamp']:.3f} s\nX_t elps: {self.cam_frames[cam_label]['metadata_store']['x_timestamp_from_start']:.3f} s\nDiff: {capture_info[cam_label]['img_diff']:.3f}")

                            imgdata = self.resize_and_pad_image(imgdata, 240, 240)
                            imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
                            self.cam_frames[cam_label]["img_store"] = PhotoImage(data=imgdata)  # Store to self to avoid issues with garbage collector
                            
                            self.cam_frames[cam_label]["w_canvas"].delete("all")
                            self.cam_frames[cam_label]["w_canvas"].create_image(0, 0, image=self.cam_frames[cam_label]["img_store"], anchor='nw')
                            if self.sv_vis_img_diff_thr.get() == "active" and capture_info[cam_label]["img_diff"] != -1 and capture_info[cam_label]["img_diff"] <= img_diff_threshold:
                                self.cam_frames[cam_label]["w_canvas"].create_rectangle(1, 1, 239, 239, outline='blue', width=2)
                        else:
                            self.cam_frames[cam_label]["w_canvas"].create_rectangle(1, 1, 239, 239, outline='yellow', width=2)
                    else:
                        self.cam_frames[cam_label]["sv_info"].set(f"FPS: {self.cam_frames[cam_label]['metadata_store']['fps']:.1f}\nFrame: {self.cam_frames[cam_label]['metadata_store']['frame_idx']}\nX_t: {self.cam_frames[cam_label]['metadata_store']['x_timestamp']:.3f} s\nX_t elps: {self.cam_frames[cam_label]['metadata_store']['x_timestamp_from_start']:.3f} s\nNo frame data")

            
            if self.last_timestamp_auto_accept != -1 and (time.time() - self.last_timestamp_auto_accept) >= self.cooldown_auto_accept:
                self.last_timestamp_auto_accept = -1
                self.capture_snapshot_state = "triggered_snapshot"
                self.sv_capture_status.set("Waiting for pattern...")

            if self.capture_snapshot_state == "triggered_snapshot":
                snapshot_trigger_pattern = self.sv_vis_calibration_pattern.get() == 'active'
                snapshot_trigger_img_threshold = self.sv_vis_img_diff_thr.get() == 'active'
                if not snapshot_trigger_pattern or all(buf["pattern_detected"] for buf in capture_info.values()):
                    if not snapshot_trigger_img_threshold or (all(buf["img_diff"] != -1 for buf in capture_info.values()) and all(buf["img_diff"] <= img_diff_threshold for buf in capture_info.values())):

                        self.capture_buffer = capture_info.copy()

                        if self.sv_auto_accept.get() == 'active':
                            self.capture_snapshot_state = "awaiting_accept"
                            self.last_timestamp_auto_accept = time.time()
                            self.accept_triggered_snapshot()
                            self.sv_capture_status.set("Snapshot saved.")
                        else:
                            self.capture_snapshot_state = "awaiting_accept"
                            self.sv_capture_status.set("Pattern detected! Accept or Reject.")
                            
    
    def update_gaze_ui(self):
        # Fetch Queue and update status text
        if not self.gaze_thread_handler["status_q"].empty():
            # Get last item from queue and clear the rest
            while not self.gaze_thread_handler["status_q"].empty():
                status_text = self.gaze_thread_handler["status_q"].get()
            self.label_gaze_status.config(text=status_text)
                

    def update(self):
        self.update_stream_ui()
        self.update_gaze_ui()
        
        if self.capture_sync_mode:
            if self.camera_handler is not None and self.camera_handler.ev_websocket_request_terminate.is_set():
                print("WebSocket requested termination, stopping capture/stream.")
                self.toggle_capture(self.capture_state)
        
        if self.capture_state == "capture":
            for stream_idx, _ in enumerate(self.curr_capture_cam_labels):
                try:
                    frame_q_data = self.camera_handler.recording_qs[stream_idx].get_nowait()
                    self.save_frame(frame_q_data["frame"], frame_q_data["metadata"][-1]["frame_idx"], frame_q_data["metadata"][-1]["x_timestamp_from_start"], self.curr_capture_cam_labels[stream_idx])
                except queue.Empty:
                    break

    def on_closing(self):
        self.root.destroy()
        self.app_running = False


    def run(self):
        #self.root.mainloop()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        while self.app_running:
            self.root.update_idletasks()
            self.update()
            self.root.update()


if __name__ == "__main__":
    app = App()
    app.run()
