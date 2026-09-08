import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import queue
import threading
import time
from tkinter import N, W, S, E, Canvas, PhotoImage, StringVar, ttk
from typing import Optional, Union

import cv2
import numpy as np
import websockets

from app.detect_pupil_ellseg import DEFAULT_WEIGHTS, automatic_brightness_and_contrast, quick_ellseg
from app.data_structures import CAPTURE_PRESET_CONFIGS, RECORDING_SUBFOLDER_FORMAT, CameraIndex, CameraType, UIPreset, UIPresetConfig
from app.camera import CameraHandler, CaptureTransferBufferFrame, record_frames_multithreaded
from app.utils import CaptureFolderManager, detect_and_visualize_corners, resize_and_pad_image


SV_CHECKBOX_ACTIVE = "active"
SV_CHECKBOX_INACTIVE = "inactive"

PRESET_ORDER: list[UIPreset] = [UIPreset.CAPTURE, UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.STEREO_RL, UIPreset.MONO_SC, UIPreset.MIRROR_R, UIPreset.MIRROR_L, UIPreset.MIRROR_SC]
DEFAULT_PRESET: UIPreset = PRESET_ORDER[0]
COOLDOWN_AUTO_ACCEPT: float = 1.0
COOLDOWN_AUTO_ACCEPT_RENDERING: float = 0.5  # seconds to wait after rendering accepted snapshot to allow user to see the pattern in the stream

DEFAULT_EXPERIMENT_NAME: str = "recordings/calibration"
DEFAULT_ADAPTER_IP: str = "" #"192.168.178.57"  # Ethernet adapter IP address
DEFAULT_AUTO_ACCEPT_IMG_DIFF_THRESHOLD: float = 4.0
LOG_FILE_NAME: str = "capture_logs.json"

@dataclass
class UIFrameCameraSettings:
    url: str
    type: CameraType
    grid_placement: list[int]
    eye_cam: bool

# Helper constants
_DEFAULT_CAMERA_URLS = [1, 2, 3, 4, 0]  # IP camera URLs and USB camera index (Find camera index via ffmpeg -f avfoundation -list_devices true -i "")
_DEFAULT_CAMERA_TYPES = [CameraType.USB, CameraType.USB, CameraType.USB, CameraType.USB, CameraType.USB]
DEFAULT_CAMERA_SETTINGS: dict[CameraIndex, UIFrameCameraSettings] = {
    CameraIndex.RO: UIFrameCameraSettings(
        url=_DEFAULT_CAMERA_URLS[0],
        type=_DEFAULT_CAMERA_TYPES[0],
        grid_placement=[0, 0],
        eye_cam=True
    ),
    CameraIndex.RI: UIFrameCameraSettings(
        url=_DEFAULT_CAMERA_URLS[1],
        type=_DEFAULT_CAMERA_TYPES[1],
        grid_placement=[1, 0],
        eye_cam=True
    ),
    CameraIndex.LO: UIFrameCameraSettings(
        url=_DEFAULT_CAMERA_URLS[3],
        type=_DEFAULT_CAMERA_TYPES[3],
        grid_placement=[0, 1],
        eye_cam=True
    ),
    CameraIndex.LI: UIFrameCameraSettings(
        url=_DEFAULT_CAMERA_URLS[2],
        type=_DEFAULT_CAMERA_TYPES[2],
        grid_placement=[1, 1],
        eye_cam=True
    ),
    CameraIndex.SC: UIFrameCameraSettings(
        url=_DEFAULT_CAMERA_URLS[4],
        type=_DEFAULT_CAMERA_TYPES[4],
        grid_placement=[0, 2],
        eye_cam=False
    )
}


class UICaptureState(Enum):
    INACTIVE = "inactive"
    STREAM = "stream"
    CAPTURE = "capture"


class UISnapshotState(Enum):
    IDLE = "idle"
    TRIGGERED_SNAPSHOT = "triggered_snapshot"
    AWAITING_ACCEPT = "awaiting_accept"
    COOLDOWN = "cooldown"


@dataclass
class CaptureBuffers:
    """A class to hold the capture buffers for each camera."""
    
    @dataclass
    class CaptureBufferFrame(CaptureTransferBufferFrame):
        frame: Optional[np.ndarray] = None
        timestamp: Optional[float] = None
        pattern_detected: Optional[bool] = None
        img_diff: Optional[float] = None
    
    type CaptureBuffer = dict[CameraIndex, CaptureBufferFrame]
    
    old: Optional[CaptureBuffer] = None
    current: Optional[CaptureBuffer] = None


@dataclass
class UIAppCaptureState:
    capture_state: UICaptureState = UICaptureState.INACTIVE
    snapshot_state: UISnapshotState = UISnapshotState.IDLE
    curr_capture_cam_idxs: list[CameraIndex] = field(default_factory=list)
    curr_capture_cam_urls: dict[CameraIndex, str|int] = field(default_factory=dict)
    camera_handler: CameraHandler | None = None
    capture_sync_mode: bool = False
    last_timestamp_auto_accept: float | None = None
    folder_manager: CaptureFolderManager = field(default_factory=CaptureFolderManager)
    current_preset: UIPreset = DEFAULT_PRESET
    capture_buffers: CaptureBuffers = field(default_factory=CaptureBuffers)


class UITabCapture:
    state: UIAppCaptureState
    
    def __init__(self, parent_frame: ttk.Frame):
        self.state = UIAppCaptureState()
        
        # Layout
        
        self.ui_cam_mainframe = ttk.LabelFrame(parent_frame, text="Cameras", padding=(5,5))
        self.ui_cam_mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        class UICameraFrame:
            def __init__(self, parent_frame: ttk.Frame, label: CameraIndex, ui_frame_camera_settings: UIFrameCameraSettings):
                self.frame = ttk.LabelFrame(parent_frame, text=f"Cam [{label.value}]", padding=(5,5))
                self.frame.grid(column=ui_frame_camera_settings.grid_placement[0], row=ui_frame_camera_settings.grid_placement[1]+1, sticky=(N, W))

                # https://stackoverflow.com/questions/4310489/how-do-i-remove-the-light-grey-border-around-my-canvas-widget
                self.w_canvas = Canvas(self.frame, width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
                self.w_canvas.grid(column=0, row=1, sticky=(N, W))
                self.w_canvas.xview_moveto(5)  # ref highlightthickness
                self.w_canvas.yview_moveto(5)  # ref highlightthickness

                self.frame_info = ttk.Frame(self.frame)
                self.frame_info.grid(column=1, row=1, sticky=(N, W))

                ttk.Label(self.frame_info, text=f"Cam IP").grid(column=1, row=2, sticky=(N,W))
                self.sv_ip = StringVar()
                self.sv_ip.set(ui_frame_camera_settings.url)
                ttk.Entry(self.frame_info, textvariable=self.sv_ip, width=15).grid(column=2, row=2, sticky=(N,W))

                self.sv_active = StringVar()
                self.sv_active.set(SV_CHECKBOX_ACTIVE)
                ttk.Checkbutton(self.frame_info, text="Active", variable=self.sv_active, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=1, row=3, sticky=W)

                # Dropdown for camera type (IP/USB)
                self.sv_type = StringVar()
                ttk.OptionMenu(self.frame_info, self.sv_type, ui_frame_camera_settings.type.value, *[camera_type.value for camera_type in CameraType]).grid(column=1, row=4, sticky=W)

                self.sv_info = StringVar()
                ttk.Label(self.frame_info, text=f"Cam Info:").grid(column=1, row=5, sticky=(N,W))
                ttk.Label(self.frame_info, textvariable=self.sv_info, font=("Courier", 10)).grid(column=2, row=5, sticky=(N,W))
        
        
        self.ui_camera_frames: dict[CameraIndex, UICameraFrame] = {}
        for label, cam_settings in DEFAULT_CAMERA_SETTINGS.items():
            self.ui_camera_frames[label] = UICameraFrame(self.ui_cam_mainframe, label, cam_settings)

        # -- Controls

        self.frame_controls = ttk.LabelFrame(parent_frame, text="Controls", padding=(5,5))
        self.frame_controls.grid(column=1, row=0, sticky=(N, W, E, S))
        
        # --- Preset

        self.capture_preset_frame = ttk.LabelFrame(self.frame_controls, text="Capture Preset", padding=(5,5))
        self.capture_preset_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.capture_preset_frame, text="Capture Preset:").grid(column=0, row=0, sticky=(N,W))
        self.sv_capture_preset = StringVar()
        self.capture_preset_optionmenu = ttk.OptionMenu(self.capture_preset_frame, self.sv_capture_preset, DEFAULT_PRESET.value, *[preset.value for preset in PRESET_ORDER])
        self.capture_preset_optionmenu.grid(column=1, row=0, sticky=W)
        self.sv_capture_preset.trace_add("write", self.cb_set_capture_preset)

        # --- General Settings

        self.frame_controls_general = ttk.LabelFrame(self.frame_controls, text="General Settings", padding=(5,5))
        self.frame_controls_general.grid(column=0, row=1, sticky=(N, W, E, S))

        ttk.Label(self.frame_controls_general, text="Adapter IP").grid(column=0, row=0, sticky=(N,W))
        self.sv_adapter_ip = StringVar()
        self.sv_adapter_ip.set(DEFAULT_ADAPTER_IP)
        ttk.Entry(self.frame_controls_general, textvariable=self.sv_adapter_ip).grid(column=1, row=0, sticky=(N,W))
        
        
        ttk.Label(self.frame_controls_general, text="Gaze Socket Endpoint").grid(column=0, row=1, sticky=(N,W))
        self.sv_gaze_socket_url = StringVar()
        self.sv_gaze_socket_url.set("")
        ttk.Entry(self.frame_controls_general, textvariable=self.sv_gaze_socket_url).grid(column=1, row=1, sticky=(N,W))

        self.sv_sync_recording_with_stimulus = StringVar()
        self.sv_sync_recording_with_stimulus.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_general, text="Sync with Stimulus", variable=self.sv_sync_recording_with_stimulus, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=2, sticky=W)

        self.sv_use_external_trigger = StringVar()
        self.sv_use_external_trigger.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_general, text="Use External Trigger", variable=self.sv_use_external_trigger, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=3, sticky=W)

        self.sv_adjust_contrast = StringVar()
        self.sv_adjust_contrast.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_general, text="Auto Contrast", variable=self.sv_adjust_contrast, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=4, sticky=W)

        # --- Stream and Visualization

        self.frame_controls_stream = ttk.LabelFrame(self.frame_controls, text="Stream and Visualization", padding=(5,5))
        self.frame_controls_stream.grid(column=0, row=2, sticky=(N, W, E, S))
        
        self.sv_btn_start_stream = StringVar()
        self.sv_btn_start_stream.set("Start Stream")
        ttk.Button(self.frame_controls_stream, textvariable=self.sv_btn_start_stream, command=lambda: self.cb_toggle_capture(UICaptureState.STREAM)).grid(column=0, row=0, sticky=W)

        self.sv_vis_calibration_pattern = StringVar()
        self.sv_vis_calibration_pattern.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Calibration Pattern", variable=self.sv_vis_calibration_pattern, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=1, sticky=W)
        
        self.sv_vis_img_diff_thr = StringVar()
        self.sv_vis_img_diff_thr.set(SV_CHECKBOX_ACTIVE)
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Img Diff Threshold", variable=self.sv_vis_img_diff_thr, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=2, sticky=W)
        
        self.sv_auto_accept_img_diff_threshold = StringVar()
        self.sv_auto_accept_img_diff_threshold.set(DEFAULT_AUTO_ACCEPT_IMG_DIFF_THRESHOLD)
        ttk.Entry(self.frame_controls_stream, textvariable=self.sv_auto_accept_img_diff_threshold, width=10).grid(column=1, row=2, sticky=(N,W))
        
        self.sv_vis_pupil_detection = StringVar()
        self.sv_vis_pupil_detection.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_stream, text="Visualize Pupil Detection", variable=self.sv_vis_pupil_detection, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=3, sticky=W)

        # --- Capture

        self.frame_controls_capture = ttk.LabelFrame(self.frame_controls, text="Capture", padding=(5,5))
        self.frame_controls_capture.grid(column=0, row=3, sticky=(N, W, E, S))
        
        # ---- Export Settings

        self.frame_controls_capture_export = ttk.LabelFrame(self.frame_controls_capture, text="Export Settings", padding=(5,5))
        self.frame_controls_capture_export.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.frame_controls_capture_export, text="Path").grid(column=0, row=0, sticky=(N,W))
        self.sv_experiment_name = StringVar()
        self.sv_experiment_name.set(DEFAULT_EXPERIMENT_NAME)
        ttk.Entry(self.frame_controls_capture_export, textvariable=self.sv_experiment_name, width=50).grid(column=1, row=0, sticky=(N,W))

        # ---- Synchronized Capture

        self.frame_controls_capture_record = ttk.LabelFrame(self.frame_controls_capture, text="Synchronized Capture", padding=(5,5))
        self.frame_controls_capture_record.grid(column=0, row=1, sticky=(N, W, E, S))
        
        self.sv_recording = StringVar()
        self.sv_recording.set("Start Capture")
        ttk.Button(self.frame_controls_capture_record, textvariable=self.sv_recording, command=lambda: self.cb_toggle_capture(UICaptureState.CAPTURE)).grid(column=0, row=0, sticky=W)

        self.sv_capture_with_stream = StringVar()
        self.sv_capture_with_stream.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_capture_record, text="Show Stream", variable=self.sv_capture_with_stream, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=1, sticky=W)

        # ---- Pattern-Based Capture
        
        self.frame_controls_capture_pattern = ttk.LabelFrame(self.frame_controls_capture, text="Pattern-Based Capture", padding=(5,5))
        self.frame_controls_capture_pattern.grid(column=0, row=2, sticky=(N, W, E, S))

        ttk.Button(self.frame_controls_capture_pattern, text="Triggered Snapshot", command=self.cb_onclick_triggered_snapshot).grid(column=0, row=0, sticky=W)
        ttk.Button(self.frame_controls_capture_pattern, text="Accept", command=self.cb_accept_triggered_snapshot).grid(column=0, row=1, sticky=W)
        ttk.Button(self.frame_controls_capture_pattern, text="Stop", command=self.cb_stop_triggered_snapshot).grid(column=0, row=2, sticky=W)

        self.sv_auto_accept = StringVar()
        self.sv_auto_accept.set(SV_CHECKBOX_INACTIVE)
        ttk.Checkbutton(self.frame_controls_capture_pattern, text="Auto-Accept", variable=self.sv_auto_accept, onvalue=SV_CHECKBOX_ACTIVE, offvalue=SV_CHECKBOX_INACTIVE).grid(column=0, row=3, sticky=W)

        self.sv_capture_status = StringVar()
        self.sv_capture_status.set("Idle")
        ttk.Label(self.frame_controls_capture_pattern, textvariable=self.sv_capture_status).grid(column=0, row=5, sticky=W)

    
    def init_logic(self):
        self.cb_set_capture_preset()
    
    
    def get_current_ui_preset_config(self) -> UIPresetConfig:
        return CAPTURE_PRESET_CONFIGS[self.state.current_preset]
    
    
    def stop_all_capturing(self):
        if self.state.capture_state == UICaptureState.STREAM:
            self.cb_toggle_capture(UICaptureState.STREAM)
        elif self.state.capture_state == UICaptureState.CAPTURE:
            self.cb_toggle_capture(UICaptureState.CAPTURE)
        
        
    def cb_set_capture_preset(self, *args):
        # Stop capture and streaming before changing settings
        self.stop_all_capturing()
                    
        self.state.current_preset = UIPreset(self.sv_capture_preset.get())

        for camera_index, frame in self.ui_camera_frames.items():
            frame.sv_active.set(SV_CHECKBOX_ACTIVE if camera_index in self.get_current_ui_preset_config().camera_indices else SV_CHECKBOX_INACTIVE)

        self.sv_vis_calibration_pattern.set(SV_CHECKBOX_ACTIVE if self.get_current_ui_preset_config().use_pattern is not None else SV_CHECKBOX_INACTIVE)
        self.sv_vis_img_diff_thr.set(SV_CHECKBOX_ACTIVE if self.get_current_ui_preset_config().capture_settings.use_threshold else SV_CHECKBOX_INACTIVE)
        self.sv_auto_accept.set(SV_CHECKBOX_ACTIVE if self.get_current_ui_preset_config().capture_settings.auto_accept else SV_CHECKBOX_INACTIVE)
        self.sv_adjust_contrast.set(SV_CHECKBOX_ACTIVE if self.get_current_ui_preset_config().capture_settings.auto_contrast else SV_CHECKBOX_INACTIVE)


    def cb_onclick_triggered_snapshot(self):
        if self.state.capture_state != UICaptureState.STREAM:
            print("Stream not running, cannot capture.")
            return

        self.state.folder_manager.set_root_path(Path(self.sv_experiment_name.get()))
        self.state.folder_manager.set_save_path(self.state.folder_manager.get_path_for_step(self.state.current_preset))
        self.state.folder_manager.create_capture_folders(self.state.curr_capture_cam_idxs)
        self.state.snapshot_state = UISnapshotState.TRIGGERED_SNAPSHOT
        self.sv_capture_status.set("Waiting for trigger...")
        self.state.last_timestamp_auto_accept = None


    def cb_accept_triggered_snapshot(self, *args):
        if self.state.capture_state != UICaptureState.STREAM:
            print("Stream not running, cannot capture.")
            return

        if self.state.snapshot_state == UISnapshotState.AWAITING_ACCEPT:
            self.save_snapshot_from_capture_buffer()

            self.state.snapshot_state = UISnapshotState.COOLDOWN
            self.sv_capture_status.set("Snapshot saved.")


    def cb_stop_triggered_snapshot(self, *args):
        self.state.snapshot_state = UISnapshotState.IDLE
        self.state.last_timestamp_auto_accept = None
        self.sv_capture_status.set("Idle")


    def websocket_sync_handler_thread(self, camera_handler: CameraHandler, url: str="wss://hctlsrvc.edu.sot.tum.de/eventdetectionwsmarker2/"):
        async def websocket_sync_handler(camera_handler: CameraHandler, url: str):
            ws_messages = []
            
            system_ts = time.time_ns() / 1e9
            log_entry = {
                "system_unix_ts": system_ts,
                "ws_message": "START_RECORDING"
            }
            ws_messages.append(log_entry)
            
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
                                #print(f"WS Message: {json.dumps(log_entry)}")
                            
                            eventType = parsed.get("eventType")
                            
                            if eventType == "TaskStart":
                                print("Received TaskStart event.")
                                
                                # Start camera recording
                                #camera_handler.ev_start_capture.set()
                                
                            if eventType == "TaskEnd":
                                print("Received TaskEnd event.")
                                
                                # Stop camera recording
                                #camera_handler.ev_websocket_request_terminate.set()
                                break

                            ws_messages.append(log_entry)

                        except Exception as e:
                            print(f"WS receive error: {e}")
                            #camera_handler.ev_websocket_request_terminate.set()
                            break
                        
            except Exception as e:
                print(f"WebSocket connection failed: {e}")
                print("Terminating")
                camera_handler.ev_websocket_request_terminate.set()

            camera_handler.ws_message_q.put(ws_messages)

            return
        
        asyncio.run(websocket_sync_handler(camera_handler, url))
        return


    def cb_toggle_capture(self, target_capture_state: UICaptureState, *args):
        print("Capture triggered")

        if self.state.capture_state == UICaptureState.INACTIVE:
            self.state.curr_capture_cam_idxs = []
            self.state.curr_capture_cam_urls = {}

            for camera_index, cam_frame in self.ui_camera_frames.items():
                if cam_frame.sv_active.get() == SV_CHECKBOX_ACTIVE:
                    if CameraType(cam_frame.sv_type.get()) == CameraType.USB:
                        self.state.curr_capture_cam_urls[camera_index] = int(cam_frame.sv_ip.get())
                    else:
                        self.state.curr_capture_cam_urls[camera_index] = cam_frame.sv_ip.get()

                    self.state.curr_capture_cam_idxs.append(camera_index)
            
            self.state.capture_buffers.current = {
                cam_index: CaptureBuffers.CaptureBufferFrame() for cam_index in self.state.curr_capture_cam_idxs
            }
            self.state.capture_buffers.old = {
                cam_index: CaptureBuffers.CaptureBufferFrame() for cam_index in self.state.curr_capture_cam_idxs
            }

            if target_capture_state == UICaptureState.CAPTURE:
                stream_enabled = self.sv_capture_with_stream.get() == SV_CHECKBOX_ACTIVE
                recording_enabled = True
                self.sv_recording.set("Stop Capture")
                print("Capture activated")
                
                # Add datetime string to folder path
                now = datetime.now()
                dt_subfolder = now.strftime(RECORDING_SUBFOLDER_FORMAT)
                self.state.folder_manager.set_root_path(Path(self.sv_experiment_name.get()))
                self.state.folder_manager.set_save_path(self.state.folder_manager.get_path_for_step(self.state.current_preset) / dt_subfolder)
                self.state.folder_manager.create_capture_folders(self.state.curr_capture_cam_idxs)
            else:  # target_capture_state == UICaptureState.STREAM
                stream_enabled = True
                recording_enabled = False
                self.sv_btn_start_stream.set("Stop Stream")
                print("Stream activated")
            
            self.state.camera_handler = CameraHandler(
                camera_indexes=self.state.curr_capture_cam_idxs,
                urls=self.state.curr_capture_cam_urls,
                capture_folder_manager=self.state.folder_manager,
            )
                
            if self.sv_sync_recording_with_stimulus.get() == SV_CHECKBOX_ACTIVE:
                print("Syncing capture with stimulus...")
                self.state.capture_sync_mode = True
                gaze_socket_url = self.sv_gaze_socket_url.get()
                self.t_websocket_sync_handler = threading.Thread(target=self.websocket_sync_handler_thread, args=(self.state.camera_handler, gaze_socket_url))
            else:
                self.state.capture_sync_mode = False

            use_external_trigger = self.sv_use_external_trigger.get() == SV_CHECKBOX_ACTIVE

            self.state.capture_buffers.current = {
                cam_index: CaptureBuffers.CaptureBufferFrame() for cam_index in self.state.curr_capture_cam_idxs
            }

            self.t_rec = threading.Thread(target=record_frames_multithreaded, args=(self.state.camera_handler, stream_enabled, recording_enabled, False, use_external_trigger))  #self.state.capture_sync_mode
            self.t_rec.start()
            
            if self.state.capture_sync_mode:
                self.t_websocket_sync_handler.start()

            print("Camera threads started")

            self.state.capture_state = target_capture_state
            
        elif target_capture_state == self.state.capture_state:
            print(f"{target_capture_state} deactivated")
            self.state.camera_handler.ev_request_terminate.set()

            if target_capture_state == UICaptureState.STREAM and self.state.capture_state == UICaptureState.STREAM:
                # Relabel button
                self.sv_btn_start_stream.set("Start Stream")

            elif target_capture_state == UICaptureState.CAPTURE and self.state.capture_state == UICaptureState.CAPTURE:
                # Relabel button
                self.sv_recording.set("Start Capture")

                logs={
                    "camera_urls": {cam.value: url for cam, url in self.state.curr_capture_cam_urls.items()},
                    "camera_indices": [cam.value for cam in self.state.curr_capture_cam_idxs],
                    "folder_root_path": str(self.state.folder_manager.get_root_path()),
                    "folder_save_path": str(self.state.folder_manager.get_save_path()),
                }
                
                if self.state.capture_sync_mode:
                    print("Waiting for WebSocket sync handler to finish...")
                    self.t_websocket_sync_handler.join()
                    print("WebSocket sync handler joined")
                    
                    try:
                        ws_messages = self.state.camera_handler.ws_message_q.get_nowait()
                    except queue.Empty:
                        ws_messages = []
                    
                    logs["websocket_messages"] = ws_messages
                    logs["recording_begin_trigger"] = "VIVA Stimuli Websocket TaskStart"
                else:
                    logs["recording_begin_trigger"] = "Manual Capture"
                
                self.state.folder_manager.save_json(logs, self.state.folder_manager.get_save_path() / LOG_FILE_NAME)
            
            #self.t_rec.join()
            #print("Camera threads joined")
            # TODO shouldn't run in main thread, but in a separate thread to avoid blocking the GUI
            
            for camera_index in self.state.curr_capture_cam_idxs:
                self.ui_camera_frames[camera_index].w_canvas.create_rectangle(1, 1, 239, 239, outline='red', width=2)
                
            self.state.capture_state = UICaptureState.INACTIVE
            self.state.camera_handler = None

        else:
            print("Action not possible, CameraHandler is already running. Please stop current capture/stream before starting a new one.")
            return


    def save_snapshot_from_capture_buffer(self):
        # TODO add frame ID and/or x_timestamp to file name?
        # Add datetime string to file name path
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        for cam_index, buffer in self.state.capture_buffers.current.items():
            if buffer.frame_raw is not None:
                self.state.folder_manager.save_frame(frame=buffer.frame_raw, frame_id=None, timestamp=dt_string, camera_index=cam_index)
            else:
                print(f"Warning: No frame data available for camera {cam_index.value}, skipping snapshot save.")
    
    
    def update(self):
        # TODO use SSIM as similarity metric instead of mean absolute difference? https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
        
        # if self.state.capture_sync_mode:
        #     if self.state.camera_handler is not None and self.state.camera_handler.ev_websocket_request_terminate.is_set():
        #         print("WebSocket requested termination, stopping capture/stream.")
        #         self.cb_toggle_capture(self.state.capture_state)
        
        if self.state.capture_state != UICaptureState.INACTIVE:
            try:
                img_diff_threshold = float(self.sv_auto_accept_img_diff_threshold.get())
            except ValueError:
                img_diff_threshold = 0

            for camera_index in self.state.curr_capture_cam_idxs:
                if self.state.capture_state != UICaptureState.INACTIVE and not self.state.camera_handler.stream_qs[camera_index].empty():
                    self.state.capture_buffers.old = self.state.capture_buffers.current.copy()
                    
                    stream_entry: CaptureTransferBufferFrame = self.state.camera_handler.stream_qs[camera_index].get()
                    self.state.capture_buffers.current[camera_index] = CaptureBuffers.CaptureBufferFrame(
                        frame_raw=stream_entry.frame_raw.copy() if stream_entry.frame_raw is not None else None,
                        x_timestamp=stream_entry.x_timestamp,
                        x_timestamp_from_start=stream_entry.x_timestamp_from_start,
                        frame_idx=stream_entry.frame_idx,
                        fps=stream_entry.fps)

                    if self.state.capture_buffers.current[camera_index].frame_raw is not None:
                        imgdata = self.state.capture_buffers.current[camera_index].frame_raw.copy()

                        if self.state.capture_buffers.old[camera_index].frame_raw is not None:
                            self.state.capture_buffers.current[camera_index].img_diff = np.mean(cv2.absdiff(self.state.capture_buffers.old[camera_index].frame_raw, imgdata))
                        
                        if self.sv_adjust_contrast.get() == SV_CHECKBOX_ACTIVE:
                            imgdata, _, _ = automatic_brightness_and_contrast(imgdata, clip_hist_percent=0.5)

                        if self.sv_vis_calibration_pattern.get() == SV_CHECKBOX_ACTIVE:
                            imgdata, corners = detect_and_visualize_corners(imgdata, CAPTURE_PRESET_CONFIGS[self.state.current_preset].use_pattern.corners)

                            if corners is not None:
                                self.state.capture_buffers.current[camera_index].pattern_detected = True
                                
                        elif self.sv_vis_pupil_detection.get() == SV_CHECKBOX_ACTIVE:
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
                                self.state.capture_buffers.current[camera_index].frame = None

                            self.state.snapshot_state = UISnapshotState.AWAITING_ACCEPT

                        if (not self.state.snapshot_state == UISnapshotState.AWAITING_ACCEPT) and (self.state.last_timestamp_auto_accept is None or (time.time() - self.state.last_timestamp_auto_accept) > (COOLDOWN_AUTO_ACCEPT_RENDERING / 2)):
                            imgdata = resize_and_pad_image(imgdata, 240, 240)
                            imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
                            self.state.capture_buffers.current[camera_index].frame = PhotoImage(data=imgdata)
                            
                            self.ui_camera_frames[camera_index].l_frame_buffer = self.state.capture_buffers.current[camera_index].frame.copy()  # Store to avoid issues with garbage collector
                            self.ui_camera_frames[camera_index].w_canvas.delete("all")
                            self.ui_camera_frames[camera_index].w_canvas.create_image(0, 0, image=self.ui_camera_frames[camera_index].l_frame_buffer, anchor='nw')
                            if self.sv_vis_img_diff_thr.get() == SV_CHECKBOX_ACTIVE and self.state.capture_buffers.current[camera_index].img_diff is not None and self.state.capture_buffers.current[camera_index].img_diff <= img_diff_threshold:
                                self.ui_camera_frames[camera_index].w_canvas.create_rectangle(1, 1, 239, 239, outline='blue', width=2)
                        else:
                            self.ui_camera_frames[camera_index].w_canvas.create_rectangle(1, 1, 239, 239, outline='yellow', width=2)
                    
                    # Update camera info text
                    fps_str = f"{self.state.capture_buffers.current[camera_index].fps:.1f}" if self.state.capture_buffers.current[camera_index].fps is not None else "N/A"
                    frame_idx_str = f"{self.state.capture_buffers.current[camera_index].frame_idx}"
                    x_timestamp_str = f"{self.state.capture_buffers.current[camera_index].x_timestamp:.3f}" if self.state.capture_buffers.current[camera_index].x_timestamp is not None else "N/A"
                    x_timestamp_from_start_str = f"{self.state.capture_buffers.current[camera_index].x_timestamp_from_start:.3f}" if self.state.capture_buffers.current[camera_index].x_timestamp_from_start is not None else "N/A"
                    img_diff_str = f"{self.state.capture_buffers.current[camera_index].img_diff:.3f}" if self.state.capture_buffers.current[camera_index].img_diff is not None else "N/A"
                    additional_info = ""
                    if self.state.capture_buffers.current[camera_index].pattern_detected is not None:
                        additional_info += f"\nPattern Detected: {self.state.capture_buffers.current[camera_index].pattern_detected}"
                    if self.state.capture_buffers.current[camera_index].frame is None:
                        additional_info += "\nFrame: None"
                    self.ui_camera_frames[camera_index].sv_info.set(f"FPS: {fps_str}\nFrame: {frame_idx_str}\nX_t: {x_timestamp_str} s\nX_t elps: {x_timestamp_from_start_str} s\nDiff: {img_diff_str}\n{additional_info}")
            
            if self.state.snapshot_state == UISnapshotState.COOLDOWN:
                if self.state.last_timestamp_auto_accept is not None and (time.time() - self.state.last_timestamp_auto_accept) >= COOLDOWN_AUTO_ACCEPT:
                    self.state.last_timestamp_auto_accept = None
                    self.state.snapshot_state = UISnapshotState.TRIGGERED_SNAPSHOT
                    self.sv_capture_status.set("Waiting for pattern...")

            if self.state.snapshot_state == UISnapshotState.TRIGGERED_SNAPSHOT:
                snapshot_trigger_pattern = self.sv_vis_calibration_pattern.get() == SV_CHECKBOX_ACTIVE
                snapshot_trigger_img_threshold = self.sv_vis_img_diff_thr.get() == SV_CHECKBOX_ACTIVE
                if not snapshot_trigger_pattern or all(buf.pattern_detected for buf in self.state.capture_buffers.current.values()):
                    if not snapshot_trigger_img_threshold or all(buf.img_diff is not None and buf.img_diff <= img_diff_threshold for buf in self.state.capture_buffers.current.values()):

                        if self.sv_auto_accept.get() == SV_CHECKBOX_ACTIVE:
                            self.state.snapshot_state = UISnapshotState.AWAITING_ACCEPT
                            self.state.last_timestamp_auto_accept = time.time()
                            self.cb_accept_triggered_snapshot()
                            self.sv_capture_status.set("Snapshot saved.")
                        else:
                            self.state.snapshot_state = UISnapshotState.AWAITING_ACCEPT
                            self.sv_capture_status.set("Pattern detected! Accept or Reject.")