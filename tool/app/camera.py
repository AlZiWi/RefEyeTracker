from asyncio import subprocess
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing
import multiprocessing.synchronize
import os
import subprocess
from typing import Optional
import requests
import cv2
import numpy as np
import threading
import time
import queue
import sys

from app.data_structures import CameraIndex
from app.utils import CaptureFolderManager


class VTimer:
    def __init__(self):
        self.time_steps = {
            "names": [],
            "times": np.array([]),
            "deltas": np.array([])
        }
        
        self.time_steps["names"] = ["init"]
        self.time_steps["times"] = np.array([time.perf_counter()])
        self.time_steps["deltas"] = np.array([0])

    def time_log(self, name=""):
        curr_time = time.perf_counter()
        self.time_steps["names"].append(name)
        self.time_steps["times"] = np.append(self.time_steps["times"], curr_time)
        self.time_steps["deltas"] = np.append(self.time_steps["deltas"], self.time_steps["times"][-1] - self.time_steps["times"][-2])

    def time_print_log(self):
        print(f"Times:")
        for i in range(len(self.time_steps["names"])):
            print(f"{self.time_steps["names"][i]}: {self.time_steps["deltas"][i]:0.3f}s")
            
    def get(self):
        return self.time_steps
    
    
class CameraHandler:
    
    adapter_ip: str  # IP address of the network adapter to use for camera connections
    session: requests.Session  # HTTP session for camera connections
    debug: bool  # Debug flag for logging
    capture_folder_manager: CaptureFolderManager  # Manager for handling capture folders
    camera_indexes: list[CameraIndex]  # List of camera indexes for capturing
    urls: dict[CameraIndex, str]  # List of camera URLs or USB indices
    
    ev_request_terminate: multiprocessing.synchronize.Event  # Event to signal termination of camera capture
    ev_websocket_request_terminate: multiprocessing.synchronize.Event  # Event to signal termination of websocket connection
    ev_start_capture: multiprocessing.synchronize.Event  # Event to signal start of camera capture
    ev_running: multiprocessing.synchronize.Event  # Event to signal that camera capture is running
    
    ws_message_q: multiprocessing.Queue  # Queue for websocket messages
    stream_qs: dict[CameraIndex, multiprocessing.Queue]  # List of queues for streaming frames
    recording_qs: dict[CameraIndex, multiprocessing.Queue]  # List of queues for recording frames
    
    def __init__(self, camera_indexes: list[CameraIndex], urls: dict[CameraIndex, str], capture_folder_manager: CaptureFolderManager, adapter_ip: str | None=None, debug: bool=False):
        self.urls = urls
        self.adapter_ip = adapter_ip
        self.session = None
        self.debug = debug
        self.capture_folder_manager = capture_folder_manager
        self.camera_indexes = camera_indexes

        # Events

        self.ev_request_terminate = multiprocessing.Event()
        self.ev_websocket_request_terminate = multiprocessing.Event()
        self.ev_start_capture = multiprocessing.Event()
        self.ev_running = multiprocessing.Event()

        # Queues

        self.ws_message_q = multiprocessing.Queue()
        
        self.stream_qs = {}
        self.recording_qs = {}
        for camera_index in self.camera_indexes:
            self.stream_qs[camera_index] = multiprocessing.Queue(maxsize=1)
            self.recording_qs[camera_index] = multiprocessing.Queue()

    
@dataclass
class CaptureTransferBufferFrame:
    frame_raw: Optional[np.ndarray] = None
    x_timestamp: Optional[float] = None
    x_timestamp_from_start: Optional[float] = None
    frame_idx: Optional[int] = None
    fps: Optional[float] = None



# TODO outdated
def record_frames_from_ip(camera_handler, thr_idx, stream_enabled=False, recording_enabled=False, trigger_enabled=False):

    def _session_for_src_addr(addr: str) -> requests.Session:
        """
        Create `Session` which will bind to the specified local address
        rather than auto-selecting it.
        """
        session = requests.Session()
        for prefix in ('http://', 'https://'):
            session.get_adapter(prefix).init_poolmanager(
                # those are default values from HTTPAdapter's constructor
                connections=requests.adapters.DEFAULT_POOLSIZE,
                maxsize=requests.adapters.DEFAULT_POOLSIZE,
                # This should be a tuple of (address, port). Port 0 means auto-selection.
                source_address=(addr, 0),
            )

        return session
    
    url = camera_handler.urls[thr_idx]
    
    # Append trigger endpoint if enabled
    if trigger_enabled:
        if url[-1] == '/':
            url = url[:-1]
        url += "/trigger"

    content_length = 0
    content_type = ""
    timestamp = 0.0
    initial_timestamp = None

    curr_frame_count = 0
    bstr = b''

    metadata = []
    
    vtimer = VTimer()

    # Channel through selected adapter IP
    s = _session_for_src_addr(camera_handler.adapter_ip)  # from system settings‚
    requests_stream = s.get(url, stream=True, timeout=10)

    # Parse frames
    for line in requests_stream.iter_lines(delimiter=b'\r\n'):
        if camera_handler.ev_request_terminate.is_set():
            break

        if camera_handler.debug:
            print(line)
            
        if line.startswith(b'--'):
            vtimer.time_log(str(curr_frame_count))

            if content_type != b'' and content_length > 0 and timestamp > 0.0:
                if camera_handler.debug:
                    print("#####  New frame  #####")

                # JPEG boundaries
                bstr = bstr[bstr.rfind(b'\xff\xd8'):bstr.rfind(b'\xff\xd9')+2]
                
                metadata.append({
                    "frame_idx": curr_frame_count,
                    "content_length": content_length,
                    "content_type": content_type.decode('utf-8'),
                    "x_timestamp": timestamp,
                    "x_timestamp_from_start": timestamp - initial_timestamp if initial_timestamp is not None else 0.0
                })
        
                curr_frame_count += 1
                
                if camera_handler.debug:
                    print(f"Frame {curr_frame_count} captured from {url} with timestamp {timestamp}")
                
                if curr_frame_count >= 3:
                    times = np.array([md["x_timestamp"] for md in metadata[max(0, curr_frame_count - 50):-1]])
                    deltas = np.diff(times)
                    fps = 1 / np.mean(deltas)
                else:
                    fps = 0.0
                    
                if stream_enabled or recording_enabled:
                    frame_np = np.frombuffer(bstr, np.uint8)
                    frame_cv = cv2.imdecode(frame_np, cv2.IMREAD_GRAYSCALE)
                else:
                    frame_cv = None
                    
                if camera_handler.ev_request_terminate.is_set():
                    if camera_handler.debug:
                        print(f"Terminating USB camera {url}.")
                    break
                    
                try:
                    camera_handler.stream_qs[thr_idx].put_nowait({
                        "metadata": metadata.copy(),
                        "time_steps": vtimer.get(),
                        "fps": fps,
                        "frame": frame_cv.copy() if stream_enabled else None
                    })
                except queue.Full:
                    try:
                        camera_handler.stream_qs[thr_idx].get_nowait()  # drop oldest
                        camera_handler.stream_qs[thr_idx].put_nowait({
                            "metadata": metadata.copy(),
                            "time_steps": vtimer.get(),
                            "fps": fps,
                            "frame": frame_cv.copy() if stream_enabled else None
                        })
                    except queue.Empty:
                        pass
                
                if recording_enabled:
                    try:
                        camera_handler.recording_qs[thr_idx].put_nowait({
                            "metadata": metadata.copy(),
                            "time_steps": vtimer.get(),
                            "fps": fps,
                            "frame": frame_cv.copy()
                        })
                    except queue.Full:
                        pass
                
        elif line.startswith(b'Content-Type:'):
            content_type = line.split(b' ')[1]
        elif line.startswith(b'Content-Length:'):
            content_length = int(line.split(b' ')[1])
        elif line.startswith(b'X-Timestamp:'):
            timestamp = float(line.split(b' ')[1])
            if initial_timestamp is None:
                initial_timestamp = timestamp
        else:
            bstr += line + b'\r\n'  # add \n back

    return


def record_frames_from_usb(camera_handler: CameraHandler, camera_index: CameraIndex, stream_enabled: bool=False, recording_enabled: bool=False, external_trigger_enabled: bool=False):
    serial_idx = camera_handler.urls[camera_index]

    # ring buffer for fps
    FPS_WINDOW_LEN = 10
    fps_buffer = np.zeros(FPS_WINDOW_LEN)
    start_time = 0

    api_preference = cv2.CAP_ANY  # default
    if sys.platform == "win32":
        api_preference = cv2.CAP_DSHOW  # DirectShow backend for Windows
    elif sys.platform == "darwin":
        api_preference = cv2.CAP_AVFOUNDATION  # AVFoundation backend for macOS
    elif sys.platform.startswith("linux"):
        api_preference = cv2.CAP_V4L2  # V4L2 backend for Linux
        
    # Apply trigger mode settings based on platform and external trigger flag
    if camera_index == CameraIndex.SC:
        if sys.platform == "darwin":
            # call ucv_utils to set trigger mode for macOS: ./uvc-util -V 0x0c45:0x636d -s auto-focus=true
            auto_focus_str = "true" if external_trigger_enabled else "false"
            ret = subprocess.run(["./app/uvc-util/uvc-util", "-V", "0x0c45:0x636d", "-s", f"auto-focus={auto_focus_str}"], capture_output=True, text=True)
            print(f"Camera {serial_idx} autofocus set to {auto_focus_str}, return: {ret.stdout.strip()}")
            ret = subprocess.run(["./app/uvc-util/uvc-util", "-V", "0x0c45:0x636d", "-g", "auto-focus"], capture_output=True, text=True)
            print(f"Camera {serial_idx} autofocus set to {auto_focus_str}, actual value: {ret.stdout.strip()}")
        
    cap = cv2.VideoCapture(serial_idx, api_preference)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    width = 240
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Apply trigger mode settings based on platform and external trigger flag
    if camera_index == CameraIndex.SC:
        if sys.platform == "win32" or sys.platform.startswith("linux"):
            if external_trigger_enabled:
                ok = cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # remapped to trigger mode for some cameras
                actual_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
                print(f"Camera {serial_idx} autofocus set to 1 {ok}, actual value: {actual_af}")
            else:
                ok = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # remapped to continuous autofocus mode for some cameras
                actual_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
                print(f"Camera {serial_idx} autofocus set to 0 {ok}, actual value: {actual_af}")
    
    frame_idx = 0
    while True:
        if camera_handler.ev_request_terminate.is_set():
            if camera_handler.debug:
                print(f"Terminating USB camera {serial_idx}.")
            break
        
        ret, frame = cap.read()
        
        if not ret:
            if camera_handler.debug:
                print(f"Failed to read frame from USB camera {serial_idx}.")
            continue  # Skip this iteration if frame read fails, can happen if camera is externally triggered and times out due to lack of trigger signals.

        if start_time == 0:
            start_time = time.perf_counter()

        curr_fps_window_len = min(frame_idx + 1, FPS_WINDOW_LEN)
        fps_buffer[frame_idx % curr_fps_window_len] = time.perf_counter()
        if frame_idx > 1:
            fps = 1.0 / (fps_buffer[(frame_idx) % curr_fps_window_len] - fps_buffer[(frame_idx + 1) % curr_fps_window_len]) * (float(curr_fps_window_len) - 1)
        else:
            fps = 0.0

        x_timestamp = float(fps_buffer[frame_idx % curr_fps_window_len] - start_time)  # relative timestamp from start
        #x_timestamp_hw = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #x_frame_idx_hw = cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # if camera_handler.debug:
        #     print(f"Frame {frame_idx} captured from {serial_idx} with timestamp {x_timestamp}")
        
        if stream_enabled or recording_enabled:
            frame_cv = frame
        else:
            frame_cv = None
            
        if camera_handler.ev_request_terminate.is_set():
            if camera_handler.debug:
                print(f"Terminating USB camera {serial_idx}.")
            break
        
        try:
            camera_handler.stream_qs[camera_index].put_nowait(CaptureTransferBufferFrame(
                frame_raw = frame_cv.copy() if stream_enabled else None,
                x_timestamp = x_timestamp,
                x_timestamp_from_start = x_timestamp,
                frame_idx = frame_idx,
                fps = fps,
            ))
        except queue.Full:
            pass
            # try:
            #     camera_handler.stream_qs[thr_idx].get_nowait()  # drop oldest
            #     camera_handler.stream_qs[thr_idx].put_nowait({
            # except queue.Empty:
            #     pass
        
        if recording_enabled:
            try:
                camera_handler.recording_qs[camera_index].put_nowait(CaptureTransferBufferFrame(
                    frame_raw = frame_cv.copy(),
                    x_timestamp = x_timestamp,
                    x_timestamp_from_start = x_timestamp,
                    frame_idx = frame_idx,
                    fps = fps,
                ))
            except queue.Full:
                pass
        
        frame_idx += 1
    
    cap.release()

    if camera_handler.debug:
        print(f"USB camera {serial_idx} recording finished. Captured {frame_idx} frames.")

    return


def record_frames_multithreaded(camera_handler: CameraHandler, stream_enabled: bool=False, recording_enabled: bool=False, request_sync_enabled: bool=False, trigger_enabled: bool=False):
        # Start threads for each url
        threads = []
        threads_rec = []
        
        # Reset events and results
        camera_handler.ev_request_terminate.clear()
        
        for camera_index in camera_handler.camera_indexes:
            url = camera_handler.urls[camera_index]
            if type(url) == int:
                t = threading.Thread(target=record_frames_from_usb, args=(camera_handler, camera_index, stream_enabled, recording_enabled, trigger_enabled))
                if recording_enabled:
                    t_rec = threading.Thread(target=save_frames_threaded, args=(camera_handler, camera_index))
                    threads_rec.append(t_rec)
            else:
                t = threading.Thread(target=record_frames_from_ip, args=(camera_handler, camera_index, stream_enabled, recording_enabled, trigger_enabled))
            threads.append(t)

        if request_sync_enabled:
            while True:
                if camera_handler.ev_request_terminate.is_set():
                    return
                if camera_handler.ev_start_capture.is_set():
                    break
                time.sleep(0.01)  # Wait for start event
                
        # Start each thread
        for t in threads:
            t.start()

        for t_rec in threads_rec:
            t_rec.start()

        camera_handler.ev_running.set()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        for t_rec in threads_rec:
            t_rec.join()

        camera_handler.ev_running.clear()

        return
    

def save_frames_threaded(camera_handler: CameraHandler, camera_index: CameraIndex):
    
    l_capture_folder_manager = CaptureFolderManager(root_path=camera_handler.capture_folder_manager.get_root_path(), save_path=camera_handler.capture_folder_manager.get_save_path())  # local instance for thread
    
    while True:
        if camera_handler.ev_request_terminate.is_set():
            break
        try:
            frame_q_data: CaptureTransferBufferFrame = camera_handler.recording_qs[camera_index].get(timeout=0.1)
            l_capture_folder_manager.save_frame(frame_q_data.frame_raw, frame_q_data.frame_idx, frame_q_data.x_timestamp_from_start, camera_index)
        except queue.Empty:
            # wait for a short time to avoid busy waiting
            time.sleep(0.01)
            continue
    
    # gather remaining frames after termination
    while not camera_handler.recording_qs[camera_index].empty():
        try:
            frame_q_data: CaptureTransferBufferFrame = camera_handler.recording_qs[camera_index].get_nowait()
            l_capture_folder_manager.save_frame(frame_q_data.frame_raw, frame_q_data.frame_idx, frame_q_data.x_timestamp_from_start, camera_index)
        except queue.Empty:
            break
            

if __name__ == "__main__":
    # Example usage
    usb_index = 0  # Change this to the index of your USB camera
    camera_handler = CameraHandler(urls=[usb_index], capture_folder_path="captures", capture_cam_labels=["usb_camera"], debug=True)
    record_frames_from_usb(camera_handler, 0, count=100, stream_enabled=True, recording_enabled=True)