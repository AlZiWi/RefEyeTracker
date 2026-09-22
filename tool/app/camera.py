from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import multiprocessing
import multiprocessing.synchronize
import os
import subprocess
from pathlib import Path
import struct
from typing import Optional
import requests
import cv2
import numpy as np
import threading
import time
import queue

from app.data_structures import CameraIndex, CameraType
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
            print(f"{self.time_steps['names'][i]}: {self.time_steps['deltas'][i]:0.3f}s")
            
    def get(self):
        return self.time_steps


class CameraHandler:
    
    adapter_ip: str  # IP address of the network adapter to use for camera connections
    session: requests.Session  # HTTP session for camera connections
    debug: bool  # Debug flag for logging
    capture_folder_manager: CaptureFolderManager  # Manager for handling capture folders
    camera_indexes: list[CameraIndex]  # List of camera indexes for capturing
    camera_types: dict[CameraIndex, CameraType]  # List of camera indexes for capturing
    urls: dict[CameraIndex, str]  # List of camera URLs or USB indices
    
    ev_request_terminate: multiprocessing.synchronize.Event  # Event to signal termination of camera capture
    ev_websocket_request_terminate: multiprocessing.synchronize.Event  # Event to signal termination of websocket connection
    ev_start_capture: multiprocessing.synchronize.Event  # Event to signal start of camera capture
    ev_running: multiprocessing.synchronize.Event  # Event to signal that camera capture is running
    
    ws_message_q: multiprocessing.Queue  # Queue for websocket messages
    stream_qs: dict[CameraIndex, multiprocessing.Queue]  # List of queues for streaming frames
    recording_qs: dict[CameraIndex, multiprocessing.Queue]  # List of queues for recording frames
    
    def __init__(self, camera_indexes: list[CameraIndex], camera_types: dict[CameraIndex, CameraType], urls: dict[CameraIndex, str], capture_folder_manager: CaptureFolderManager, adapter_ip: str | None=None, debug: bool=False):
        self.urls = urls
        self.adapter_ip = adapter_ip
        self.session = None
        self.debug = debug
        self.capture_folder_manager = capture_folder_manager
        self.camera_indexes = camera_indexes
        self.camera_types = camera_types

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
    x_timestamp_hw: Optional[float] = None
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
                jpeg_start = bstr.find(b'\xff\xd8')
                jpeg_end = bstr.rfind(b'\xff\xd9')
                if jpeg_start < 0 or jpeg_end < jpeg_start:
                    bstr = b''
                    continue
                bstr = bstr[jpeg_start:jpeg_end + 2]
                
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
                    if frame_cv is None:
                        bstr = b''
                        continue
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


def record_frames_multithreaded(camera_handler: CameraHandler, stream_enabled: bool=False, recording_enabled: bool=False, request_sync_enabled: bool=False, trigger_enabled: bool=False):
        # Start threads for each url
        threads = []
        threads_rec = []
        
        # Reset events and results
        camera_handler.ev_request_terminate.clear()
        
        for camera_index in camera_handler.camera_indexes:
            url = camera_handler.urls[camera_index]
            cam_type = camera_handler.camera_types[camera_index]
            if cam_type == CameraType.USB:
                t = threading.Thread(target=record_frames_from_usb, args=(camera_handler, camera_index, stream_enabled, recording_enabled, trigger_enabled))
            else:
                t = threading.Thread(target=record_frames_from_ip, args=(camera_handler, camera_index, stream_enabled, recording_enabled, trigger_enabled))
                if recording_enabled:
                    t_rec = threading.Thread(target=save_frames_threaded, args=(camera_handler, camera_index))
                    threads_rec.append(t_rec)
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
            

# Detect OS and set the appropriate libuvc binary path
if os.name == "nt":
    LIBUVC_BINARY = str(Path(__file__).parent / "libuvc_wrapper" / "libuvc_util_win.exe")
elif os.name == "posix":
    if os.uname().sysname == "Darwin":
        LIBUVC_BINARY = str(Path(__file__).parent / "libuvc_wrapper" / "libuvc_util_mac")
    else:
        raise RuntimeError(f"Unsupported OS: {os.name}")
else:
    raise RuntimeError(f"Unsupported OS: {os.name}")

STREAM_HEADER = struct.Struct("<4sIQIIIIQQ")
STREAM_MAGIC = b"UVCF"
FRAME_FORMAT_YUYV = 3
FRAME_FORMAT_MJPEG = 7


def _complete_jpeg(data):
    start = data.find(b"\xff\xd8")
    end = data.rfind(b"\xff\xd9")
    if start < 0 or end < start:
        return None
    return data[start:end + 2]


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _print_process_errors(process, stop_event, camera_name):
    for line in iter(process.stderr.readline, b""):
        if line:
            print(f"libuvc[{camera_name}]: {line.decode(errors='replace').rstrip()}")
    stop_event.set()


def _read_libuvc_stream(process, frames, stop_event):
    try:
        while not stop_event.is_set():
            header_bytes = _read_exact(process.stdout, STREAM_HEADER.size)
            if header_bytes is None:
                print("libuvc stream ended unexpectedly")
                return
            magic, frame_id, seconds, milliseconds, width, height, frame_format, frame_size, metadata_size = STREAM_HEADER.unpack(header_bytes)
            if magic != STREAM_MAGIC:
                raise RuntimeError(f"Invalid libuvc stream magic: {magic!r}")
            if frame_size > 100 * 1024 * 1024 or metadata_size > 10 * 1024 * 1024:
                raise RuntimeError("Invalid libuvc stream record size")
            frame_data = _read_exact(process.stdout, frame_size)
            metadata = _read_exact(process.stdout, metadata_size)
            if frame_data is None or metadata is None:
                return
            item = (frame_id, seconds + milliseconds / 1000.0, width, height, frame_format, frame_data, metadata)
            try:
                frames.put(item, timeout=0.01)
            except queue.Full:
                pass
    except Exception as error:
        try:
            frames.put_nowait(error)
        except queue.Full:
            pass


def record_frames_from_usb(camera_handler: CameraHandler, camera_index: CameraIndex, stream_enabled: bool=False, recording_enabled: bool=False, external_trigger_enabled: bool=False):
    serial_idx = camera_handler.urls[camera_index]
    width, height = (640, 480) if camera_index == CameraIndex.SC else (240, 240)
    command = [LIBUVC_BINARY, "-name", str(serial_idx), "-resx", str(width), "-resy", str(height), "-fps", "60", "-stream"]
    if external_trigger_enabled:
        command.append("-trigger")
        command.append("true")
    else:
        command.append("-trigger")
        command.append("false")
        
    if recording_enabled:
        command.append("-out")
        command.append(str(camera_handler.capture_folder_manager.get_save_path() / camera_index.value))
        
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    except OSError as error:
        print(f"Failed to start libuvc camera {serial_idx}: {error}")
        return

    frame_queue = queue.Queue(maxsize=1)
    reader_stop = threading.Event()
    reader = threading.Thread(target=_read_libuvc_stream, args=(process, frame_queue, reader_stop), daemon=True)
    error_reader = threading.Thread(
        target=_print_process_errors,
        args=(process, reader_stop, serial_idx),
        daemon=True,
    )
    reader.start()
    error_reader.start()
    fps_window_len = 10
    fps_buffer = np.zeros(fps_window_len)
    start_time = 0
    frame_count = 0

    try:
        while not camera_handler.ev_request_terminate.is_set():
            try:
                frame_data = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if isinstance(frame_data, Exception):
                print(f"Failed to read frame from USB camera {serial_idx}: {frame_data}")
                break

            frame_id, hardware_timestamp, frame_width, frame_height, frame_format, raw_frame, metadata = frame_data
            if stream_enabled or recording_enabled:
                raw_array = np.frombuffer(raw_frame, dtype=np.uint8)
                if frame_format == FRAME_FORMAT_YUYV:
                    frame = cv2.cvtColor(raw_array.reshape((frame_height, frame_width, 2)), cv2.COLOR_YUV2BGR_YUYV)
                elif frame_format == FRAME_FORMAT_MJPEG:
                    jpeg = _complete_jpeg(raw_frame)
                    if jpeg is None:
                        continue
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                else:
                    print(f"Skipping unsupported frame format {frame_format} from camera {serial_idx}")
                    continue
                if frame is None:
                    print(f"Could not decode JPEG frame {frame_id} from camera {serial_idx}")
                    continue
                frame_cv = frame
            else:
                frame_cv = None

            now = time.perf_counter()
            if start_time == 0:
                start_time = now
            window_len = min(frame_count + 1, fps_window_len)
            fps_buffer[frame_count % window_len] = now
            fps = 0.0 if frame_count <= 1 else (window_len - 1) / (fps_buffer[frame_count % window_len] - fps_buffer[(frame_count + 1) % window_len])
            x_timestamp = now - start_time
            transfer = CaptureTransferBufferFrame(
                frame_raw=frame_cv.copy() if stream_enabled else None,
                x_timestamp=x_timestamp,
                x_timestamp_from_start=x_timestamp,
                x_timestamp_hw=hardware_timestamp,
                frame_idx=frame_id,
                fps=fps,
            )
            try:
                camera_handler.stream_qs[camera_index].put_nowait(transfer)
            except queue.Full:
                pass
            frame_count += 1
    finally:
        reader_stop.set()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=2)

    if process.returncode not in (0, -15, -9) and not camera_handler.ev_request_terminate.is_set():
        print(f"libuvc camera {serial_idx} exited with status {process.returncode}")

    if camera_handler.debug:
        print(f"USB camera {serial_idx} recording finished. Captured {frame_count} frames.")


if __name__ == "__main__":
    # Example usage
    usb_index = 0  # Change this to the index of your USB camera
    camera_handler = CameraHandler(urls=[usb_index], capture_folder_path="captures", capture_cam_labels=["usb_camera"], debug=True)
    record_frames_from_usb(camera_handler, 0, count=100, stream_enabled=True, recording_enabled=True)