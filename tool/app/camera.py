import multiprocessing
from queue import LifoQueue
import re
import requests
import cv2
import numpy as np
import threading
import time
import queue

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
    

def record_frames_from_ip(camera_handler, thr_idx, count=None, stream_enabled=False, recording_enabled=False, trigger_enabled=False):

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

        if count is not None:
            if curr_frame_count > count:
                break
                
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

# Capture stream from USB camera
def record_frames_from_usb(camera_handler, thr_idx, count=None, stream_enabled=False, recording_enabled=False):
    serial_idx = camera_handler.urls[thr_idx]
    
    metadata = []

    vtimer = None

    cap = cv2.VideoCapture(serial_idx, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    width = 240
    height = 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
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
        else:
            if vtimer is None:
                vtimer = VTimer()

        vtimer.time_log(frame_idx)

        x_timestamp = vtimer.get()["times"][-1] - vtimer.get()["times"][0]  # relative timestamp from start
        x_timestamp_hw = cap.get(cv2.CAP_PROP_POS_FRAMES)
        x_frame_idx_hw = cap.get(cv2.CAP_PROP_POS_MSEC)
        metadata.append({
            "frame_idx": frame_idx,
            "x_timestamp": x_timestamp,
            "x_timestamp_from_start": x_timestamp
        })
        
        frame_idx += 1
        
        if camera_handler.debug:
            print(f"Frame {frame_idx} captured from {serial_idx} with timestamp {x_timestamp}")
            
        if frame_idx >= 3:
            times = np.array([md["x_timestamp"] for md in metadata[max(0, frame_idx - 50):-1]])
            deltas = np.diff(times)
            fps = 1 / np.mean(deltas)
        else:
            fps = 0.0
            
        
        if stream_enabled or recording_enabled:
            frame_cv = frame
        else:
            frame_cv = None
            
        if camera_handler.ev_request_terminate.is_set():
            if camera_handler.debug:
                print(f"Terminating USB camera {serial_idx}.")
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
        
        if count is not None and frame_idx >= count:
            if camera_handler.debug:
                print(f"Reached frame count limit for USB camera {serial_idx}.")
            break
    
    cap.release()

    if camera_handler.debug:
        print(f"USB camera {serial_idx} recording finished. Captured {frame_idx} frames.")

    return


def record_frames_multithreaded(camera_handler, count=None, stream_enabled=False, recording_enabled=False, request_sync_enabled=False, trigger_enabled=False):
        # Start threads for each url
        threads = []
        
        # Reset events and results
        camera_handler.ev_request_terminate.clear()
        
        for idx, url in enumerate(camera_handler.urls):
            if type(url) == int:
                t = threading.Thread(target=record_frames_from_usb, args=(camera_handler, idx, count, stream_enabled, recording_enabled))
            else:
                t = threading.Thread(target=record_frames_from_ip, args=(camera_handler, idx, count, stream_enabled, recording_enabled, trigger_enabled))
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

        camera_handler.ev_running.set()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        camera_handler.ev_running.clear()

        return
    

#  Class mostly used for thread synchronization
class CameraHandler:
    def __init__(self, urls, adapter_ip=None, debug=False):
        self.urls = urls
        self.adapter_ip = adapter_ip
        self.session = None
        self.debug = debug
        
        # Events

        self.ev_request_terminate = multiprocessing.Event()
        self.ev_websocket_request_terminate = multiprocessing.Event()
        self.ev_start_capture = multiprocessing.Event()
        self.ev_running = multiprocessing.Event()

        # Queues

        self.ws_message_q = multiprocessing.Queue()
        
        self.stream_qs = []
        self.recording_qs = []
        for i in range(len(self.urls)):
            self.stream_qs.append(multiprocessing.Queue(maxsize=1))
            self.recording_qs.append(multiprocessing.Queue())