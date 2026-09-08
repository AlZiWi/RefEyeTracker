from enum import Enum

from app.data_structures import CameraIndex


class GazeViewModes(Enum):
    
    def __init__(self, description: str, path_suffix: str):
        self.description: str = description
        self.path_suffix: str = path_suffix

    INPUT = "Input", ""
    SEGMENTATION = "Segmentation", "_output/segmentation"
    GAZE_ESTIMATION = "Gaze Estimation", "_output/gaze_estimation"


ESTIMATION_OUTPUT_FILENAME: str = "gaze_estimation_results"
CAMERA_LABEL_ORDER: list[CameraIndex] = [CameraIndex.RO, CameraIndex.RI, CameraIndex.LO, CameraIndex.LI]

DEFAULT_EXPERIMENT_NAME: str = "recordings/a_test/saccade33_1m__2026_03_12_17_15_35"
DEFAULT_CALIBRATION_NAME: str = "recordings/calibration_2026_05"


# class UITabGazeEstimation:
#     @dataclass
#     class UIAppGazeState:
        
#         @dataclass
#         class UIGazeThreadHandler:
#             status_q: multiprocessing.Queue = multiprocessing.Queue()
#             thread_status: str = "idle"
#             thread: multiprocessing.Process | None = None
            
#         gaze_thread_handler: UIGazeThreadHandler = UIGazeThreadHandler()
#         gaze_imgs: dict = {}  # list of available frames and filenames (filled by load_gaze_paths)
#         gaze_current_frame_idx: int = 0
#         gaze_max_frame_idxs: dict = {cam_label: 0 for cam_label in self.camera_settings.keys()}
#         gaze_animating: bool = False
#         gaze_anim_after_id: str | None = None
    
#     state: UIAppGazeState
        
#     def __init__(self, parent_frame: ttk.Frame):
#         self.state = self.UIAppGazeState()

#         # Layout
        
#         self.frame_gaze_estimation = ttk.LabelFrame(parent_frame, text="Gaze Estimation", padding=(5,5))
#         self.frame_gaze_estimation.grid(column=0, row=0, sticky=(N, W, E, S))

#         # --- Gaze Estimation Visualization Layout
#         self.gaze_vis_frame = ttk.LabelFrame(self.frame_gaze_estimation, text="Visualization", padding=(5,5))
#         self.gaze_vis_frame.grid(column=0, row=0, sticky=(N, W, E, S))

#         # Small camera canvases (2x2)
#         class UIGazeCamFrame:
#             def __init__(self, parent_frame: ttk.Frame, label: CameraIndex, i: int):
#                 self.frm = ttk.Frame(parent_frame)
                
#                 if label == CameraIndex.SC:
#                     self.frm.grid(column=0, row=4, columnspan=2, padx=5, pady=5, sticky=(N, W))
#                 else:
#                     self.frm.grid(column=i % 2, row=(i // 2) * 2, padx=5, pady=5, sticky=(N, W))

#                 ttk.Label(self.frm, text=f"Cam [{label}]").grid(column=0, row=0, sticky=(N, W))

#                 self.subframe = ttk.Frame(self.frm)
#                 self.subframe.grid(column=0, row=0, sticky=(N, W))

#                 if label == CameraIndex.SC:
#                     # Scene camera and gaze estimation results are larger
                    
#                     self.c = Canvas(self.subframe, width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
#                     self.c.grid(column=0, row=1, sticky=(N, W))
#                     self.c.xview_moveto(5)  # ref highlightthickness
#                     self.c.yview_moveto(5)  # ref highlightthickness

#                     self.info = ttk.Label(self.subframe, text="--", font=("Courier", 9))
#                     self.info.grid(column=0, row=2, sticky=(N, W))
                    
#                     ttk.Label(self.subframe, text="Gaze Results").grid(column=1, row=0, sticky=(N, W))
#                     self.c2 = FigureCanvasTkAgg(Figure(figsize=(3, 3)), master=self.subframe)
#                     self.c2.get_tk_widget().grid(column=1, row=1, sticky=(N, W))
                    
#                     self.info2 = ttk.Label(self.subframe, text="--", font=("Courier", 9))
#                     self.info2.grid(column=1, row=2, sticky=(N, W))
#                 else:
#                     self.c = Canvas(self.subframe, width=120, height=120, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
#                     self.c.grid(column=0, row=0, sticky=(N, W))
#                     self.c.xview_moveto(5)  # ref highlightthickness
#                     self.c.yview_moveto(5)  # ref highlightthickness
                    
#                     self.c2 = Canvas(self.subframe, width=120, height=120, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
#                     self.c2.grid(column=1, row=0, sticky=(N, W))
#                     self.c2.xview_moveto(5)  # ref highlightthickness
#                     self.c2.yview_moveto(5)  # ref highlightthickness
                    
#                     self.info = ttk.Label(self.frm, text="--", font=("Courier", 9))
#                     self.info.grid(column=0, row=2, sticky=(N, W))

        
#         self.gaze_cam_frames: dict[CameraIndex, UIGazeCamFrame] = {}
#         for i, cam_label in enumerate(AppConstants.GazeConstants.CAMERA_LABEL_ORDER):
#             self.gaze_cam_frames[cam_label] = UIGazeCamFrame(self.gaze_vis_frame, cam_label, i)

#         # --- Gaze Estimation Controls
#         self.gaze_ctrl_frame = ttk.LabelFrame(self.frame_gaze_estimation, text="Controls", padding=(5,5))
#         self.gaze_ctrl_frame.grid(column=1, row=0, sticky=(N, W, E, S))

#         ttk.Label(self.gaze_ctrl_frame, text="Experiment Path:").grid(column=0, row=0, sticky=(N, W))
#         self.sv_gaze_experiment_path = StringVar()
#         self.sv_gaze_experiment_path.set(AppConstants.GazeConstants.DEFAULT_EXPERIMENT_NAME)
#         ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_experiment_path, width=40).grid(column=1, row=0, sticky=(N, W))

#         ttk.Label(self.gaze_ctrl_frame, text="Calibration Path:").grid(column=0, row=1, sticky=(N, W))
#         self.sv_gaze_calib_path = StringVar()
#         self.sv_gaze_calib_path.set(AppConstants.GazeConstants.DEFAULT_CALIBRATION_NAME)
#         ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_calib_path, width=40).grid(column=1, row=1, sticky=(N, W))
        
#         ttk.Label(self.gaze_ctrl_frame, text="View:").grid(column=0, row=2, sticky=(N, W, E, S))
#         self.sv_gaze_view_mode = StringVar()
#         self.gaze_view_mode_optionmenu = ttk.OptionMenu(self.gaze_ctrl_frame, self.sv_gaze_view_mode, self.gaze_view_modes[0], *self.gaze_view_modes)
#         self.gaze_view_mode_optionmenu.grid(column=1, row=2, sticky=W)
#         self.sv_gaze_view_mode.trace_add("write", self.display_current_gaze_frame)

#         ttk.Button(self.gaze_ctrl_frame, text="Load", command=self.load_gaze_paths).grid(column=0, row=3, sticky=W)
#         ttk.Button(self.gaze_ctrl_frame, text="Run Segmentation", command=self.start_segmentation).grid(column=0, row=4, sticky=W)
#         ttk.Button(self.gaze_ctrl_frame, text="Run Gaze Estimation", command=self.start_gaze_estimation).grid(column=0, row=5, sticky=W)
        
#         ttk.Label(self.gaze_ctrl_frame, text="Until Frame").grid(column=0, row=6, sticky=(N, W))
#         self.sv_gaze_until_frame_id = StringVar()
#         self.sv_gaze_until_frame_id.set("-1")
#         ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_until_frame_id, width=8).grid(column=1, row=6, sticky=(N, W))
        
#         ttk.Label(self.gaze_ctrl_frame, text="alpha").grid(column=0, row=7, sticky=(N, W))
#         self.sv_gaze_alpha = StringVar()
#         self.sv_gaze_alpha.set("1.0")
#         ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_alpha, width=8).grid(column=1, row=7, sticky=(N, W))

#         ttk.Label(self.gaze_ctrl_frame, text="beta").grid(column=0, row=8, sticky=(N, W))
#         self.sv_gaze_beta = StringVar()
#         self.sv_gaze_beta.set("0.0")
#         ttk.Entry(self.gaze_ctrl_frame, textvariable=self.sv_gaze_beta, width=8).grid(column=1, row=8, sticky=(N, W))

#         self.label_gaze_status = ttk.Label(self.gaze_ctrl_frame, text="Status: Idle", font=("Courier", 10))
#         self.label_gaze_status.grid(column=0, row=10, columnspan=2, sticky=(N, W), pady=(8,0))

#         # Frame navigation and animation
#         nav_wrap = ttk.Frame(self.gaze_ctrl_frame)
#         nav_wrap.grid(column=0, row=9, columnspan=2, pady=(8,0), sticky=(W))

#         ttk.Button(nav_wrap, text="<", width=3, command=self.goto_frame_prev).grid(column=0, row=0, padx=(0,4))
#         self.sv_gaze_frame_id = StringVar()
#         self.sv_gaze_frame_id.set("0")
#         ttk.Entry(nav_wrap, textvariable=self.sv_gaze_frame_id, width=8).grid(column=1, row=0)
#         self.sv_gaze_frame_id.trace_add("write", self.onchange_gaze_frame_id)
#         ttk.Button(nav_wrap, text=">", width=3, command=self.goto_frame_next).grid(column=2, row=0, padx=(4,6))

#         self.btn_animate_gaze = ttk.Button(nav_wrap, text="Animate", command=self.toggle_animate_gaze)
#         self.btn_animate_gaze.grid(column=3, row=0)
        
    
#     def load_gaze_paths(self):
#         """Load gaze experiment frames from the experiment path entry."""
        
#         exp_path = Path(self.sv_gaze_experiment_path.get())
#         cam_subfolders = self.camera_settings.keys()  # Assuming camera labels correspond to subfolder names in the experiment path, e.g. "ro", "ri", "lo", "li", "sc".
#         self.gaze_imgs = {cam_label: {} for cam_label in cam_subfolders}
#         self.gaze_max_frame_idxs = {cam_label: 0 for cam_label in cam_subfolders}
        
#         gaze_view_mode = self.sv_gaze_view_mode.get()
        
#         for cam_label in cam_subfolders:
#             cam_path = exp_path / f"{cam_label}{self.gaze_view_modes_path_suffixes['Input']}"
#             if not (cam_path.exists() and cam_path.is_dir()):
#                 print(f"[Gaze] Input camera path not found for {cam_label}: {cam_path}.")
#                 continue

#             img_names = [p.name for p in cam_path.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
#             if not img_names:
#                 print(f"[Gaze] No image files found in {cam_path} for camera {cam_label}.")
#                 continue
            
#             def get_frame_metadata_from_filename(filename):
#                 match = re.search(r'frame_(\d+)_timestamp_[\d\.]+', filename)
#                 if match:
#                     frame_idx = int(match.group(1))
#                     frame_timestamp = float(match.group(0).split('_timestamp_')[1].split('.png')[0][:-1])
#                     return frame_idx, frame_timestamp
#                 else:
#                     return float('inf'), float('inf')  # If no match, place at the end

#             # Sort img_names by frame index extracted from filename
#             img_names.sort(key=lambda x: get_frame_metadata_from_filename(x)[0])

#             img_names_aligned_by_frame_idx = {}
#             img_names_aligned_by_timestamp = {}
#             # Parse frame index using regex. Pattern: frame_0_timestamp_0.000.png
#             for img_name in img_names:
#                 frame_idx, frame_timestamp = get_frame_metadata_from_filename(img_name)
#                 if frame_idx != float('inf'):
#                     img_names_aligned_by_frame_idx[frame_idx] = {"filename": img_name, "timestamp": frame_timestamp}
#                     # Create bins matching framerate of 44.5fps (~22.47ms per frame)
#                     base_framerate = 44.5  # TODO
#                     bin_idx = int((frame_timestamp + .5*(1/base_framerate)) // (1/base_framerate))
#                     if bin_idx not in img_names_aligned_by_timestamp.keys():
#                         img_names_aligned_by_timestamp[bin_idx] = {"filename": img_name, "frame_idx": frame_idx}  # IF support for multiple images per bin is needed, change to list
#                     else:
#                         pass
#                         #print(f"[Gaze] Warning: Multiple images found for timestamp bin {bin_idx} in camera {cam_label}. Using the first one.")

#             curr_highest_idx = max(img_names_aligned_by_timestamp.keys())
#             self.gaze_max_frame_idxs[cam_label] = curr_highest_idx

#             self.gaze_imgs[cam_label] = {
#                 "cam_path": cam_path,
#                 "segmentation_path": exp_path / f"{cam_label}{GazeViewModes.SEGMENTATION.value.PATH_SUFFIX}",
#                 "gaze_estimation_path": exp_path / f"{cam_label}{GazeViewModes.GAZE_ESTIMATION.value.PATH_SUFFIX}",
#                 "img_names": img_names,
#                 "img_names_aligned_by_frame_idx": img_names_aligned_by_frame_idx,
#                 "img_names_aligned_by_timestamp": img_names_aligned_by_timestamp
#             }
#             print(f"[Gaze] Loaded {len(img_names)} frames for camera {cam_label} from {cam_path}")
            
#         # fill all missing frame indices with None to ensure continuous indexing
#         for cam_label in self.gaze_imgs.keys():
#             img_names_aligned_by_frame_idx = self.gaze_imgs[cam_label]["img_names_aligned_by_frame_idx"]
#             if img_names_aligned_by_frame_idx:
#                 min_frame_idx = min(img_names_aligned_by_frame_idx.keys())
#                 max_frame_idx_cam = max(img_names_aligned_by_frame_idx.keys())
#                 for idx in range(min_frame_idx, max_frame_idx_cam + 1):
#                     if idx not in img_names_aligned_by_frame_idx:
#                         img_names_aligned_by_frame_idx[idx] = None
        
#         self.gaze_current_frame_idx = 0
#         self.sv_gaze_frame_id.set('0')  # Side effect: triggers onchange event to display first frame

    
#     def start_segmentation(self):
#         self.label_gaze_status.config(text="Status: Starting Segmentation...")
        
#         until_frame_id = int(self.sv_gaze_until_frame_id.get())
#         experiment_path = Path(self.sv_gaze_experiment_path.get())
#         cam_keys = [key for key, value in self.camera_settings.items() if value["eye_cam"]]
#         output_path_suffix = self.gaze_view_modes_path_suffixes['Segmentation']
        
#         calib_results = self.load_calibration_results(Path(self.sv_gaze_calib_path.get()))
#         if calib_results is None:
#             camera_tilt_angles_deg = {cam_key: 0 for cam_key in cam_keys}
#             camera_matrices = {cam_key: None for cam_key in cam_keys}
#             distortion_coeffs = {cam_key: None for cam_key in cam_keys}
#         else:
#             camera_tilt_angles_deg = {cam_key: self.estimate_camera_tilt_angle(calib_results, cam_key) for cam_key in cam_keys}
#             camera_matrices = {cam_key: np.array(calib_results["intrinsics"][cam_key]["K"]) for cam_key in cam_keys}
#             distortion_coeffs = {cam_key: np.array(calib_results["intrinsics"][cam_key]["dist"]) for cam_key in cam_keys}
        
#         params = {
#             "alpha": float(self.sv_gaze_alpha.get()),
#             "beta": float(self.sv_gaze_beta.get()),
#             "crop_ratio": 0.5,
#             "camera_tilt_angles_deg": camera_tilt_angles_deg,
#             "camera_matrices": camera_matrices,
#             "distortion_coeffs": distortion_coeffs
#         }

#         self.state.gaze_thread_handler["thread_status"] = "starting"
#         self.state.gaze_thread_handler["status_q"] = multiprocessing.Queue()  # Reset the queue for new segmentation run
#         self.state.gaze_thread_handler["thread"] = threading.Thread(target=self.run_segmentation, args=(self.state.gaze_thread_handler, self.gaze_imgs.copy(), until_frame_id, cam_keys, experiment_path, output_path_suffix, params))
#         self.state.gaze_thread_handler["thread"].start()


#     def start_gaze_estimation(self):
#         self.label_gaze_status.config(text="Status: Starting Gaze Estimation...")
        
#         until_frame_id = int(self.sv_gaze_until_frame_id.get())
#         experiment_path = Path(self.sv_gaze_experiment_path.get())
#         calib_root_path = Path(self.sv_gaze_calib_path.get())
#         output_path_suffix = self.gaze_estimation_output_filename
#         gaze_max_frame_idxs = self.gaze_max_frame_idxs
#         cam_keys = [key for key, value in self.camera_settings.items() if value["eye_cam"]]

#         _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration", root_path=calib_root_path)
#         if calib_pkl_path.exists():
#             calib_results = self.load_pkl(calib_pkl_path)

#         self.gaze_thread_handler["thread_status"] = "starting"
#         self.gaze_thread_handler["status_q"] = multiprocessing.Queue()  # Reset the queue for new segmentation run
#         self.gaze_thread_handler["thread"] = threading.Thread(target=self.run_gaze_estimation, args=(self.gaze_thread_handler, self.gaze_imgs.copy(), until_frame_id, gaze_max_frame_idxs.copy(), cam_keys, experiment_path, output_path_suffix, calib_results))

#         self.gaze_thread_handler["thread"].start()
    
    
#     def load_calibration_results(self, calib_root_path):
#         _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration", root_path=calib_root_path)
#         if calib_pkl_path.exists():
#             calib_results = self.load_pkl(calib_pkl_path)
#             return calib_results
#         else:
#             print(f"[Gaze] Calibration results file not found: {calib_pkl_path}")
#             return None

#     # alpha 3 beta -100
#     def display_current_gaze_frame(self, *args):
#         """Render a simple placeholder visualization for the current gaze frame into the canvases."""
#         idx = int(self.sv_gaze_frame_id.get())
#         gaze_view_mode = self.sv_gaze_view_mode.get()
        
#         vis_data = {}
        
#         calib_results = self.load_calibration_results(Path(self.sv_gaze_calib_path.get()))
#         if calib_results is None:
#             self.canvas_gaze_result.draw()
#             return

#         for cam_key in self.gaze_cam_frames.keys():
#             if cam_key in self.gaze_imgs.keys() and idx in self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"].keys() and self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][idx] is not None:
#                 try:
#                     # Load and display image
#                     fname = self.gaze_imgs[cam_key]["img_names_aligned_by_timestamp"][idx]["filename"]
#                     img_path = self.gaze_imgs[cam_key]["cam_path"] / fname
#                     imgdata = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                     imgdata = cv2.undistort(imgdata, cameraMatrix=calib_results["intrinsics"][cam_key]["K"], distCoeffs=calib_results["intrinsics"][cam_key]["dist"])
                    
#                     if cam_key != 'sc':  #  Don't run segmentation on scene camera
#                         imgdata2 = np.zeros_like(imgdata)
#                         if gaze_view_mode == 'Segmentation':

#                             cam_tilt_angle_deg = self.estimate_camera_tilt_angle(calib_results, cam_key)

#                             try:
#                                 res, res_visualization = self.detect_pupil_ellseg(imgdata, float(self.sv_gaze_alpha.get()), float(self.sv_gaze_beta.get()), .5, cam_tilt_angle_deg)
                                
#                                 imgdata = res["image"]
#                                 imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)  # TODO appears to operate in place
#                                 imgdata = cv2.ellipse(imgdata, (int(res["center"][0]), int(res["center"][1])), (int(res["axes"][0]), int(res["axes"][1])), res["angle_deg"], 0, 360, (0, 0, 255), 1)
                                
#                                 imgdata2 = res_visualization["image"]
#                                 imgdata2 = cv2.cvtColor(imgdata2, cv2.COLOR_GRAY2RGB)
#                                 imgdata2 = cv2.ellipse(imgdata2, (int(res_visualization["center"][0]), int(res_visualization["center"][1])), (int(res_visualization["axes"][0]), int(res_visualization["axes"][1])), res_visualization["angle_deg"], 0, 360, (0, 0, 255), 1)
#                             except Exception as e:
#                                 print(f"[Gaze] Pupil detection error for {cam_key} frame {idx}: {e}")
#                                 imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
#                                 imgdata = cv2.putText(imgdata, "Segmentation Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#                                 imgdata2 = imgdata.copy()
                        
#                         elif gaze_view_mode == 'Gaze Estimation':
#                             imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)  # Convert to RGB for visualization if not done before
                        
#                         if gaze_view_mode in ['Segmentation', 'Gaze Estimation']:                            
#                             # load json data for segmentation overlay
#                             segmentation_path = self.gaze_imgs[cam_key]["segmentation_path"] / fname
#                             segmentation_path = segmentation_path.with_suffix(".json")
#                             if segmentation_path.exists():
#                                 with open(segmentation_path, "r") as f:
#                                     seg_data = json.load(f)
#                                     vis_data[cam_key] = seg_data
#                                 imgdata = cv2.ellipse(imgdata, (int(seg_data["center"][0]), int(seg_data["center"][1])), (int(seg_data["axes"][0]), int(seg_data["axes"][1])), seg_data["angle_deg"], 0, 360, (0, 255, 0), 1)
#                     else:
#                         imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
                    
#                     self.gaze_cam_frames[cam_key]["img_store_raw"] = imgdata.copy()  # Store raw image
                    
#                     if cam_key != 'sc':  # scene camera gets larger preview
#                         imgdata = self.resize_and_pad_image(imgdata, 120, 120)
#                         imgdata2 = self.resize_and_pad_image(imgdata2, 120, 120)

#                         imgdata2 = cv2.imencode(".png", imgdata2)[1].tobytes()
#                         img_tk2 = PhotoImage(data=imgdata2)
#                         self.gaze_cam_frames[cam_key]["img_store2"] = img_tk2  # prevent garbage collection
#                         self.gaze_cam_frames[cam_key]["canvas2"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames[cam_key]["img_store2"])
#                     else:
#                         imgdata = self.resize_and_pad_image(imgdata, 240, 240)
                    
#                     imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
#                     img_tk = PhotoImage(data=imgdata)
#                     self.gaze_cam_frames[cam_key]["img_store"] = img_tk  # prevent garbage collection
#                     self.gaze_cam_frames[cam_key]["canvas"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames[cam_key]["img_store"])

#                     print(f"[Gaze] Displayed image for {cam_key} frame {idx}: {img_path}")
#                 except Exception as e:
#                     print(f"[Gaze] Error loading image for {cam_key} frame {idx}: {e}")
#                     self.gaze_cam_frames[cam_key]["canvas"].delete("all")
#                     self.gaze_cam_frames[cam_key]["canvas"].create_rectangle(2, 2, 118, 118, outline='red')
#                     self.gaze_cam_frames[cam_key]["canvas"].create_text(60, 60, text="Error", fill='red', font=("Courier", 10), anchor='center')
#             else:
#                 fname = "N/A"
#                 self.gaze_cam_frames[cam_key]["canvas"].delete("all")
#                 self.gaze_cam_frames[cam_key]["canvas"].create_rectangle(2, 2, 118, 118, outline='white')
#                 self.gaze_cam_frames[cam_key]["canvas"].create_text(60, 60, text="N/A", fill='white', font=("Courier", 10), anchor='center')
            
#             self.gaze_cam_frames[cam_key]["info"].config(text=f"{cam_key.upper()}: {fname}\nFrame idx: {idx}\nMax idx: {self.gaze_max_frame_idxs[cam_key]}")
        
        
#         if gaze_view_mode == 'Gaze Estimation':
            
#             fig = self.canvas_gaze_result.figure
#             fig.clf()
#             ax = fig.add_subplot(111, projection='3d')
            
#             imgdata = self.gaze_cam_frames["sc"]["img_store_raw"].copy()
#             img_dims = imgdata.shape
#             sc_distortion = calib_results["intrinsics"]["sc"]["dist"]
#             sc_K = calib_results["intrinsics"]["sc"]["K"]
#             imgdata = cv2.undistort(imgdata, cameraMatrix=sc_K, distCoeffs=sc_distortion)
            
#             # Load gaze estimation json results for the current frame
#             gaze_estimation_path = Path(self.sv_gaze_experiment_path.get()) / self.gaze_estimation_output_filename
#             if gaze_estimation_path.exists():
#                 gaze_results_data = self.load_pkl(gaze_estimation_path)
#                 fit_results = gaze_results_data.get("fit_results", {})
            
#                 for eye in ["left", "right"]:
#                     if fit_results[eye] is not None:
#                         C_eye = fit_results[eye]["center"]
#                         R_eye = fit_results[eye]["radius"]
#                         u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#                         x = C_eye[0] + R_eye * np.cos(u) * np.sin(v)
#                         y = C_eye[1] + R_eye * np.sin(u) * np.sin(v)
#                         z = C_eye[2] + R_eye * np.cos(v)
#                         ax.plot_wireframe(x, y, z, color='k', alpha=0.3)
                        
#                         # Plot optical axes
#                         if idx in fit_results[eye]["optical_dirs"] and idx in fit_results[eye]["origins"]:
#                             origin = fit_results[eye]["origins"][idx]
#                             optical_dir = fit_results[eye]["optical_dirs"][idx]
#                             ax.quiver(origin[0], origin[1], origin[2], optical_dir[0], optical_dir[1], optical_dir[2], length=30.0, color='r' if eye == 'left' else 'b', alpha=0.5)
                            
#                             if "sc" in self.camera_settings.keys() and "sc" in calib_results["extrinsics"]:
#                                 print(optical_dir)
#                                 optical_dir_from_sc = calib_results["extrinsics"]["sc"]["origin"] + optical_dir*1000
#                                 optical_dir_homog = np.append(optical_dir_from_sc, 1)  # Convert to homogeneous coordinates
#                                 sc_T = calib_results["extrinsics"]["sc"]["T"]
#                                 sc_M = sc_K @ sc_T
#                                 optical_dir_proj = sc_M @ optical_dir_homog
#                                 optical_dir_proj /= optical_dir_proj[2]  # Normalize to get pixel coordinates
#                                 optical_dir_proj = optical_dir_proj[:2]# + (np.array(img_dims[:2]) / 2)  # Get x, y pixel coordinates
#                                 print(f"[Gaze] Projected optical direction into scene camera image plane: {eye} {optical_dir_proj}")
#                                 optical_dir_proj[0] = np.clip(optical_dir_proj[0], 0, img_dims[1] - 1)
#                                 optical_dir_proj[1] = np.clip(optical_dir_proj[1], 0, img_dims[0] - 1)
#                                 if eye == "left":
#                                     color = (0, 0, 255)  # Red for left eye
#                                 else:
#                                     color = (255, 0, 0)  # Blue for right eye
#                                 imgdata = cv2.circle(imgdata, (int(optical_dir_proj[0]), int(optical_dir_proj[1])), 10, color, -1)

#             else:
#                 print(f"[Gaze] Gaze estimation results file not found: {gaze_estimation_path}")
                
#             # Update canvas with scene camera image and projected optical axes
#             imgdata = self.resize_and_pad_image(imgdata, 240, 240)
#             imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
#             img_tk = PhotoImage(data=imgdata)
#             self.gaze_cam_frames["sc"]["img_store"] = img_tk  # prevent garbage collection
#             self.gaze_cam_frames["sc"]["canvas"].create_image(0, 0, anchor='nw', image=self.gaze_cam_frames["sc"]["img_store"])

#             # plot full extrinsics from calibration results
#             for cam_key in self.camera_settings.keys():
#                 if cam_key in calib_results["extrinsics"]:
#                     extrinsics = calib_results["extrinsics"][cam_key]
#                     ax.quiver(extrinsics["origin"][0],
#                             extrinsics["origin"][1],
#                             extrinsics["origin"][2],
#                             extrinsics["z"][0],
#                             extrinsics["z"][1],
#                             extrinsics["z"][2],
#                             length=10, color='b', arrow_length_ratio=0.1)
#                     ax.quiver(extrinsics["origin"][0],
#                             extrinsics["origin"][1],
#                             extrinsics["origin"][2],
#                             extrinsics["y"][0],
#                             extrinsics["y"][1],
#                             extrinsics["y"][2],
#                             length=5, color='g', arrow_length_ratio=0.1)
                        
#             ax.set_xlabel('X [mm]')
#             ax.set_ylabel('Y [mm]')
#             ax.set_zlabel('Z [mm]')
#             ax.set_title('Fitted Eye Spheres and Optical Axes')
#             ax.set_aspect('equal')
#             self.canvas_gaze_result.draw()
            
#         elif gaze_view_mode == 'Segmentation':
#             # Load full calibration results for gaze estimation visualization
#             fig = self.canvas_gaze_result.figure
#             fig.clf()  # Clear the figure to avoid overlapping plots
            
#             _, calib_pkl_path = self.get_calibration_output_pkl_path("Full Calibration")
#             if calib_pkl_path.exists():
#                 calib_results = self.load_pkl(calib_pkl_path)
                
#                 cam_key_pairs = [("ro", "ri"), ("lo", "li")]
#                 triang_calib_data = [calib_results["calibration_steps"]["Stereo R-R"]["data"], calib_results["calibration_steps"]["Stereo L-L"]["data"]]
#                 subplot_idxs = [211, 212]

#                 for cam_pair, calib_data, subplot_idx in zip(cam_key_pairs, triang_calib_data, subplot_idxs):
#                     if all(cam_key in vis_data for cam_key in cam_pair):
#                         point_coords = [vis_data[cam_key]["center"] for cam_key in cam_pair]
#                         point_wc, (cam1_point_vector_wc, cam2_point_vector_wc), projection_error = triangulate_point(calib_data, point_coords)

#                         if point_wc is not None:
                            
#                             if cam_pair == ("lo", "li"):
#                                 T = transformation_matrix_from_calib_ext(calib_results["calibration_steps"]["Stereo R-L"]["data"]["camera_params_1"].extrinsic)
#                                 point_wc = T * np.matrix(np.append(point_wc, 1)).T  # Convert to homogeneous coordinates for transformation
#                                 point_wc = point_wc[:3]  # Convert back to 3D coordinates

#                             ax = fig.add_subplot(subplot_idx, projection='3d')

#                             # Triangulated point
#                             ax.scatter(point_wc[0], point_wc[1], point_wc[2], c='r', marker='o')
                            
#                             extrinsics_cam1 = calib_results["extrinsics"][cam_pair[0]]
#                             extrinsics_cam2 = calib_results["extrinsics"][cam_pair[1]]

#                             # Cameras
#                             ax.quiver(extrinsics_cam1["origin"][0],
#                                     extrinsics_cam1["origin"][1],
#                                     extrinsics_cam1["origin"][2],
#                                     extrinsics_cam1["z"][0],
#                                     extrinsics_cam1["z"][1],
#                                     extrinsics_cam1["z"][2],
#                                     length=10, color='b', arrow_length_ratio=0.1)
#                             ax.quiver(extrinsics_cam1["origin"][0],
#                                     extrinsics_cam1["origin"][1],
#                                     extrinsics_cam1["origin"][2],
#                                     extrinsics_cam1["y"][0],
#                                     extrinsics_cam1["y"][1],
#                                     extrinsics_cam1["y"][2],
#                                     length=5, color='g', arrow_length_ratio=0.1)

#                             ax.quiver(extrinsics_cam2["origin"][0],
#                                     extrinsics_cam2["origin"][1],
#                                     extrinsics_cam2["origin"][2],
#                                     extrinsics_cam2["z"][0],
#                                     extrinsics_cam2["z"][1],
#                                     extrinsics_cam2["z"][2],
#                                     length=10, color='b', arrow_length_ratio=0.1)
#                             ax.quiver(extrinsics_cam2["origin"][0],
#                                     extrinsics_cam2["origin"][1],
#                                     extrinsics_cam2["origin"][2],
#                                     extrinsics_cam2["y"][0],
#                                     extrinsics_cam2["y"][1],
#                                     extrinsics_cam2["y"][2],
#                                     length=5, color='g', arrow_length_ratio=0.1)

#                             ax.set_xlabel('X [mm]')
#                             ax.set_ylabel('Y [mm]')
#                             ax.set_zlabel('Z [mm]')
#                             ax.set_title(f'3D Point (Frame {idx}, Pair: {cam_pair}, Projection Error: {projection_error:.2f} mm)')
#                             ax.set_aspect('equal')

#                             fig.tight_layout()
#                         else:
#                             ax = fig.add_subplot(subplot_idx)
#                             ax.text(0.5, 0.5, f"Triangulation Failed for {cam_pair}", ha='center', va='center', fontsize=8, color='red')
#                             ax.axis('off')
#                     else:
#                         ax = fig.add_subplot(subplot_idx)
#                         ax.text(0.5, 0.5, f"Insufficient Data for Triangulation for {cam_pair}", ha='center', va='center', fontsize=8, color='red')
#                         ax.axis('off')
#             else:
#                 ax = fig.add_subplot(111)
#                 ax.text(0.5, 0.5, "Calibration Results Not Found", ha='center', va='center', fontsize=12, color='red')
#                 ax.axis('off')
            
#             self.canvas_gaze_result.draw()
            
#         self.gaze_info.config(text=f"Gaze: idx={idx}")
            
            
#     def onchange_gaze_frame_id(self, *args):
#         idx_str = self.sv_gaze_frame_id.get()
#         idx = int(idx_str)
#         if 0 <= idx <= max(self.gaze_max_frame_idxs.values()):
#             self.gaze_current_frame_idx = idx
#             self.display_current_gaze_frame()
#         else:
#             print(f"[Gaze] Frame index out of range: {idx}")


#     def goto_frame_prev(self):
#         self.gaze_current_frame_idx = max(0, self.gaze_current_frame_idx - 1)
#         self.sv_gaze_frame_id.set(str(self.gaze_current_frame_idx))  # Side effect: triggers onchange event to display first frame


#     def goto_frame_next(self):
#         self.gaze_current_frame_idx = min(max(self.gaze_max_frame_idxs.values()), self.gaze_current_frame_idx + 1)
#         self.sv_gaze_frame_id.set(str(self.gaze_current_frame_idx))  # Side effect: triggers onchange event to display first frame


#     def _gaze_anim_step(self):
#         # increment and schedule next
#         new_gaze_frame_idx = (self.gaze_current_frame_idx + 1) % max(self.gaze_max_frame_idxs.values())
#         self.sv_gaze_frame_id.set(str(new_gaze_frame_idx))  # Side effect: triggers onchange event to display first frame
#         # schedule next step
#         self.gaze_anim_after_id = self.root.after(int(1000/44.5), self._gaze_anim_step)


#     def toggle_animate_gaze(self):
#         if getattr(self, 'gaze_animating', False):
#             # stop
#             if getattr(self, 'gaze_anim_after_id', None) is not None:
#                 try:
#                     self.root.after_cancel(self.gaze_anim_after_id)
#                 except Exception:
#                     pass
#                 self.gaze_anim_after_id = None
#             self.gaze_animating = False
#             try:
#                 self.btn_animate_gaze.config(text='Animate')
#             except Exception:
#                 pass
#         else:
#             # start
#             self.gaze_animating = True
#             try:
#                 self.btn_animate_gaze.config(text='Stop')
#             except Exception:
#                 pass
#             self.gaze_anim_after_id = self.root.after(200, self._gaze_anim_step)


    # def update_gaze_ui(self):
    #     # Fetch Queue and update status text
    #     if not self.gaze_thread_handler["status_q"].empty():
    #         # Get last item from queue and clear the rest
    #         while not self.gaze_thread_handler["status_q"].empty():
    #             status_text = self.gaze_thread_handler["status_q"].get()
    #         self.label_gaze_status.config(text=status_text)