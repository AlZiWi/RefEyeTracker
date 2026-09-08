from dataclasses import dataclass, field
import os
from pathlib import Path
from tkinter import HORIZONTAL, SINGLE, N, W, S, E, Canvas, Listbox, PhotoImage, StringVar, ttk

from enum import Enum

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from app.pair_images import pair_stereo_images_smart
from app.data_structures import CAPTURE_PRESET_CONFIGS, CalibrationLabel, CalibrationSummary, CameraCoordinateFrame, CameraParamsIntrinsic, LabelCoordinates, MirrorCalibrationResultsForType, MonoReprojectionErrors, UIPreset, UIPresetConfig
from app.calibration_utils import run_mirror_points_calibration, run_mono_calibration, run_scene_camera_extrinsics_calibration, run_stereo_calibration
from app.utils import FolderManager, detect_and_visualize_corners


plt.rcParams["font.size"] = 5


SV_CHECKBOX_ACTIVE = "active"
SV_CHECKBOX_INACTIVE = "inactive"


PRESET_ORDER: list[UIPreset] = [UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.STEREO_RL, UIPreset.MONO_SC, UIPreset.MIRROR_R, UIPreset.MIRROR_L, UIPreset.MIRROR_SC, UIPreset.FULL_CALIBRATION]
DEFAULT_PRESET: UIPreset = PRESET_ORDER[0]
PIXEL_PITCH_UM: np.array = np.array([2700/240.0, 2700/240.0])  # x, y
PIXEL_PITCH_SC_UM: np.array = np.array([3, 3])  # OV9281
OUTPUT_DIRECTORY: str = "cal"  # Appended to the calibration step folder path, e.g. "calibration/stereo_ro_ri/cal"
OUTPUT_FILENAME: str = "calibration_step_results"  # without extension, .pkl or .json will be added automatically
SUMMARY_FILENAME: str = "calibration_summary"  # without extension, .pkl or .json will be added automatically
LABELS_SUFFIX: str = "_labels"

DEFAULT_EXPERIMENT_NAME: str = "recordings/calibration_bst"


class CalibrationSMState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class CalibrationFolderManager(FolderManager):
    def get_calibration_output_folder_path_for_step(self, step: UIPreset) -> Path:
        output_folder_path = self.root_path / CAPTURE_PRESET_CONFIGS[step].folder_path / OUTPUT_DIRECTORY
        return output_folder_path
    
    
    def get_calibration_output_file_path_for_step(self, step: UIPreset) -> Path:
        output_file_path = self.get_calibration_output_folder_path_for_step(step) / OUTPUT_FILENAME
        return output_file_path


    def get_calibration_summary_output_path(self, step: UIPreset) -> Path:
        output_file_path = self.get_path_for_step(UIPreset.FULL_CALIBRATION) / OUTPUT_DIRECTORY / SUMMARY_FILENAME
        return output_file_path


    def get_calibration_input_paths(self, step):
        paths_cams = [self.get_path_for_step_and_camera(step, camera_idc) for camera_idc in CAPTURE_PRESET_CONFIGS[step].camera_indices]
        return paths_cams
    
    
    def get_label_coordinates_path(self, step: UIPreset, img_filename: str) -> Path:
        save_path = self.get_calibration_output_file_path_for_step(step) / (Path(img_filename).stem + LABELS_SUFFIX)
        return save_path


@dataclass
class UIAppCalibrationState:
    calibration_status: CalibrationSMState = CalibrationSMState.IDLE
    selected_calibration_label: CalibrationLabel | None = None
    folder_manager: CalibrationFolderManager = field(default_factory=CalibrationFolderManager)
    calibration_summary: CalibrationSummary = field(default_factory=CalibrationSummary)
    current_calibration_step: UIPreset = DEFAULT_PRESET
    
    available_label_img_filenames: list[str] = field(default_factory=list)
    selected_label_img_id: int | None = None
    label_coordinates: LabelCoordinates = field(default_factory=dict)


class UITabCalibration:
    state: UIAppCalibrationState
                    
    def __init__(self, parent_frame: ttk.Frame):
        self.state = UIAppCalibrationState()
        
        self.calibration_mainframe = ttk.LabelFrame(parent_frame, text="Calibration Workflow", padding=(5,5))
        self.calibration_mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        # --- Calibration Configuration

        self.calib_config_frame = ttk.LabelFrame(self.calibration_mainframe, text="Calibration Configuration", padding=(5,5))
        self.calib_config_frame.grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Label(self.calib_config_frame, text="Calibration Preset:").grid(column=0, row=0, sticky=(N, W, E, S))
        self.sv_calib_preset = StringVar()
        self.calib_preset_optionmenu = ttk.OptionMenu(self.calib_config_frame, self.sv_calib_preset, DEFAULT_PRESET.value, *[preset.value for preset in PRESET_ORDER])
        self.calib_preset_optionmenu.grid(column=1, row=0, sticky=W)
        self.sv_calib_preset.trace_add("write", self.cb_set_calibration_preset)
        
        ttk.Label(self.calib_config_frame, text="Path").grid(column=0, row=1, sticky=(N,W))
        self.sv_experiment_name = StringVar()
        self.sv_experiment_name.set(DEFAULT_EXPERIMENT_NAME)
        ttk.Entry(self.calib_config_frame, textvariable=self.sv_experiment_name, width=50).grid(column=1, row=1, sticky=(N,W))
        ttk.Button(self.calib_config_frame, text="Load Folder", command=self.cb_load_folder).grid(column=2, row=1, sticky=W)

        ttk.Button(self.calib_config_frame, text="Run Calibration Step", command=self.cb_run_calibration_step).grid(column=0, row=2, sticky=W)
        ttk.Button(self.calib_config_frame, text="Save Calibration", command=self.cb_save_calibration_summary).grid(column=0, row=3, sticky=W)
        ttk.Button(self.calib_config_frame, text="Reload Calibration", command=self.cb_load_calibration_summary).grid(column=0, row=4, sticky=W)
        ttk.Button(self.calib_config_frame, text="Reset Calibration", command=self.cb_reset_calibration_summary).grid(column=0, row=5, sticky=W)
        

        # --- Labeling Frames

        self.label_mainframe = ttk.LabelFrame(self.calibration_mainframe, text="Calibration Labeling", padding=(5,5))
        self.label_mainframe.grid(column=0, row=1, sticky=(N, W, E, S))
                        
        self.label_file_frame = ttk.LabelFrame(self.label_mainframe, text=f"Image and Label selection", padding=(5,5))
        self.label_file_frame.grid(column=0, row=0, sticky=(N, W))

        ## Images

        self.sv_available_label_imgs = StringVar(value=[])
        self.listbox_available_label_imgs = Listbox(self.label_file_frame, listvariable=self.sv_available_label_imgs, height=7, width=28, selectmode=SINGLE)
        self.listbox_available_label_imgs.grid(column=0, row=0, sticky=W)
        self.listbox_available_label_imgs.bind("<<ListboxSelect>>", self.cb_onclick_listbox_available_label_imgs)

        ttk.Button(self.label_file_frame, text="Save Labels", command=self.cb_save_label_coordinates).grid(column=0, row=1, sticky=W)

        ## Labels

        ttk.Separator(self.label_file_frame, orient=HORIZONTAL).grid(column=0, row=2, sticky=(E, W), pady=5)

        self.sv_available_label_names = StringVar(value=[])
        self.listbox_available_label_names = Listbox(self.label_file_frame, listvariable=self.sv_available_label_names, height=7, width=16, selectmode=SINGLE)
        self.listbox_available_label_names.grid(column=0, row=3, sticky=W)
        self.listbox_available_label_names.bind("<<ListboxSelect>>", self.cb_onclick_listbox_available_label_names)

        ttk.Label(self.label_file_frame, text=f"Click image to label").grid(column=0, row=4, sticky=(N,W))

        ## Label Frames

        class UICalibrationLabelFrame:
            def __init__(self, parent_frame: ttk.Frame, i: int, cb_reset_label_coordinates_for_frame, cb_reset_all_label_coordinates_for_frame, onclick_set_label_coordinates):
                # Variables

                self.img_store = None  # Store PhotoImage to avoid garbage collection
                self.img_store_raw = None  # Store imported raw image
                self.pattern_corners_store = None  # Store detected pattern corners for visualization

                # Frame

                self.frame = ttk.LabelFrame(parent_frame, text=f"CAM [{i+1}] Labels", padding=(5,5))
                self.frame.grid(column=i+1, row=0, sticky=(N, W))

                self.w_canvas = Canvas(self.frame, width=240, height=240, background='gray75', borderwidth=0, border=0, relief='flat', bd=0, highlightthickness=5)
                self.w_canvas.grid(column=0, row=0, sticky=(N, W), padx=0, pady=0)
                self.w_canvas.bind("<Button-1>", lambda event, idx=i: onclick_set_label_coordinates(event, idx))
                self.w_canvas.xview_moveto(5)  # ref highlightthickness
                self.w_canvas.yview_moveto(5)  # ref highlightthickness

                ttk.Button(self.frame, text="Reset Label", command=lambda idx=i: cb_reset_label_coordinates_for_frame(idx)).grid(column=0, row=1, sticky=W)
                ttk.Button(self.frame, text="Reset All Labels", command=lambda idx=i: cb_reset_all_label_coordinates_for_frame(idx)).grid(column=0, row=2, sticky=W)

                self.sv_label_coordinates = StringVar()
                ttk.Label(self.frame, textvariable=self.sv_label_coordinates, font=("Courier", 12)).grid(column=0, row=3, sticky=(N,W))

        self.label_frames = [UICalibrationLabelFrame(self.label_mainframe, i, self.cb_reset_label_coordinates_for_frame, self.cb_reset_all_label_coordinates_for_frame, self.onclick_set_label_coordinates) for i in range(2)]

        # -- Calibration Configuration

        self.frame_calib_conf = ttk.LabelFrame(parent_frame, text="Output", padding=(5,5))
        self.frame_calib_conf.grid(column=1, row=0, sticky=(N, W, E, S))

        # --- Visualization

        self.frame_calib_conf_vis = ttk.LabelFrame(self.frame_calib_conf, text="Visualization", padding=(5,5))
        self.frame_calib_conf_vis.grid(column=0, row=10, sticky=(N, W, E, S))

        self.canvas_calib_1 = FigureCanvasTkAgg(Figure(figsize=(6, 4)), master=self.frame_calib_conf_vis)
        self.canvas_calib_1.get_tk_widget().grid(column=0, row=0, sticky=(N, W, E, S))

        ttk.Separator(self.frame_calib_conf_vis, orient=HORIZONTAL).grid(column=0, row=1, sticky=(E, W), pady=5)

        self.canvas_calib_2 = FigureCanvasTkAgg(Figure(figsize=(6, 3)), master=self.frame_calib_conf_vis)
        self.canvas_calib_2.get_tk_widget().grid(column=0, row=2, sticky=(N, W, E, S))
    
    
    def init_logic(self):
        pass
    
    
    def get_current_ui_preset_config(self) -> UIPresetConfig:
        return CAPTURE_PRESET_CONFIGS[self.state.current_calibration_step]
    
    
    def reset_label_coordinates(self):
        if self.get_current_ui_preset_config().calib_settings.calib_labels is not None:
            available_labels = self.get_current_ui_preset_config().calib_settings.calib_labels.labels
            self.state.label_coordinates = {label_name: np.array([[-1, -1] for _ in range(len(self.get_current_ui_preset_config().camera_indices))]) for label_name in available_labels}
            self.state.selected_calibration_label = available_labels[0]  # Select first label type by default
            self.sv_available_label_names.set([label_name.value for label_name in available_labels]) # Update listbox
            self.listbox_available_label_names.selection_clear(0, "end")
            self.listbox_available_label_names.selection_set(0)
        else:
            # Clear available labels if none specified in preset
            self.state.label_coordinates = {}
            self.state.selected_calibration_label = None
            self.sv_available_label_names.set([])
            self.listbox_available_label_names.selection_clear(0, "end")
            self.listbox_available_label_names.selection_set(0)


    def cb_set_calibration_preset(self, *args):
        self.state.current_calibration_step = UIPreset(self.sv_calib_preset.get())
        self.state.selected_label_img_id = None
        
        for i in range(2):
            self.set_empty_label_image(i)
            
            if self.get_current_ui_preset_config().camera_indices is not None and i < len(self.get_current_ui_preset_config().camera_indices):
                camera_name = self.get_current_ui_preset_config().camera_indices[i].value
                self.label_frames[i].frame.configure(text=f"Cam [{camera_name}] Labels")
                # Re-add frame to grid in case it was removed before
                self.label_frames[i].frame.grid()
            else:
                self.label_frames[i].frame.configure(text=f"Cam [None] Labels")
                # Hide label frames for cameras not defined in preset and reset their image and label coordinates stores
                self.label_frames[i].frame.grid_remove()

        self.load_folder_for_current_step()  # Reload image to reset label coordinates visualization
        self.cb_plot_calibration_step_results()  # Visualize step results if already executed
        self.reset_label_coordinates()
    
    def cb_load_folder(self, *args):
        self.state.folder_manager.set_root_path(Path(self.sv_experiment_name.get()))
        self.load_folder_for_current_step()
        self.cb_load_calibration_summary()
        self.cb_plot_calibration_step_results()


    def load_folder_for_current_step(self):
        for i in range(2):
            self.set_empty_label_image(i)
        
        # Enumerate available images in path
        img_files = self.state.folder_manager.get_img_files_in_dirs(self.state.folder_manager.get_calibration_input_paths(self.state.current_calibration_step), common=True)
        
        if img_files == []:
            print("No image directories found for the current calibration preset.")
            self.sv_available_label_imgs.set([])  # Clear available images for cameras not defined in preset
            self.state.available_label_img_filenames = []
            self.state.selected_label_img_id = None
            return
                        
        self.sv_available_label_imgs.set(img_files)  # Clear available images for cameras not defined in preset
        self.state.available_label_img_filenames = img_files
        # select first image by default
        self.state.selected_label_img_id = 0
        self.listbox_available_label_imgs.selection_clear(0, "end")
        self.listbox_available_label_imgs.selection_set(self.state.selected_label_img_id)
        
        self.load_label_images()


    def cb_onclick_listbox_available_label_imgs(self, event):
        self.load_label_images()
        
        
    def get_selected_listbox_item(self, listbox: Listbox) -> int | None:
        idxs = listbox.curselection()
        if len(idxs) == 1:
            return idxs[0]
        else:
            return None
        
        
    def cb_onclick_listbox_available_label_names(self, event):
        idx = self.get_selected_listbox_item(self.listbox_available_label_names)
        if idx is None or idx >= len(self.state.label_coordinates.keys()):
            if self.state.selected_calibration_label is not None:
                idx = self.state.selected_calibration_label
            else:
                print("No valid label selected.")
                self.state.selected_calibration_label = None
                return
        
        self.state.selected_calibration_label = list(self.state.label_coordinates.keys())[idx]
        for ui_frame_idx in range(len(self.get_current_ui_preset_config().camera_indices)):
            self.update_label_coordinates_display(ui_frame_idx)


    def load_label_images(self):
        # TODO triggered twice
        # Update selected label image index
        img_idx = self.get_selected_listbox_item(self.listbox_available_label_imgs)
        if img_idx is None or img_idx >= len(self.state.available_label_img_filenames):
            if self.state.selected_label_img_id is not None:
                img_idx = self.state.selected_label_img_id
            else:
                print("No valid image selected. Cannot load label images.")
                for ui_frame_idx in range(len(self.get_current_ui_preset_config().camera_indices)):
                    self.set_empty_label_image(ui_frame_idx)
                self.state.selected_label_img_id = None
                return
        
        self.state.selected_label_img_id = img_idx  # Store selected image index to load it again when switching label types
        
        self.reset_label_coordinates()
        loaded_label_coordinates = self.load_label_coordinates()
        if loaded_label_coordinates is not None:
            self.state.label_coordinates = loaded_label_coordinates
        
        # Load images for each camera defined in the current calibration preset
        for ui_frame_idx, camera_index in enumerate(self.get_current_ui_preset_config().camera_indices):
            img_path = self.state.folder_manager.get_path_for_step_and_camera(self.state.current_calibration_step, camera_index) / self.state.available_label_img_filenames[img_idx]
            if os.path.exists(img_path):            
                imgdata = cv2.imread(img_path)
                #imgdata = self.resize_and_pad_image(imgdata, 240, 240)
                
                # resize canvas to image size if needed
                img_h, img_w = imgdata.shape[:2]
                if self.label_frames[ui_frame_idx].w_canvas.winfo_width() != img_w or self.label_frames[ui_frame_idx].w_canvas.winfo_height() != img_h:
                    self.label_frames[ui_frame_idx].w_canvas.config(width=img_w, height=img_h)
    
                self.label_frames[ui_frame_idx].img_store_raw = imgdata.copy()  # Store raw image
    
                if self.get_current_ui_preset_config().use_pattern is not None:
                    imgdata, corners = detect_and_visualize_corners(imgdata, self.get_current_ui_preset_config().use_pattern.corners)
                    self.label_frames[ui_frame_idx].pattern_corners_store = corners
                else:
                    self.label_frames[ui_frame_idx].pattern_corners_store = None
                    
                imgdata = cv2.imencode(".png", imgdata)[1].tobytes()
    
                self.label_frames[ui_frame_idx].img_store = PhotoImage(data=imgdata)  # Store to self to avoid issues with garbage collector
            else:
                self.set_empty_label_image(ui_frame_idx)
                print(f"Frame {ui_frame_idx+1}: Image path {img_path} does not exist.")
            
            self.update_label_coordinates_display(ui_frame_idx)
            
    
    def set_empty_label_image(self, ui_frame_idx: int):
        empty_img = np.zeros((240, 240, 3), dtype=np.uint8)
        self.label_frames[ui_frame_idx].img_store_raw = empty_img.copy()  # Store raw image
        imgdata = cv2.imencode(".png", empty_img)[1].tobytes()
        self.label_frames[ui_frame_idx].img_store = PhotoImage(data=imgdata)  # Store to self to avoid issues with garbage collector

        self.cb_reset_all_label_coordinates_for_frame(ui_frame_idx)


    def cb_reset_label_coordinates_for_frame(self, ui_frame_idx: int):
        label = self.state.selected_calibration_label
        if label is None:
            print("No label selected. Cannot reset label coordinates.")
            return

        self.reset_label_coordinates_for_frame(ui_frame_idx, label)


    def reset_label_coordinates_for_frame(self, ui_frame_idx: int, key: CalibrationLabel):
        self.state.label_coordinates[key][ui_frame_idx] = [-1,-1]
        self.update_label_coordinates_display(ui_frame_idx)


    def cb_reset_all_label_coordinates_for_frame(self, ui_frame_idx: int):
        for key in self.state.label_coordinates.keys():
            self.state.label_coordinates[key][ui_frame_idx] = [-1,-1]
        self.update_label_coordinates_display(ui_frame_idx)


    def cb_save_label_coordinates(self):
        if self.state.selected_label_img_id is None or self.state.selected_label_img_id < 0 or self.state.selected_label_img_id >= len(self.state.available_label_img_filenames):
            print("No valid label image selected. Cannot save label coordinates.")
            return
        
        path = self.state.folder_manager.get_label_coordinates_path(self.state.current_calibration_step, self.state.available_label_img_filenames[self.state.selected_label_img_id])
        self.state.folder_manager.save_pkl(self.state.label_coordinates, path)
    
    
    def load_label_coordinates(self) -> LabelCoordinates | None:  # TODO does not work when called from different step
        if self.state.selected_label_img_id is None or self.state.selected_label_img_id < 0 or self.state.selected_label_img_id >= len(self.state.available_label_img_filenames):
            print("No valid label image selected. Cannot load label coordinates.")
            return None
        
        path = self.state.folder_manager.get_label_coordinates_path(self.state.current_calibration_step, self.state.available_label_img_filenames[self.state.selected_label_img_id])
        try:
            label_coordinates = self.state.folder_manager.load_pkl(path)
        except:
            print(f"Label coordinates file {path} does not exist or is invalid. Cannot load label coordinates.")
            label_coordinates = None
        
        return label_coordinates
    

    def update_label_coordinates_display(self, ui_frame_idx: int):
        if self.state.label_coordinates is None:
            return

        self.label_frames[ui_frame_idx].w_canvas.create_image(0, 0, image=self.label_frames[ui_frame_idx].img_store, anchor='nw')

        # Draw Edge points to visualize canvas borders (for debugging and to avoid confusion about coordinate system)
        self.label_frames[ui_frame_idx].w_canvas.create_rectangle(0, 0, 1, 1, fill='green', width=0)
        self.label_frames[ui_frame_idx].w_canvas.create_rectangle(239, 0, 240, 1, fill='green', width=0)
        self.label_frames[ui_frame_idx].w_canvas.create_rectangle(0, 239, 1, 240, fill='green', width=0)
        self.label_frames[ui_frame_idx].w_canvas.create_rectangle(239, 239, 240, 240, fill='green', width=0)

        for key, val in self.state.label_coordinates.items():
            if val[ui_frame_idx][0] >= 0 and val[ui_frame_idx][1] >= 0:
                if self.state.selected_calibration_label is not None and key == self.state.selected_calibration_label:
                    color = 'blue'
                else:
                    color = 'red'
                # Draw circle at clicked position
                self.label_frames[ui_frame_idx].w_canvas.create_oval(val[ui_frame_idx][0]-2, val[ui_frame_idx][1]-2, val[ui_frame_idx][0]+1, val[ui_frame_idx][1]+1, outline='red', width=2)  # See https://anzeljg.github.io/rin2/book2/2405/docs/tkinter/create_oval.html, really counterintuitive
                self.label_frames[ui_frame_idx].w_canvas.create_text(val[ui_frame_idx][0], val[ui_frame_idx][1]-10, text=f"{key.value}", fill='red', font=('Arial', 10, 'bold'))
        
        positions_str = "\n".join([f"{key.value}: ({val[ui_frame_idx][0]}, {val[ui_frame_idx][1]})" for key, val in self.state.label_coordinates.items()])
        positions_str += "\nPattern Detected: " + ("True" if self.label_frames[ui_frame_idx].pattern_corners_store is not None else "False")
        self.label_frames[ui_frame_idx].sv_label_coordinates.set(positions_str)

        step = self.state.current_calibration_step
        if step in [UIPreset.MIRROR_R, UIPreset.MIRROR_L]:
            self.run_calibration_step_mirror_points_live(step=step)
            self.plot_calibration_step_results(step=step)
        elif step == UIPreset.MIRROR_SC:
            self.run_calibration_step_sc_extrinsics_live(step=step)
            self.plot_calibration_step_results(step=step)


    def onclick_set_label_coordinates(self, event, frame_idx: int):
        if frame_idx >= len(self.get_current_ui_preset_config().camera_indices):
            print(f"Frame index {frame_idx} is out of bounds for current calibration preset.")
            return
        
        label = self.state.selected_calibration_label
        if label is None:
            print("No label selected. Cannot set label coordinates.")
            return
        
        x, y = event.widget.canvasx(event.x), event.widget.canvasy(event.y)
        self.state.label_coordinates[label][frame_idx] = [x, y]

        # Update display
        self.update_label_coordinates_display(frame_idx)

        # cycle dropdown to next led id
        available_labels = list(self.state.label_coordinates.keys())
        label_id = available_labels.index(label)
        next_label_id = (label_id + 1) % len(self.state.label_coordinates)
        self.state.selected_calibration_label = available_labels[next_label_id]
        self.listbox_available_label_names.selection_clear(0, "end")
        self.listbox_available_label_names.selection_set(next_label_id)
        
        
    def cb_reset_calibration_summary(self):
        self.reset_calibration_summary()
    
    
    def cb_load_calibration_summary(self):
        self.load_calibration_summary()
    
    
    def cb_save_calibration_summary(self):
        self.save_calibration_summary()
    
    
    def reset_calibration_summary(self):
        self.state.calibration_summary = CalibrationSummary()
    
        
    def load_calibration_summary(self):
        try:
            self.state.calibration_summary = self.state.folder_manager.load_pkl(self.state.folder_manager.get_calibration_summary_output_path(UIPreset.FULL_CALIBRATION))
        except:
            print("No existing calibration summary found or file is invalid.")
            self.reset_calibration_summary()
    
    
    def save_calibration_summary(self):
        self.state.folder_manager.save_pkl(self.state.calibration_summary, self.state.folder_manager.get_calibration_summary_output_path(UIPreset.FULL_CALIBRATION))
        
    
    def cb_run_calibration_step(self):
        step = self.state.current_calibration_step
        self.run_calibration_step_from_files(step=step)
    
    
    def run_calibration_step_from_files(self, step: UIPreset):
        step_order = PRESET_ORDER
        step_idx = step_order.index(step)
        for next_step in step_order[step_idx:]:
            self.run_single_calibration_step(next_step)
        
        self.plot_calibration_step_results(step=step)
    
    
    def run_single_calibration_step(self, step: UIPreset):
        self.state.calibration_status = CalibrationSMState.RUNNING
        
        if step in [UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.STEREO_RL]:
            self.run_calibration_step_stereo_calibration(step=step)
        elif step == UIPreset.MONO_SC:
            self.run_calibration_step_mono_calibration(step=step)
        elif step in [UIPreset.MIRROR_R, UIPreset.MIRROR_L]:
            self.run_calibration_step_mirror_points(step=step)
        elif step == UIPreset.MIRROR_SC:
            self.run_calibration_step_sc_extrinsics(step=step)
        
        self.state.calibration_status = CalibrationSMState.COMPLETED
    
    
    def run_calibration_step_stereo_calibration(self, step: UIPreset):
        dirs = self.state.folder_manager.get_calibration_input_paths(step)
        img_path_pairs = pair_stereo_images_smart(dirs[0], dirs[1], max_dt_ms=2.0)

        pattern_size = CAPTURE_PRESET_CONFIGS[step].use_pattern.corners
        square_size_mm = CAPTURE_PRESET_CONFIGS[step].use_pattern.square_size_mm
        pixel_pitch_mm = PIXEL_PITCH_UM / 1000.0
        
        self.state.calibration_summary = run_stereo_calibration(
            calibration_summary=self.state.calibration_summary,
            step=step,
            img_path_pairs=img_path_pairs,
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            pixel_pitch_mm=pixel_pitch_mm
        )
        
    
    def run_calibration_step_mono_calibration(self, step: UIPreset):
        dirs = self.state.folder_manager.get_calibration_input_paths(step)
        img_files = self.state.folder_manager.get_img_files_in_dirs(dirs, common=False)[0]
        img_paths = [dirs[0] / img_file for img_file in img_files]

        pattern_size = CAPTURE_PRESET_CONFIGS[step].use_pattern.corners
        square_size_mm = CAPTURE_PRESET_CONFIGS[step].use_pattern.square_size_mm
        pixel_pitch_mm = PIXEL_PITCH_UM / 1000.0
        
        self.state.calibration_summary = run_mono_calibration(
            calibration_summary=self.state.calibration_summary,
            step=step,
            img_paths=img_paths,
            pattern_size=pattern_size,
            square_size_mm=square_size_mm,
            pixel_pitch_mm=pixel_pitch_mm
        )
        
        
    def run_calibration_step_mirror_points(self, step: UIPreset, label_coordinates: LabelCoordinates | None=None):
        if label_coordinates is None:
            # If no label coordinates are provided, load them from the saved pkl file
            label_coordinates=self.load_label_coordinates()
            if label_coordinates is None:
                print("No label coordinates found for the selected image. Cannot run mirror points calibration.")
                return
        
        # Prepare points and camera label pairs for calibration
        points_label_pairs: LabelCoordinates = {}
        camera_label_pairs: LabelCoordinates = {}
        camera_label_names = CAPTURE_PRESET_CONFIGS[step].calib_settings.calib_labels.source_camera_labels
        for label_name, coords in label_coordinates.items():
            if np.all(coords >= 0):
                points_label_pairs[label_name] = coords
                if label_name in camera_label_names:
                    camera_label_pairs[label_name] = coords
        
        try:
            self.state.calibration_summary = run_mirror_points_calibration(
                calibration_summary=self.state.calibration_summary,
                step=step,
                img_pair_paths=(self.state.folder_manager.get_calibration_input_paths(step)[0] / self.state.available_label_img_filenames[self.state.selected_label_img_id],
                                self.state.folder_manager.get_calibration_input_paths(step)[1] / self.state.available_label_img_filenames[self.state.selected_label_img_id]),
                points_label_pairs=points_label_pairs,
                camera_label_pairs=camera_label_pairs
            )
        except Exception as e:
           print(f"Error during mirror points calibration for step {step}: {e}")
           return



    def run_calibration_step_mirror_points_live(self, step: UIPreset):
        self.run_calibration_step_mirror_points(step=step, label_coordinates=self.state.label_coordinates)
    
    
    def run_calibration_step_sc_extrinsics(self, step: UIPreset, label_coordinates: list[list[tuple[float, float]]] | None=None):
        """Run the scene camera extrinsics calibration step."""
        
        if self.state.selected_label_img_id is None:
            print("No label image selected. Cannot run scene camera extrinsics calibration.")
            return

        pattern_size = CAPTURE_PRESET_CONFIGS[step].use_pattern.corners
        square_size_mm = CAPTURE_PRESET_CONFIGS[step].use_pattern.square_size_mm
        img_path = self.state.folder_manager.get_calibration_input_paths(step)[0] / self.state.available_label_img_filenames[self.state.selected_label_img_id]
        
        if label_coordinates is None:
            # If no label coordinates are provided, load them from the saved pkl files
            point_labels=self.load_label_coordinates()
            if point_labels is None:
                print("No label coordinates found for the selected image. Cannot run scene camera extrinsics calibration.")
                return
        else:
            point_labels=label_coordinates
        
        self.state.calibration_summary = run_scene_camera_extrinsics_calibration(
            calibration_summary=self.state.calibration_summary,
            img_path=img_path,
            calibration_pattern_size=pattern_size,
            calibration_pattern_square_size_mm=square_size_mm,
            point_labels=point_labels
        )
    
    
    def run_calibration_step_sc_extrinsics_live(self, step: UIPreset):
        self.run_calibration_step_sc_extrinsics(label_coordinates=self.state.label_coordinates, step=step)
        
        
    def cb_plot_calibration_step_results(self):
        step = self.state.current_calibration_step
        self.plot_calibration_step_results(step=step)
    
    
    def plot_calibration_step_results(self, step: UIPreset):
        if step in [UIPreset.STEREO_RR, UIPreset.STEREO_LL, UIPreset.STEREO_RL]:
            self.plot_stereo_calibration_results(step)
        elif step == UIPreset.MONO_SC:
            self.plot_mono_calibration_results(step)
        elif step in [UIPreset.MIRROR_R, UIPreset.MIRROR_L]:
            self.plot_stereo_points_calibration_results(step)
        elif step == UIPreset.MIRROR_SC:
            self.plot_scene_camera_extrinsics_calibration_results(step)
        else:
            print(f"No visualization implemented for step {step}")
            self.set_calibration_step_visualization_empty()

        self.plot_extrinsic_calibration_results()
        
    
    def plot_stereo_calibration_results(self, step: UIPreset):
        calibration_summary = self.state.calibration_summary
        
        if step == UIPreset.STEREO_RR:
            if calibration_summary.intermediate_results.STEREO_RR is None:
                self.set_calibration_step_visualization_empty()
                return
            stereo_calibration_results = calibration_summary.intermediate_results.STEREO_RR
        elif step == UIPreset.STEREO_LL:
            if calibration_summary.intermediate_results.STEREO_LL is None:
                self.set_calibration_step_visualization_empty()
                return
            stereo_calibration_results = calibration_summary.intermediate_results.STEREO_LL
        elif step == UIPreset.STEREO_RL:
            if calibration_summary.intermediate_results.STEREO_RL is None:
                self.set_calibration_step_visualization_empty()
                return
            stereo_calibration_results = calibration_summary.intermediate_results.STEREO_RL
        else:
            print(f"Invalid step {step} for stereo calibration results visualization.")
            self.set_calibration_step_visualization_empty()
            return
        
        plt_dim = (2,4)
        
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 1, projection='3d')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title("Camera Pair Extrinsics")

        self.plot_camera_frame(ax, stereo_calibration_results.camera_params_0.extrinsic.relative, vec_length=20, color='r')
        self.plot_camera_frame(ax, stereo_calibration_results.camera_params_1.extrinsic.relative, vec_length=20, color='g')
        ax.set_aspect('equal', 'box')
        
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 2)
        ax.set_title("Mono Undistortion R")
        
        self.plot_intrinsic_distortion(ax, stereo_calibration_results.camera_params_0.intrinsic)
        
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 6)
        ax.set_title("Mono Undistortion L")

        self.plot_intrinsic_distortion(ax, stereo_calibration_results.camera_params_1.intrinsic)

        # Scatter plots for errors
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 3)
        ax.set_title("Reproj Errors R")
        
        self.plot_intrinsic_reprojection_errors(ax, stereo_calibration_results.camera_params_0.intrinsic.statistics.errs_mono_reproj)

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 4)
        ax.set_title("Ini Reproj Errors R")
        
        self.plot_intrinsic_reprojection_errors(ax, stereo_calibration_results.camera_params_0.intrinsic.statistics.errs_mono_reproj_initial)

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 7)
        ax.set_title("Reproj Errors L")
        
        self.plot_intrinsic_reprojection_errors(ax, stereo_calibration_results.camera_params_1.intrinsic.statistics.errs_mono_reproj)

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 8)
        ax.set_title("Ini Reproj Errors L")
        
        self.plot_intrinsic_reprojection_errors(ax, stereo_calibration_results.camera_params_1.intrinsic.statistics.errs_mono_reproj_initial)

        fig.tight_layout()

        self.canvas_calib_1.draw()
        
        
    def plot_mono_calibration_results(self, step: UIPreset):
        calibration_summary = self.state.calibration_summary

        if calibration_summary.intermediate_results.MONO_SC is None:
            self.set_calibration_step_visualization_empty()
            return
        
        plt_dim = (1,3)
        
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 1)
        ax.set_title("Mono Distortion")
        
        self.plot_intrinsic_distortion(ax, calibration_summary.intermediate_results.MONO_SC)
        
        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 2)
        ax.set_title("Reproj Errors")
        
        self.plot_intrinsic_reprojection_errors(ax, calibration_summary.intermediate_results.MONO_SC.statistics.errs_mono_reproj)

        ax = fig.add_subplot(plt_dim[0], plt_dim[1], 3)
        ax.set_title("Ini Reproj Errors")
        
        self.plot_intrinsic_reprojection_errors(ax, calibration_summary.intermediate_results.MONO_SC.statistics.errs_mono_reproj_initial)
        
        fig.tight_layout()

        self.canvas_calib_1.draw()
        
        
    def plot_stereo_points_calibration_results(self, step: UIPreset):
        calibration_summary = self.state.calibration_summary
        
        if step == UIPreset.MIRROR_R:
            if calibration_summary.intermediate_results.MIRROR_R is None:
                self.set_calibration_step_visualization_empty()
                return
            
            mirror_calibration_results = calibration_summary.intermediate_results.MIRROR_R
            stereo_calibration_results = calibration_summary.intermediate_results.STEREO_RR
        elif step == UIPreset.MIRROR_L:
            if calibration_summary.intermediate_results.MIRROR_L is None:
                self.set_calibration_step_visualization_empty()
                return
            
            mirror_calibration_results = calibration_summary.intermediate_results.MIRROR_L
            stereo_calibration_results = calibration_summary.intermediate_results.STEREO_LL
        else:
            print(f"Invalid step {step} for stereo mirror calibration results visualization.")
            self.set_calibration_step_visualization_empty()
            return
        
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        for idx, (mirror_type_label, mirror_calibration_result_for_type) in enumerate(mirror_calibration_results.items()):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"{mirror_type_label}")
            ax.set_aspect('equal')

            self.plot_camera_frame(ax, stereo_calibration_results.camera_params_0.extrinsic.relative, vec_length=25, color='r')
            self.plot_camera_frame(ax, stereo_calibration_results.camera_params_1.extrinsic.relative, vec_length=25, color='g')

            self.plot_points(ax, mirror_calibration_result_for_type, color_world='g', color_mirrored='b', color_cpv='c', color_surface_point='m', camera_origins=np.array([stereo_calibration_results.camera_params_0.extrinsic.relative.origin, stereo_calibration_results.camera_params_1.extrinsic.relative.origin]))
        
        fig.tight_layout()

        self.canvas_calib_1.draw()


    def plot_scene_camera_extrinsics_calibration_results(self, step: UIPreset):
        calibration_summary = self.state.calibration_summary
        
        if calibration_summary.intermediate_results.MIRROR_SC is None:
            self.set_calibration_step_visualization_empty()
            return
        
        scene_camera_extrinsics_calibration_results = calibration_summary.intermediate_results.MIRROR_SC
        mono_calibration_results = calibration_summary.intermediate_results.MONO_SC
            
        # Plot results
        gs0 = plt.GridSpec(2,2, height_ratios=[1,2])
        fig = self.canvas_calib_1.figure
        fig.clf()
        
        # Plot undistorted points
        ax = fig.add_subplot(gs0[0,0])
        ax.scatter(scene_camera_extrinsics_calibration_results.points.image_points_2d_mirror_undist[:, 0], -scene_camera_extrinsics_calibration_results.points.image_points_2d_mirror_undist[:, 1], c='r', marker='o', label='Mirror Pattern')
        ax.scatter(scene_camera_extrinsics_calibration_results.points.image_points_2d_cameras_undist[:, 0], -scene_camera_extrinsics_calibration_results.points.image_points_2d_cameras_undist[:, 1], c='g', marker='x', label='Camera Labels')
        ax.scatter(scene_camera_extrinsics_calibration_results.points.image_point_2d_sc_undist[0], -scene_camera_extrinsics_calibration_results.points.image_point_2d_sc_undist[1], c='b', marker='s', label='Scene Camera Label')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('Undistorted Points')
        ax.legend()
        # TODO Would be neat to overlay undistorted image, but cv2 undistortion changes image dimensions, so we would need to remap the image first and then overlay the points on top of it, but for now we just plot the points

        # Plot undistorted image
        ax = fig.add_subplot(gs0[0,1])
        # plot img remapped according to distortion map
        img = cv2.imread(scene_camera_extrinsics_calibration_results.img_path)
        img_undistorted = cv2.remap(img, mono_calibration_results.map1, mono_calibration_results.map2, interpolation=cv2.INTER_LINEAR)
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
        mirror_corners_world = (scene_camera_extrinsics_calibration_results.mirror_pose.T_mirror[:3, :3] @ mirror_corners.T).T + scene_camera_extrinsics_calibration_results.mirror_pose.T_mirror[:3, 3]
        ax.plot_trisurf(mirror_corners_world[:, 0], mirror_corners_world[:, 1], mirror_corners_world[:, 2], color='c', alpha=0.5, label='Mirror')
        
        # sc normal
        #ax.quiver(scene_camera_extrinsics_calibration_results.mirror_pose.line_sc_normal.point[0], line_sc_normal.point[1], line_sc_normal.point[2], line_sc_normal.direction[0], line_sc_normal.direction[1], line_sc_normal.direction[2], length=20, color='m')

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
        #ax.scatter(origin_scd_scwc[0], origin_scd_scwc[1], origin_scd_scwc[2], c='g', marker='o', label="SC'")

        # Line of possible camera positions based on the scene camera label point
        for i in range(len(scene_camera_extrinsics_calibration_results.possible_camera_positions)):
            #line_points = np.array([scene_camera_extrinsics_calibration_results.lines_possible_camera_positions_from_sc[i].point + t * scene_camera_extrinsics_calibration_results.lines_possible_camera_positions_from_sc[i].direction for t in np.linspace(0, 3, 2)])
            #ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], color='b')
            if scene_camera_extrinsics_calibration_results.possible_camera_positions[i] is not None:
                ax.scatter(scene_camera_extrinsics_calibration_results.possible_camera_positions[i][0][0], scene_camera_extrinsics_calibration_results.possible_camera_positions[i][0][1], scene_camera_extrinsics_calibration_results.possible_camera_positions[i][0][2], c='r', marker='^')
                ax.scatter(scene_camera_extrinsics_calibration_results.possible_camera_positions[i][1][0], scene_camera_extrinsics_calibration_results.possible_camera_positions[i][1][1], scene_camera_extrinsics_calibration_results.possible_camera_positions[i][1][2], c='r', marker='v')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Scene camera coordinate system')
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio
        ax.legend()

        # 3D plot of the camera label points and their directions in world coordinates
        ax = fig.add_subplot(gs0[1,1], projection='3d')

        cam_zs = np.array([
            calibration_summary.CAM_RO.extrinsic.absolute.z,
            calibration_summary.CAM_RI.extrinsic.absolute.z,
            calibration_summary.CAM_LI.extrinsic.absolute.z,
            calibration_summary.CAM_LO.extrinsic.absolute.z
        ], dtype=np.float32)
        
        cam_ys = np.array([
            calibration_summary.CAM_RO.extrinsic.absolute.y,
            calibration_summary.CAM_RI.extrinsic.absolute.y,
            calibration_summary.CAM_LI.extrinsic.absolute.y,
            calibration_summary.CAM_LO.extrinsic.absolute.y
        ], dtype=np.float32)

        ax.scatter(scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 0], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 1], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 2], c='r', marker='o', label='Cams')
        ax.quiver(scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 0], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 1], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 2], cam_zs[:, 0], cam_zs[:, 1], cam_zs[:, 2], length=20, color='g', label='Cams (Z)')
        ax.quiver(scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 0], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 1], scene_camera_extrinsics_calibration_results.points.world_points_3d_frames[:, 2], cam_ys[:, 0], cam_ys[:, 1], cam_ys[:, 2], length=10, color='b', label='Cams (Y)')
        #ax.scatter(origin_sc[0], origin_sc[1], origin_sc[2], c='m', marker='^', label='SC')
        #ax.quiver(origin_sc[0], origin_sc[1], origin_sc[2], estimated_sc_z[0], estimated_sc_z[1], estimated_sc_z[2], length=20, color='c', label='SC (Z)')
        #ax.quiver(origin_sc[0], origin_sc[1], origin_sc[2], estimated_sc_y[0], estimated_sc_y[1], estimated_sc_y[2], length=10, color='y', label='SC (Y)')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('Frame Extrinsics')
        ax.set_aspect('equal', 'box')
        ax.legend()

        fig.tight_layout()

        self.canvas_calib_1.draw_idle()
        
        
    def plot_extrinsic_calibration_results(self):
        calibration_summary = self.state.calibration_summary
        if calibration_summary is None:
            self.set_extrinsic_calibration_visualization_empty()
            return
        
        fig = self.canvas_calib_2.figure
        fig.clf()
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Full Extrinsic Calibration Results")
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')

        if calibration_summary.CAM_RO is not None:
            self.plot_camera_frame(ax, calibration_summary.CAM_RO.extrinsic.absolute, vec_length=10, color='r', label='CAM_RO')
        if calibration_summary.CAM_RI is not None:
            self.plot_camera_frame(ax, calibration_summary.CAM_RI.extrinsic.absolute, vec_length=10, color='g', label='CAM_RI')
        if calibration_summary.CAM_LI is not None:
            self.plot_camera_frame(ax, calibration_summary.CAM_LI.extrinsic.absolute, vec_length=10, color='b', label='CAM_LI')
        if calibration_summary.CAM_LO is not None:
            self.plot_camera_frame(ax, calibration_summary.CAM_LO.extrinsic.absolute, vec_length=10, color='c', label='CAM_LO')
        if calibration_summary.CAM_SC is not None:
            self.plot_camera_frame(ax, calibration_summary.CAM_SC.extrinsic.absolute, vec_length=10, color='m', label='CAM_SC')
            
        if calibration_summary.POINTS_R is not None:
            self.plot_points(ax, calibration_summary.POINTS_R, color_world='g', color_mirrored='b', color_cpv='c', color_surface_point='m', camera_origins=np.array([calibration_summary.CAM_RO.extrinsic.absolute.origin, calibration_summary.CAM_RI.extrinsic.absolute.origin]))
        if calibration_summary.POINTS_L is not None:
            self.plot_points(ax, calibration_summary.POINTS_L, color_world='g', color_mirrored='r', color_cpv='c', color_surface_point='m', camera_origins=np.array([calibration_summary.CAM_LO.extrinsic.absolute.origin, calibration_summary.CAM_LI.extrinsic.absolute.origin]))

        ax.legend()
        ax.set_aspect('equal', 'box')
        fig.tight_layout()

        self.canvas_calib_2.draw()
                    
    
    def set_calibration_step_visualization_empty(self):
        fig = self.canvas_calib_1.figure
        fig.clf()
        self.canvas_calib_1.draw()
        
        
    def set_extrinsic_calibration_visualization_empty(self):
        fig = self.canvas_calib_2.figure
        fig.clf()
        self.canvas_calib_2.draw()
    
    
    def plot_camera_frame(self, ax, camera_coordinate_frame: CameraCoordinateFrame, vec_length=10, color='r', label=None):
        ax.quiver(camera_coordinate_frame.origin[0], camera_coordinate_frame.origin[1], camera_coordinate_frame.origin[2],
                    camera_coordinate_frame.z[0], camera_coordinate_frame.z[1], camera_coordinate_frame.z[2],
                    length=vec_length, color=color, label=label)
        ax.quiver(camera_coordinate_frame.origin[0], camera_coordinate_frame.origin[1], camera_coordinate_frame.origin[2],
                    camera_coordinate_frame.x[0], camera_coordinate_frame.x[1], camera_coordinate_frame.x[2],
                    length=vec_length/4, color=color)
        ax.quiver(camera_coordinate_frame.origin[0], camera_coordinate_frame.origin[1], camera_coordinate_frame.origin[2],
                    camera_coordinate_frame.y[0], camera_coordinate_frame.y[1], camera_coordinate_frame.y[2],
                    length=vec_length/2, color=color)
        
        # Draw small camera frustum for visualization
        frustum_size = vec_length / 5
        frustum_points = np.array([
            [0, 0, 0, 1],
            [-frustum_size, -frustum_size, vec_length, 1],
            [frustum_size, -frustum_size, vec_length, 1],
            [frustum_size, frustum_size, vec_length, 1],
            [-frustum_size, frustum_size, vec_length, 1]
        ])
        frustum_points_world = (camera_coordinate_frame.T @ frustum_points.T).T
        line_thicknness = .5
        
        for i in range(1, 5):
            ax.plot([frustum_points_world[0, 0], frustum_points_world[i, 0]], [frustum_points_world[0, 1], frustum_points_world[i, 1]], [frustum_points_world[0, 2], frustum_points_world[i, 2]], color=color, linewidth=line_thicknness)
            ax.plot([frustum_points_world[i, 0], frustum_points_world[i%4+1, 0]], [frustum_points_world[i, 1], frustum_points_world[i%4+1, 1]], [frustum_points_world[i, 2], frustum_points_world[i%4+1, 2]], color=color, linewidth=line_thicknness)


    def plot_points(self, ax, mirror_calibration_results_for_type: MirrorCalibrationResultsForType, color_world: str='g', color_mirrored: str | None=None, color_cpv: str='c', color_surface_point: str | None=None, camera_origins: np.ndarray | None=None):
        if color_surface_point is not None:
            ax.scatter(mirror_calibration_results_for_type.surface_point[0], mirror_calibration_results_for_type.surface_point[1], mirror_calibration_results_for_type.surface_point[2], c=color_surface_point, marker='^', s=10)
            ax.quiver(mirror_calibration_results_for_type.surface_point[0], mirror_calibration_results_for_type.surface_point[1], mirror_calibration_results_for_type.surface_point[2],
                        mirror_calibration_results_for_type.surface_normal[0], mirror_calibration_results_for_type.surface_normal[1], mirror_calibration_results_for_type.surface_normal[2],
                        length=20, color=color_surface_point)
        
        for point_id, point_data in mirror_calibration_results_for_type.calibrated_points.items():
            point_mirrored = point_data.point_mirrored.flatten()
            point_wc = point_data.point_world.flatten()
            cam_point_vectors = point_data.cam_point_vectors

            ax.scatter(point_wc[0], point_wc[1], point_wc[2], c=color_world, marker='o', s=10)
            if color_mirrored is not None:
                ax.scatter(point_mirrored[0], point_mirrored[1], point_mirrored[2], c=color_mirrored, marker='x', s=10)
            
            # write point id next to point
            ax.text(point_wc[0], point_wc[1], point_wc[2], f"{point_id.value}", color=color_world)
            
            if camera_origins is not None:
                for i in range(2):
                    curr_cam_origin = camera_origins[i].flatten()
                    curr_dst = np.linalg.norm(point_mirrored[:3] - curr_cam_origin[:3])
                    curr_cpv = cam_point_vectors[:, i] * (curr_dst * 1.1)  # scale cpv for better visualization
    
                    ax.plot([curr_cam_origin[0], curr_cam_origin[0] + curr_cpv[0]], [curr_cam_origin[1], curr_cam_origin[1] + curr_cpv[1]], [curr_cam_origin[2], curr_cam_origin[2] + curr_cpv[2]], c=color_cpv, linestyle='--', linewidth=0.5)


    def plot_intrinsic_distortion(self, ax, camera_params_intrinsic: CameraParamsIntrinsic):
        
        def _draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
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
        
        
        img_grid = np.zeros((camera_params_intrinsic.map1.shape[0], camera_params_intrinsic.map1.shape[1], 3), np.uint8)
        img_grid = _draw_grid(img_grid, (20, 20))
        
        img_undistorted = cv2.remap(img_grid, camera_params_intrinsic.map1, camera_params_intrinsic.map2, interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_undistorted)
        ax.axis('off')
        ax.set_aspect('equal')
    
    
    def plot_intrinsic_reprojection_errors(self, ax, mono_reprojection_errors: MonoReprojectionErrors):
        errs = mono_reprojection_errors.all_errs_2d
        ax.scatter(errs[:,0], errs[:,1], s=.2, c="red")
        
        ax.grid()
        ax.set_aspect('equal')
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')