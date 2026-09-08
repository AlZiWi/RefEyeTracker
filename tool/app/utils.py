import json
import os
from pathlib import Path
import pickle
from typing import Union

import cv2
import numpy as np

from app.data_structures import CAPTURE_PRESET_CONFIGS, CameraIndex, UIPreset
from app.stereo_calib_opencv import detect_corners


class FolderManager:
    root_path: Path = None
    
    
    def __init__(self, root_path: Path | None = None):
        self.root_path = root_path
    
    
    def set_root_path(self, root_path: Path):
        self.root_path = root_path
    
    
    def get_root_path(self) -> Path:
        return self.root_path


    def get_full_path(self, relative_path: str) -> Path:
        return self.root_path / relative_path
    
    
    def get_path_for_step(self, step: UIPreset) -> Path:
        step_path = self.root_path / CAPTURE_PRESET_CONFIGS[step].folder_path
        return step_path
    
    
    def get_path_for_step_and_camera(self, step: UIPreset, camera_index: CameraIndex) -> Path:
        step_path = self.root_path / CAPTURE_PRESET_CONFIGS[step].folder_path
        camera_path = step_path / camera_index.value
        return camera_path


    def load_pkl(self, file_path: Path):
        file_path = file_path.with_suffix('.pkl')
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'rb') as handle:
            try:
                data = pickle.load(handle)
                return data
            except Exception as e:
                print(f"Error occurred while loading {file_path}: {e}")
                raise

    
    def load_json(self, file_path: Path):
        file_path = file_path.with_suffix('.json')
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r') as handle:
            data = json.load(handle)
            return data

    
    def save_pkl(self, data, file_path: Path):
        file_path = file_path.with_suffix('.pkl')
        
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(file_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def save_json(self, data, file_path: Path):
        file_path = file_path.with_suffix('.json')
        
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(file_path, 'w') as handle:
            json.dump(data, handle)


    def get_img_files_in_dirs(self, dirs: list[Path], common=True) -> list[list[str]] | list[str]:
        """
        Get image files in the given directories. If common is True, return only the files that are common to all directories as a single list.
        """
        
        files_per_dir = []
        for dir in dirs:
            if os.path.exists(dir) and os.path.isdir(dir):
                files = [f for f in os.listdir(dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files_per_dir.append(files)
            else:
                print(f"Directory path {dir} does not exist or is not a directory.")
                files_per_dir.append([])

        if common:
            files_per_dir = sorted(list(set.intersection(*map(set, files_per_dir))))
            
        # TODO smart pairing here!
            
        return files_per_dir
    

class CaptureFolderManager(FolderManager):
    save_path: Path = None
    
    def __init__(self, root_path: Path | None = None, save_path: Path | None = None):
        super().__init__(root_path)
        self.save_path = save_path


    def set_save_path(self, save_path: Path):
        self.save_path = save_path


    def get_save_path(self) -> Path:
        return self.save_path
    
    
    def create_capture_folders(self, camera_indexes: list[CameraIndex]) -> Union[Path, None]:
        #create folder if it does not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

            subfolder_paths = [self.save_path / camera_indexes[cam_i].value for cam_i in range(len(camera_indexes))]
            for subfolder_path in subfolder_paths:
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)

            print(f"Created folder {self.save_path} for capture.")
        else:
            print(f"Folder {self.save_path} already exists. Not creating new folder.")


    def save_frame(self, frame: np.ndarray, frame_id: int | None, timestamp: float | str, camera_index: CameraIndex):
        subfolder_path = self.save_path / camera_index.value
        fname = ""
        
        # frame_id with leading zeros for better sorting
        if frame_id is not None:
            frame_id_str = f"{frame_id:06d}"
            fname = f"frame_{frame_id_str}"
        else:
            fname = "snapshot"
            
        if type(timestamp) == str:
            fname += f"__{timestamp}"
        else:
            fname += f"__{timestamp:.3f}"
            fname = fname.replace(".", "-")  # replace dot with dashes for better sortings
        
        file_path = subfolder_path / fname
        file_path = file_path.with_suffix('.png')
        
        print(f"Saving frame to {file_path}")
        cv2.imwrite(str(file_path), frame)
    

def resize_and_pad_image(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
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


def detect_and_visualize_corners(frame: np.ndarray, pattern_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    corners = detect_corners(frame, pattern_size)  # cols, rows
    corners_detected = corners is not None

    if corners_detected:
        imgdata = frame.copy()
        if len(imgdata.shape) == 2:
            imgdata = cv2.cvtColor(imgdata, cv2.COLOR_GRAY2RGB)
        for idx, corner in enumerate(corners):
            color = (0, 255, 0) if idx < 2 else (255, 0, 0)
            if corner is not None:
                imgdata = cv2.circle(imgdata, (int(corner[0]), int(corner[1])), 3, color, -1)
    else:
        imgdata = frame.copy()

    return imgdata, corners
