from tkinter import *
from tkinter import ttk


from app.detect_pupil_ellseg import *
from app.ui_calibration import UITabCalibration
from app.ui_capture import UITabCapture
from app.stereo_calib_opencv import *

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIVARefSysApp:
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
        # 3.6) (Optional) Refine all calibration results using bundle adjustment with all images and all labeled points as input
        
        # This is efficient enough to run as one workflow and with caching even live when changing parameters.
        # UI concept:
        # General settings
        # - pattern sizes
        # - main calibration Folder
        # Tab 1 Presets for all image acquisition steps
        # Tab 2 Presets for all labeling steps (mostly implemented)
        # Tab 2b Live visualization of results (per step) -> Drop down with cycle buttons to switch between steps, visualize reprojection errors, etc.


        @dataclass
        class UIAppState:
            app_running: bool = True
    
        self.state = UIAppState()

        # Layout
        class UIRoot:
            def __init__(self):
                # Layout
                self.tk_root = Tk()
                self.tk_root.title("VIVA Reference System Calibration Utility")
                self.tk_root.columnconfigure(0, weight=1)
                self.tk_root.rowconfigure(0, weight=1)

                self.ui_tab_control = ttk.Notebook(self.tk_root)

                self.ui_tab_capture = ttk.Frame(self.ui_tab_control)
                self.ui_tab_calibration = ttk.Frame(self.ui_tab_control)
                self.ui_tab_gaze_estimation = ttk.Frame(self.ui_tab_control)

                self.ui_tab_control.add(self.ui_tab_capture, text ='Capture')
                self.ui_tab_control.add(self.ui_tab_calibration, text ='Calibration')
                self.ui_tab_control.add(self.ui_tab_gaze_estimation, text ='Gaze Estimation')
                self.ui_tab_control.pack(expand = 1, fill ="both")
                
        self.ui_root = UIRoot()
        
        self.tab_capture = UITabCapture(self.ui_root.ui_tab_capture)
        self.tab_calibration = UITabCalibration(self.ui_root.ui_tab_calibration)
        #self.tab_gaze_estimation = UITabGazeEstimation(self.ui_root.ui_tab_gaze_estimation)

        ##### RUN

        self.init_logic()


    def init_logic(self):
        self.tab_capture.init_logic()
        self.tab_calibration.init_logic()
        #self.tab_gaze_estimation.init_logic()
                

    def update(self):
        self.tab_capture.update()
        #self.update_gaze_ui()
        
        
    def on_closing(self):
        self.ui_root.tk_root.destroy()
        self.state.app_running = False


    def run(self):
        self.ui_root.tk_root.protocol("WM_DELETE_WINDOW", self.on_closing)

        while self.state.app_running:
            self.ui_root.tk_root.update_idletasks()
            self.update()
            self.ui_root.tk_root.update()
