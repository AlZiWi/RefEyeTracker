# ellseg_wrapper.py
# Drop-in pupil detector using EllSeg (ritnet_v3)
# Returns the same dict shape as your "detect_pupil_as_precise_dict_*" functions.

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import torch
import numpy as np
import tqdm
# ---- EllSeg modules (import from the EllSeg repo)
# Make sure this file sits inside the EllSeg repo folder or that folder is on sys.path
# at the very top of detect_pupil_ellseg.py, before importing EllSeg modules
from pathlib import Path
import sys
# ======================= BATCHED INFERENCE =======================
from dataclasses import dataclass
from typing import List
import os

HERE = Path(__file__).resolve().parent
# Add the parent of the EllSeg package to sys.path
# (…/Stereo_System must be on sys.path so we can import EllSeg.*)
sys.path.insert(0, str(HERE.parent))

from EllSeg.modelSummary import model_dict               # provides model_dict['ritnet_v3']
# optional: EllSeg's helper can give nicer argmax, but we can do torch.argmax directly
# from utils import get_predictions

# ======================= knobs =======================
ELLSEG_INPUT_HW = (240, 240)   # (H,W)
dirname = os.path.dirname(__file__)
DEFAULT_WEIGHTS = os.path.join(dirname, 'EllSeg/weights/all.git_ok')
USE_GPU = torch.cuda.is_available()
USE_METAL = torch.backends.mps.is_available() and not USE_GPU
# ====================================================



@dataclass
class Ellipse2D:
    cx: float
    cy: float
    a: float
    b: float
    angle_deg: float

def _prep_one_for_batch(
    path: str,
    use_auto_brightness: bool,
    alpha: float,
    beta: float,
):
    """Read → (optional) auto-contrast → align+zscore. Returns (orig_gray, x[1,1,H,W], scale_shift) or None."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        # try manual decode (UNC paths / special chars)
        try:
            with open(path, "rb") as f:
                buf = np.frombuffer(f.read(), np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None
    if im is None:
        return None
    if use_auto_brightness:
        im, _, _ = automatic_brightness_and_contrast(im, clip_hist_percent=0.08)
        im = cv2.convertScaleAbs(im, alpha=float(alpha), beta=float(beta))
    x, scale_shift = _preprocess_align_ritnet(im, out_hw=ELLSEG_INPUT_HW)  # [1,1,H,W]
    return im, x, scale_shift

def _map_small_ellipse_back(
    ell_small, scale_shift: Tuple[float, int]
) -> Tuple[Tuple[float,float], Tuple[float,float], float]:
    """Invert width-scale and vertical pad/crop to original image coords."""
    (cx_s, cy_s), (A_s, B_s), ang = ell_small
    sc, pad_rows = scale_shift
    if pad_rows > 0:
        top = pad_rows // 2
        cy_s = cy_s - top
    elif pad_rows < 0:
        top = (-pad_rows) // 2
        cy_s = cy_s + top
    cx = cx_s / sc
    cy = cy_s / sc
    A = A_s / sc
    B = B_s / sc
    (center, axes, angle_deg) = _normalize_ellipse_opencv(((cx, cy), (A, B), float(ang)))
    return center, axes, angle_deg

def detect_pupil_batch_ellseg(
    img_paths: List[str],
    *,
    checkpoint: str = DEFAULT_WEIGHTS,
    include_image: bool = False,
    use_auto_brightness: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    batch_size: int = 8,
    prefer_pupil_class: int = 2,   # try 2 (pupil) first, fall back to 1 (iris)
) -> List[Optional[Dict[str, Any]]]:
    """
    Batched EllSeg pupil detection. Returns a list aligned with img_paths.
    Each element is either the same dict as detect_pupil_as_precise_dict_ellseg(...) or None on failure.

    Dict keys:
      center (cx,cy), axes (major,minor), angle_deg, contour=None, resid_median_px=None, resid_p95_px=None,
      method, [image if include_image=True]
    """
    if not img_paths:
        return []

    # 1) Load model once
    model, dev = _load_ellseg_model(checkpoint)

    # 2) Pre-process all images on CPU
    prepped: List[Optional[Tuple[np.ndarray, torch.Tensor, Tuple[float,int]]]] = [None]*len(img_paths)
    for i, p in enumerate(img_paths):
        try:
            prepped[i] = _prep_one_for_batch(p, use_auto_brightness, alpha, beta)
        except Exception:
            prepped[i] = None

    # 3) Run forward pass in chunks on target device
    out: List[Optional[Dict[str, Any]]] = [None]*len(img_paths)
    with torch.no_grad():
        i = 0
#        pbar = tqdm(total=len(img_paths), desc="EllSeg batched inference", unit="img", dynamic_ncols=True)

        while i < len(img_paths):
            j = min(i + batch_size, len(img_paths))
            print(f"Batch: {i + batch_size} / 377 ")
            # gather valid prepped entries
            idxs, tuples = [], []
            for k in range(i, j):
                if prepped[k] is not None:
                    idxs.append(k)
                    tuples.append(prepped[k])
            if not idxs:
                i = j
                continue

            # each t[1] is [1,1,H,W]; remove the leading singleton before stacking
            xs = torch.stack([t[1].squeeze(0) for t in tuples], dim=0).to(dev)  # [B,1,H,W]
            assert xs.ndim == 4 and xs.shape[1] == 1, f"bad batch shape: {tuple(xs.shape)}"

            seg = torch.argmax(_forward_seg_logits(xs, model), dim=1).cpu().numpy()  # [B,H,W]

            # 4) Fit ellipse per sample, map back to original resolution, pack dict
            for b, k in enumerate(idxs):
                im, _, scale_shift = tuples[b]
                seg_b = seg[b]

                # try preferred class, then fallback to the other (1 <-> 2)
                cls_try = [prefer_pupil_class, 1 if prefer_pupil_class == 2 else 2]
                ell = None
                for cls in cls_try:
                    ell = _fit_ellipse_from_class_mask(seg_b, pupil_class=cls)
                    if ell is not None:
                        chosen_cls = cls
                        break

                if ell is None:
                    out[k] = None
                    continue

                center, axes, angle_deg = _map_small_ellipse_back(ell, scale_shift)
                result = {
                    "center": center,                     # (cx,cy) in px (original image)
                    "axes": axes,                         # (major,minor) in px
                    "angle_deg": angle_deg,               # deg, major-axis angle
                    "contour": None,
                    "resid_median_px": None,
                    "resid_p95_px": None,
                    "method": f"ellseg(ritnet_v3) cls={chosen_cls}",
                }
                if include_image:
                    result["image"] = im
                out[k] = result
            i = j

 #       pbar.close()
    return out
# ===================== END BATCHED INFERENCE =====================




def automatic_brightness_and_contrast(image: np.ndarray, clip_hist_percent: float = 0.2):
    """Histogram clipping auto-contrast. Returns enhanced image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    acc = np.cumsum(hist)
    maximum = acc[-1]
    clip = clip_hist_percent * (maximum / 100.0) / 2.0
    min_g = int(np.searchsorted(acc, clip))
    max_g = int(np.searchsorted(acc, maximum - clip)) - 1
    if max_g <= min_g:
        return image.copy(), 1.0, 0.0
    alpha = 255.0 / (max_g - min_g)
    beta = -min_g * alpha
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta), alpha, beta


def _normalize_ellipse_opencv(e):
    """Ensure major >= minor and angle in [0,180)."""
    (cx, cy), (A, B), ang = e
    if B > A:
        A, B = B, A
        ang = (ang + 90.0) % 180.0
    return (float(cx), float(cy)), (float(A), float(B)), float(ang)


def _preprocess_align_ritnet(gray_u8: np.ndarray, out_hw=(240, 320)) -> Tuple[torch.Tensor, Tuple[float, int]]:
    """
    EllSeg/RITnet-style preprocessing:
      - align width to W=out_hw[1], scale by sc, then vertically pad/crop to H=out_hw[0]
      - z-score normalize
    Returns:
      x: torch tensor [1,1,H,W] float32
      scale_shift: (sc, pad_rows)   # used to invert to original resolution
    """
    Ht, Wt = out_hw
    H0, W0 = gray_u8.shape[:2]

    # align width
    if W0 != Wt:
        sc = Wt / float(W0)
        new_w = Wt
        new_h = int(round(H0 * sc))
        img_res = cv2.resize(gray_u8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        sc = 1.0
        img_res = gray_u8

    # pad/crop vertically to Ht
    h = img_res.shape[0]
    pad_rows = Ht - h
    if pad_rows > 0:
        # pad equally on top/bottom (if odd, put the extra on bottom)
        top = pad_rows // 2
        bot = pad_rows - top
        img_aligned = np.pad(img_res, ((top, bot), (0, 0)), mode="constant")
    elif pad_rows < 0:
        # crop equally to center
        pad_rows_abs = -pad_rows
        top = pad_rows_abs // 2
        bot = pad_rows_abs - top
        img_aligned = img_res[top:h - bot, :]
    else:
        img_aligned = img_res

    # z-score normalization
    g = img_aligned.astype(np.float32)
    m = float(g.mean())
    s = float(g.std() + 1e-6)
    g = (g - m) / s

    x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(torch.float32)  # [1,1,H,W]
    return x, (sc, pad_rows)


def _rescale_mask_back(seg_map_small: np.ndarray, scale_shift: Tuple[float, int], orig_shape: Tuple[int, int]) -> np.ndarray:
    """
    Bring segmentation indices (HxW) from aligned (240x320) back to original H0xW0.
    """
    H0, W0 = orig_shape
    sc, pad_rows = scale_shift

    # Undo vertical pad/crop
    if pad_rows > 0:
        top = pad_rows // 2
        bot = pad_rows - top
        seg = seg_map_small[top:seg_map_small.shape[0] - bot, :]
    elif pad_rows < 0:
        # We cropped earlier; to go back we need to pad (fill with background=0)
        pad_rows_abs = -pad_rows
        top = pad_rows_abs // 2
        bot = pad_rows_abs - top
        seg = np.pad(seg_map_small, ((top, bot), (0, 0)), mode="edge")
    else:
        seg = seg_map_small

    # Undo width scaling
    # We originally scaled width by sc -> original width = W0
    # So resize to (W0, round(H0*sc)) then final resize to (W0,H0)
    # Easier path: directly resize to (W0,H0) using nearest
    seg_up = cv2.resize(seg.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST)
    return seg_up


# lazy singletons
_MODEL = None
_DEVICE = None


def _load_ellseg_model(weights_path: str = DEFAULT_WEIGHTS):
    """
    Load EllSeg's ritnet_v3 and weights (e.g., ./weights/all.git_ok).
    """
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"EllSeg weights not found at {weights_path}.\n"
            f"Make sure you've run `git lfs pull` in the EllSeg repo."
        )

    ckpt = torch.load(weights_path, map_location="cpu")
    model = model_dict["ritnet_v3"]
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    if USE_GPU:
        dev = torch.device("cuda:0")
    elif USE_METAL:
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    model.to(dev)

    _MODEL, _DEVICE = model, dev
    return _MODEL, _DEVICE


def _forward_seg_logits(x_1x1hw: torch.Tensor, model) -> torch.Tensor:
    """
    Run EllSeg encoder/decoder to get segmentation logits like the official script.
    Returns: seg_logits [1,C,H,W]
    """
    with torch.no_grad():
        x4, x3, x2, x1, x = model.enc(x_1x1hw)
        # latent = torch.mean(x.flatten(start_dim=2), -1)  # not needed unless using regression head
        seg_out = model.dec(x4, x3, x2, x1, x)
    return seg_out


def _fit_ellipse_from_class_mask(seg_map: np.ndarray, pupil_class: int) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Fit an OpenCV ellipse on the largest blob of the selected class in seg_map (HxW).
    """
    mask = (seg_map == pupil_class).astype(np.uint8)
    if mask.sum() < 25:
        return None
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    return cv2.fitEllipse(c)


def detect_pupil_as_precise_dict_ellseg(
    img_path: str,
    *,
    checkpoint: str = DEFAULT_WEIGHTS,      # ./weights/all.git_ok
    include_image: bool = True,
    use_auto_brightness: bool = True,       # optional: apply your auto-contrast before z-score
    alpha: float = 1.0,
    beta: float = 0.0,
    debug_plot: bool = False,
) -> Dict[str, Any]:
    """
    Drop-in EllSeg-backed pupil detector.
    Returns:
      {
        "center": (cx,cy), "axes": (major,minor), "angle_deg": float,
        "contour": None, "resid_median_px": None, "resid_p95_px": None,
        "method": "ellseg(ritnet_v3) …",
        "image": gray uint8 (if include_image=True)
      }
    """

    # -- load gray
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        with open(img_path, "rb") as f:
            buf = np.frombuffer(f.read(), np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(img_path)

    if use_auto_brightness:
        im, _, _ = automatic_brightness_and_contrast(im, clip_hist_percent=0.08)
        im = cv2.convertScaleAbs(im, alpha=float(alpha), beta=float(beta))

    H0, W0 = im.shape[:2]

    # -- preprocess to 240x320 (align width + pad/crop) and z-score
    x, scale_shift = _preprocess_align_ritnet(im, out_hw=ELLSEG_INPUT_HW)  # [1,1,240,320]

    # -- forward
    
    model, dev = _load_ellseg_model(checkpoint)
    seg_logits = _forward_seg_logits(x.to(dev), model)         # [1,C,240,320]
    seg = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()     # (240,320), class indices

    # In EllSeg, final classes typically: 0=bg/sclera, 1=iris, 2=pupil (for the common weights).
    # Try class 2, fallback to 1 if nothing sensible is found.
    ell_small = _fit_ellipse_from_class_mask(seg, pupil_class=2)
    chosen_cls = 2
    if ell_small is None:
        ell_small = _fit_ellipse_from_class_mask(seg, pupil_class=1)
        chosen_cls = 1
    if ell_small is None:
        raise RuntimeError("EllSeg: could not fit ellipse on segmentation mask (no suitable pupil blob).")

    # -- bring segmentation back (not strictly needed, but handy for debug overlay)
    seg_up = _rescale_mask_back(seg, scale_shift, (H0, W0))

    # -- map ellipse back to original resolution using the same inverse as _rescale_mask_back
    # We can recover scale/shift analytically:
    (cx_s, cy_s), (A_s, B_s), ang = ell_small          # small space (aligned)
    sc, pad_rows = scale_shift

    # Undo vertical pad/crop
    if pad_rows > 0:
        top = pad_rows // 2
        cy_s = cy_s - top
    elif pad_rows < 0:
        # we had cropped earlier (i.e., smaller than 240); now coordinates are within the cropped slice
        # to go back, add the top crop
        top = (-pad_rows) // 2
        cy_s = cy_s + top

    # Undo width scaling (we had scaled by sc)
    cx = cx_s / sc
    cy = cy_s / sc
    A = A_s / sc
    B = B_s / sc

    center, axes, angle_deg = _normalize_ellipse_opencv(((cx, cy), (A, B), float(ang)))

    out = {
        "center": center,
        "axes": axes,
        "angle_deg": angle_deg,
        "contour": None,
        "resid_median_px": None,
        "resid_p95_px": None,
        "method": f"ellseg(ritnet_v3) cls={chosen_cls}"
    }
    if include_image:
        out["image"] = im

    if debug_plot:
        # Build a quick overlay for visual verification
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(vis, (center, axes, angle_deg), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(center[0])), int(round(center[1]))), 2, (0, 0, 255), -1, cv2.LINE_AA)
        # Optionally draw the upscaled seg boundaries
        edges = cv2.Canny((seg_up * (255 // max(1, seg_up.max()))).astype(np.uint8), 50, 150)
        vis[edges > 0] = (0, 165, 255)
        cv2.imshow("EllSeg pupil debug", vis)
        cv2.waitKey(1)  # non-blocking; close yourself when done

    return out


def quick_ellseg(
    im,
    *,
    checkpoint: str = DEFAULT_WEIGHTS,      # ./weights/all.git_ok
    include_image: bool = True,
    use_auto_brightness: bool = True,       # optional: apply your auto-contrast before z-score
    alpha: float = 1.0,
    beta: float = 0.0,
    debug_plot: bool = False,
) -> Dict[str, Any]:
    """
    Drop-in EllSeg-backed pupil detector.
    Returns:
      {
        "center": (cx,cy), "axes": (major,minor), "angle_deg": float,
        "contour": None, "resid_median_px": None, "resid_p95_px": None,
        "method": "ellseg(ritnet_v3) …",
        "image": gray uint8 (if include_image=True)
      }
    """
    
    # Profile time using timer
    import time
    time_cps = {}
    time_cps["start"] = time.perf_counter()
    
    if use_auto_brightness:
        im, _, _ = automatic_brightness_and_contrast(im, clip_hist_percent=0.08)
        im = cv2.convertScaleAbs(im, alpha=float(alpha), beta=float(beta))

    time_cps["auto_brightness"] = time.perf_counter()

    H0, W0 = im.shape[:2]

    # -- preprocess to 240x320 (align width + pad/crop) and z-score
    x, scale_shift = _preprocess_align_ritnet(im, out_hw=ELLSEG_INPUT_HW)  # [1,1,240,320]

    time_cps["preprocess"] = time.perf_counter()

    # -- forward
    model, dev = _load_ellseg_model(checkpoint)
    time_cps["load_model"] = time.perf_counter()
    seg_logits = _forward_seg_logits(x.to(dev), model)         # [1,C,240,320]
    time_cps["forward"] = time.perf_counter()
    seg = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()     # (240,320), class indices
    time_cps["argmax"] = time.perf_counter()

    # In EllSeg, final classes typically: 0=bg/sclera, 1=iris, 2=pupil (for the common weights).
    # Try class 2, fallback to 1 if nothing sensible is found.
    ell_small = _fit_ellipse_from_class_mask(seg, pupil_class=2)
    chosen_cls = 2
    if ell_small is None:
        ell_small = _fit_ellipse_from_class_mask(seg, pupil_class=1)
        chosen_cls = 1
    if ell_small is None:
        raise RuntimeError("EllSeg: could not fit ellipse on segmentation mask (no suitable pupil blob).")

    # -- bring segmentation back (not strictly needed, but handy for debug overlay)
    seg_up = _rescale_mask_back(seg, scale_shift, (H0, W0))

    # -- map ellipse back to original resolution using the same inverse as _rescale_mask_back
    # We can recover scale/shift analytically:
    (cx_s, cy_s), (A_s, B_s), ang = ell_small          # small space (aligned)
    sc, pad_rows = scale_shift

    # Undo vertical pad/crop
    if pad_rows > 0:
        top = pad_rows // 2
        cy_s = cy_s - top
    elif pad_rows < 0:
        # we had cropped earlier (i.e., smaller than 240); now coordinates are within the cropped slice
        # to go back, add the top crop
        top = (-pad_rows) // 2
        cy_s = cy_s + top

    # Undo width scaling (we had scaled by sc)
    cx = cx_s / sc
    cy = cy_s / sc
    A = A_s / sc
    B = B_s / sc

    center, axes, angle_deg = _normalize_ellipse_opencv(((cx, cy), (A, B), float(ang)))

    time_cps["fit_ellipse"] = time.perf_counter()
    
    time_deltas = {k: (time_cps[k] - time_cps["start"]) for k in time_cps if k != "start"}

    out = {
        "center": center,
        "axes": axes,
        "angle_deg": angle_deg,
        "contour": None,
        "resid_median_px": None,
        "resid_p95_px": None,
        "method": f"ellseg(ritnet_v3) cls={chosen_cls}",
        "timings": time_deltas
    }
    if include_image:
        out["image"] = im

    if debug_plot:
        # Build a quick overlay for visual verification
        vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(vis, (center, axes, angle_deg), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (int(round(center[0])), int(round(center[1]))), 2, (0, 0, 255), -1, cv2.LINE_AA)
        # Optionally draw the upscaled seg boundaries
        edges = cv2.Canny((seg_up * (255 // max(1, seg_up.max()))).astype(np.uint8), 50, 150)
        vis[edges > 0] = (0, 165, 255)
        cv2.imshow("EllSeg pupil debug", vis)
        cv2.waitKey(1)  # non-blocking; close yourself when done

    return out


# ---- quick local test
if __name__ == "__main__":
    img = r"/Users/ge83nax/Desktop/code/ref_system/tum_track/wearable_tests/random_series_02_pursuit/1/frame_0_timestamp_0.000.png"
    res = detect_pupil_as_precise_dict_ellseg(
        img,
        checkpoint=DEFAULT_WEIGHTS,
        include_image=True,
        use_auto_brightness=True,
        alpha=1.0, beta=0.0,
        debug_plot=True,
    )
    print(res["center"], res["axes"], res["angle_deg"], res["method"])
    if "image" in res:
        cv2.imshow("Input image", res["image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()