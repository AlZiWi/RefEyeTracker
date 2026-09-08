from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pathlib import Path
from typing import Optional, Tuple, List

from app.data_structures import CameraCoordinateFrame, CameraParams, CameraParamsExtrinsic, CameraParamsIntrinsic, MonoCalibrationStatistics, MonoReprojectionErrors, StereoCalibrationResults


# --- unchanged core ---
def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)

def robust_threshold(scores: np.ndarray, k: float = 3.0) -> float:
    """Median + k*MAD (robust gegen Ausreißer)."""
    med = np.median(scores)
    return float(med + k * mad(scores))

def per_view_mean_error(objpoints, imgpoints, rvecs, tvecs, K, dist) -> np.ndarray:
    """Mean reprojection error je Bild (px)."""
    errs = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        diff = imgp.reshape(-1,2) - proj.reshape(-1,2)
        errs.append(np.linalg.norm(diff, axis=1).mean())
    return np.array(errs, float)

# --- NEW: detailed per-view stats (p95 & max) but optional to use ---
def per_view_error_stats(objpoints, imgpoints, rvecs, tvecs, K, dist):
    """Returns dict of arrays: mean, p95, max (px) per view."""
    meanv, p95v, maxv = [], [], []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err = np.linalg.norm(imgp.reshape(-1,2) - proj.reshape(-1,2), axis=1)
        meanv.append(float(err.mean()))
        p95v.append(float(np.percentile(err, 95.0)))
        maxv.append(float(err.max()))
    return {
        "mean": np.asarray(meanv, float),
        "p95":  np.asarray(p95v,  float),
        "max":  np.asarray(maxv,  float),
    }

def filter_by_mono_errors(obj_cal, img_cal, rvecs, tvecs, K, dist,
                          k_mad: float = 3.0,
                          k_mad_p95: float = 3.0,
                          max_cap_px: float = 1.0):
    """
    Entfernt Paare, die links oder rechts auffällig schlecht sind.
    Kompatibel zur alten Signatur; zusätzliche Filter sind optional.

    Regeln:
      - mean-Score (robust): median + k*MAD auf max(left_mean, right_mean)
      - optional p95-Score (robust): median + k*MAD auf max(left_p95, right_p95)
      - harte Kappe auf max-Fehler: max(left_max, right_max) <= max_cap_px
    """
    
    mean_errors = per_view_mean_error(obj_cal, img_cal, rvecs, tvecs, K, dist)
    stats = per_view_error_stats(obj_cal, img_cal, rvecs, tvecs, K, dist)
    
    score_mean = mean_errors
    score_p95 = stats["p95"]
    score_max = stats["max"]
    
    thr_mean = robust_threshold(score_mean, k=k_mad)
    thr_p95 = robust_threshold(score_p95, k=k_mad_p95)

    # Combine scores
    keep_mask = (score_mean <= thr_mean) & (score_max <= max_cap_px) & (score_p95 <= thr_p95)

    keep = np.where(keep_mask)[0].tolist()
    drop = np.where(~keep_mask)[0].tolist()
    
    details = {
        "score_mean": score_mean,
        "score_p95": score_p95,
        "score_max": score_max,
        "thr_mean": float(thr_mean),
        "thr_p95": float(thr_p95),
        "thr_max_cap": float(max_cap_px),
        "stats": stats
    }
    
    return keep, drop, details


def filter_pairs_by_mono_errors(obj_cal, img1_cal, img2_cal,
                          rvecs1, tvecs1, K1, dist1,
                          rvecs2, tvecs2, K2, dist2,
                          k_mad: float = 3.0,
                          k_mad_p95: float = 3.0,
                          max_cap_px: float = 1.0):
    
    _, _, details_1 = filter_by_mono_errors(obj_cal, img1_cal, rvecs1, tvecs1, K1, dist1, k_mad=k_mad, k_mad_p95=k_mad_p95, max_cap_px=max_cap_px)
    _, _, details_2 = filter_by_mono_errors(obj_cal, img2_cal, rvecs2, tvecs2, K2, dist2, k_mad=k_mad, k_mad_p95=k_mad_p95, max_cap_px=max_cap_px)
    
    score_mean = np.maximum(details_1["score_mean"], details_2["score_mean"])
    score_p95 = np.maximum(details_1["score_p95"], details_2["score_p95"])
    score_max = np.maximum(details_1["score_max"], details_2["score_max"])
    
    thr_mean = robust_threshold(score_mean, k=k_mad)
    thr_p95 = robust_threshold(score_p95, k=k_mad_p95)

    keep_mask = (score_mean <= thr_mean) & (score_max <= max_cap_px) & (score_p95 <= thr_p95)

    keep = np.where(keep_mask)[0].tolist()
    drop = np.where(~keep_mask)[0].tolist()

    details = {
        "score_mean": score_mean,
        "score_p95": score_p95,
        "score_max": score_max,
        "thr_mean": float(thr_mean),
        "thr_p95": float(thr_p95),
        "thr_max_cap": float(max_cap_px),
        "details_1":  details_1,
        "details_2": details_2
    }
    
    return keep, drop, details


def per_view_sampson(F: np.ndarray, imgL: list[np.ndarray], imgR: list[np.ndarray]) -> np.ndarray:
    """Mean Sampson distance je Bildpaar (px). Erwartet Listen (Train/Calib-Set)."""
    out = []
    for pl, pr in zip(imgL, imgR):
        xl = cv2.convertPointsToHomogeneous(pl).reshape(-1, 3)
        xr = cv2.convertPointsToHomogeneous(pr).reshape(-1, 3)
        Fx_l  = (F @ xl.T).T
        Ft_xr = (F.T @ xr.T).T
        xrT_F_xl = np.sum(xr * (xl @ F.T), axis=1)
        d = np.sqrt((xrT_F_xl**2) / (Fx_l[:,0]**2 + Fx_l[:,1]**2 + Ft_xr[:,0]**2 + Ft_xr[:,1]**2 + 1e-12))
        out.append(d.mean())
    return np.array(out, float)
    

def build_camparams_from_K(
    K: np.ndarray,
    dist: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    img_size: Tuple[int, int],           # (nx, ny)
    R: np.ndarray,
    t: np.ndarray,
    pixel_pitch_mm: Tuple[float, float] = (1.0, 1.0),
    statistics: Optional[MonoCalibrationStatistics] = None
) -> CameraParams:
    
    K = np.asarray(K, float).reshape(3, 3)
    nx, ny = map(int, img_size)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if not (0.0 <= cx <= nx and 0.0 <= cy <= ny):
        print(f"[warn] principal point outside image: (cx,cy)=({cx:.1f},{cy:.1f}) not in [0,{nx}]x[0,{ny}]")

    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3, 1)
    
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    T_R = np.eye(4, dtype=float)
    T_R[:3, :3] = R
    
    T_t = np.eye(4, dtype=float)
    T_t[:3, 3] = t.flatten()

    px_mm, py_mm = map(float, pixel_pitch_mm)
    f_mm_fx = fx * px_mm
    f_mm_fy = fy * py_mm
    f_mm = 0.5 * (f_mm_fx + f_mm_fy)
    if abs(f_mm_fx - f_mm_fy) > 0.01 * max(abs(f_mm_fx), abs(f_mm_fy), 1.0):
        print(f"[warn] fx*px_mm != fy*py_mm: {f_mm_fx:.6f} vs {f_mm_fy:.6f} mm (check pixel_pitch_mm or K anisotropy)")
    W_mm = nx * px_mm
    H_mm = ny * py_mm

    camera_params = CameraParams(
        intrinsic=CameraParamsIntrinsic(
            K=K,
            dist=dist,
            map1=map1,
            map2=map2,
            f_mm=f_mm,
            W_mm=W_mm,
            H_mm=H_mm,
            nx=nx,
            ny=ny,
            px_mm=px_mm,
            py_mm=py_mm,
            cx=cx,
            cy=cy,
            statistics=statistics
        ),
        extrinsic=CameraParamsExtrinsic(
            relative=CameraCoordinateFrame(
                T=T,
                T_R=T_R,
                T_t=T_t
            )
        )
    )

    return camera_params


def build_object_points(pattern_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_mm)
    return objp


def detect_corners(img, pattern_size, verbose=False):
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1) SB (super robust)
    # zla2fe added markers
    ok, corners, meta = cv2.findChessboardCornersSBWithMeta(
        gray, pattern_size, flags=cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_EXHAUSTIVE# + cv2.CALIB_CB_MARKER  # zla2fe appears to be slow
    )
    flipped = False
    if not ok:
        # 2) klassisch als Fallback
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not ok:
            if verbose:
                print(f"[dbg] corner detection failed")
            return None
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    else:
        if meta[0,0] == 2:  # detected corners start at bottom of calibration pattern instead of top -> rotate arrays (meta == 2 -> top-left corner of white cell)
            # rotate corners and meta by 180 degrees. Needs to reshape to 2D grid first, then rotate, then reshape back to 1D list of corners
            corners = corners.reshape(meta.shape[0], meta.shape[1], 2)
            corners = np.rot90(corners, 2)
            corners = corners.reshape(-1, 2)
            meta = np.rot90(meta, 2)
            flipped = True
    if verbose:
        c0, cN = corners[0].ravel(), corners[-1].ravel()
        print(f"[dbg] first corner: ({c0[0]:.2f},{c0[1]:.2f})  last: ({cN[0]:.2f},{cN[1]:.2f})  n={len(corners)}  flipped={flipped}")
    return corners


def calibrate_mono(img_points: List[np.ndarray], obj_points: List[np.ndarray], image_size: Tuple[int, int]):
    flags = 0  # standard 5-param model
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    fx_guess = fy_guess = (1.5 / 2.7) * image_size[0] #100.0 / (2*0.00345)  # ≈ 23188 px   # zla2fe patch (focal_mm / sensor_width_mm) * image_width_in_pixels
    cx_guess, cy_guess = image_size[0] / 2, image_size[1] / 2  # zla2fe patch
    K_init = np.array([[fx_guess, 0, cx_guess],
                       [0, fy_guess, cy_guess],
                       [0, 0, 1]], dtype=np.float64)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, K_init, None, flags= cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_PRINCIPAL_POINT, criteria=term  # |cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_PRINCIPAL_POINT
    )
    return rms, K, dist, rvecs, tvecs


# === PATCH: Hold-out-Indices erzeugen =======================================
# --- PATCH: Hold-out Split --------------------------------------------------
def make_holdout_indices(n: int, holdout_ratio: float = 0.2, seed: int = 42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_hold = max(1, int(round(n * holdout_ratio)))
    hold = sorted(idx[:n_hold].tolist())
    calib = sorted(idx[n_hold:].tolist()) if n > n_hold else []
    return calib, hold
# ----------------------------------------------------------------------------
# ============================================================================
    


def compute_mono_reproj_errors(
    img_points: List[np.ndarray],
    obj_points: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray
) -> MonoReprojectionErrors:
    per_view = []
    all_errs = []
    all_errs_2d = []
    for i, (rvec, tvec, objp, imgp) in enumerate(zip(rvecs, tvecs, obj_points, img_points)):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(imgp.reshape(-1, 2) - proj, axis=1)
        per_view.append(MonoReprojectionErrors.PerViewMonoReprojectionError(
            index=i,
            mean_px=float(err.mean()),
            std_px=float(err.std()),
            max_px=float(err.max()),
            num_pts=int(err.size),
        ))
        all_errs.append(err)
        all_errs_2d.append(imgp.reshape(-1, 2) - proj)
    all_errs = np.concatenate(all_errs) if all_errs else np.array([])
    all_errs_2d = np.concatenate(all_errs_2d) if all_errs_2d else np.array([])
    return MonoReprojectionErrors(
        mean_px=float(all_errs.mean()) if all_errs.size > 0 else None,
        std_px=float(all_errs.std()) if all_errs.size > 0 else None,
        max_px=float(all_errs.max()) if all_errs.size > 0 else None,
        per_view=per_view,
        all_errs=all_errs,
        all_errs_2d=all_errs_2d
    )


def per_view_reproj_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    errs = []
    for i, (objp, imgp) in enumerate(zip(objpoints, imgpoints)):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(imgp.reshape(-1, 2) - proj, axis=1).mean()
        errs.append(err)
    return np.array(errs)


def vertical_disparity_stats(ptsL, ptsR, K1, dist1, K2, dist2, R, T, image_size):
    w, h = image_size  # <-- (w,h)

    # Listen → stapeln
    if isinstance(ptsL, list): ptsL = np.vstack([p.reshape(-1,1,2) for p in ptsL])
    else: ptsL = ptsL.reshape(-1,1,2)
    if isinstance(ptsR, list): ptsR = np.vstack([p.reshape(-1,1,2) for p in ptsR])
    else: ptsR = ptsR.reshape(-1,1,2)

    # Rectify-Parameter
    assert isinstance(K1, np.ndarray) and isinstance(K2, np.ndarray)
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(K1, dist1, K2, dist2, (w, h), R, T, alpha=0)

    def undist_rectify(pts, K, dist, Rr, Pr):
        out = cv2.undistortPoints(pts, K, dist, R=Rr, P=Pr).reshape(-1,2)
        return out

    vL = undist_rectify(ptsL, K1, dist1, R1, P1)
    vR = undist_rectify(ptsR, K2, dist2, R2, P2)

    # Sanity: Koordinaten sollten in Pixelbereich liegen (± etwas Puffer)
    vmax = np.max(np.abs(np.concatenate([vL, vR], axis=0)), axis=0)
    if vmax[0] > 4*w or vmax[1] > 4*h:
        print(f"[warn] vertical_disparity_stats: projizierte Pixelkoords unrealistisch groß "
              f"(max|x|={vmax[0]:.1f}, max|y|={vmax[1]:.1f}) – prüfe image_size=(w,h) Reihenfolge, "
              f"K/dist, oder falsche Punktlisten (Train/Holdout vertauscht).")
        return {"mean_abs_px": None, "std_px": None, "max_abs_px": None, "n": 0}

    dy = vL[:,1] - vR[:,1]
    return {
        "mean_abs_px": float(np.mean(np.abs(dy))),
        "std_px": float(np.std(dy)),
        "max_abs_px": float(np.max(np.abs(dy))),
        "n": int(dy.size),
    }


def vertical_disparity_via_remap(
    pairs, K1, dist1, K2, dist2, R, T, image_size,
    pattern_size, max_pairs_eval=10, verbose=True
):

    # 1) Rectify-Matrizen & Maps
    R1, R2, P1, P2, Q, _, _, map1_l, map2_l, map1_r, map2_r = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T, alpha=0.0
    )

    def detect(img):
        # gleiche Chessboard-Detektion wie im Hauptpfad
        ok, c = cv2.findChessboardCornersSB(img, pattern_size,
                    flags=cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if not ok:
            ok, c = cv2.findChessboardCorners(img, pattern_size,
                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(img, c, (11,11), (-1,-1),
                                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-6))
        return c if ok else None

    dy_all = []
    used = 0
    for i, (lp, rp) in enumerate(pairs[:max_pairs_eval]):
        L = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        Rimg = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if L is None or Rimg is None: continue
        Lr = cv2.remap(L, map1_l, map2_l, cv2.INTER_LINEAR)
        Rr = cv2.remap(Rimg, map1_r, map2_r, cv2.INTER_LINEAR)
        cl = detect(Lr); cr = detect(Rr)
        if cl is None or cr is None: continue
        # gleiche Orientierung sicherstellen, falls nötig:
        # cr = ensure_same_orientation(cl, cr, pattern_cols, pattern_rows)

        # dy in PIXEL nach Rectify
        vL = cl.reshape(-1,2); vR = cr.reshape(-1,2)
        if vL.shape != vR.shape:  # Sicherheitsnetz
            n = min(len(vL), len(vR)); vL = vL[:n]; vR = vR[:n]
        dy = vL[:,1] - vR[:,1]
        dy_all.append(dy); used += 1

    if not dy_all:
        return {"mean_abs_px": None, "std_px": None, "max_abs_px": None, "n": 0, "used_pairs": 0}
    dy_all = np.concatenate(dy_all)
    return {
        "mean_abs_px": float(np.mean(np.abs(dy_all))),
        "std_px": float(np.std(dy_all)),
        "max_abs_px": float(np.max(np.abs(dy_all))),
        "n": int(dy_all.size),
        "used_pairs": used
    }


def calibrate_stereo(
    obj_points: List[np.ndarray],
    img_points_l: List[np.ndarray],
    img_points_r: List[np.ndarray],
    image_size: Tuple[int, int],
    K1: np.ndarray, dist1: np.ndarray,
    K2: np.ndarray, dist2: np.ndarray,
):
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms, K1o, dist1o, K2o, dist2o, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r, K1, dist1, K2, dist2,
        image_size, criteria=term, flags=(cv2.CALIB_FIX_INTRINSIC|cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_FOCAL_LENGTH|cv2.CALIB_FIX_ASPECT_RATIO)  # zla2fe Tangentdist fix
    )
    return rms, K1o, dist1o, K2o, dist2o, R, T, E, F


def ensure_same_orientation(ptsA: np.ndarray, ptsB: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """
    Erzwingt gleiche Zeilen/Spalten-Orientierung von ptsB wie ptsA.
    pts*: (N,1,2) Corner-Reihenfolge wie findChessboardCorners.
    """
    N = cols * rows
    A = ptsA.reshape(N, 2)
    B = ptsB.reshape(N, 2)

    # Richtungsvektoren: erste Reihe (0 -> cols-1) und erste Spalte (0 -> (rows-1)*cols)
    def row_col_dirs(P):
        v_row = P[cols-1] - P[0]
        v_col = P[(rows-1)*cols] - P[0]
        return v_row, v_col

    a_r, a_c = row_col_dirs(A)
    b_r, b_c = row_col_dirs(B)

    # Gleiche Orientierung = beide Skalarprodukte > 0
    s_row = float(np.dot(a_r, b_r))
    s_col = float(np.dot(a_c, b_c))

    Bcorr = B.copy()
    if s_row < 0:
        # Spaltenrichtung spiegeln: jede Reihe umdrehen
        Bcorr = Bcorr.reshape(rows, cols, 2)
        Bcorr = Bcorr[:, ::-1, :]
        Bcorr = Bcorr.reshape(N, 2)
    # Nach möglicher Spiegellung Spaltenrichtung neu berechnen:
    b_r2, b_c2 = row_col_dirs(Bcorr)

    if float(np.dot(a_c, b_c2)) < 0:
        # Zeilenrichtung spiegeln: Zeilen umdrehen
        Bcorr = Bcorr.reshape(rows, cols, 2)
        Bcorr = Bcorr[::-1, :, :]
        Bcorr = Bcorr.reshape(N, 2)

    return Bcorr.reshape(N, 1, 2)


def compute_stereo_epipolar_error(
    img_points_l: List[np.ndarray],
    img_points_r: List[np.ndarray],
    F: np.ndarray
) -> Dict[str, Any]:
    # mean absolute Sampson distance per view (approx. epipolar error)
    per_view = []
    all_errs = []
    for i, (pl, pr) in enumerate(zip(img_points_l, img_points_r)):
        xl = cv2.convertPointsToHomogeneous(pl).reshape(-1, 3)
        xr = cv2.convertPointsToHomogeneous(pr).reshape(-1, 3)
        # Sampson distance
        Fx_l = (F @ xl.T).T
        Ft_xr = (F.T @ xr.T).T
        xrT_F_xl = np.sum(xr * (xl @ F.T), axis=1)
        num = xrT_F_xl ** 2
        den = Fx_l[:, 0] ** 2 + Fx_l[:, 1] ** 2 + Ft_xr[:, 0] ** 2 + Ft_xr[:, 1] ** 2
        d = np.sqrt(num / (den + 1e-12))
        per_view.append({
            "index": i,
            "mean_px": float(np.mean(d)),
            "std_px": float(np.std(d)),
            "max_px": float(np.max(d)),
            "num_pts": int(d.size),
        })
        all_errs.append(d)
    all_errs = np.concatenate(all_errs) if all_errs else np.array([])
    return {
        "mean_px": float(all_errs.mean()) if all_errs.size else None,
        "std_px": float(all_errs.std()) if all_errs.size else None,
        "max_px": float(all_errs.max()) if all_errs.size else None,
        "per_view": per_view,
    }


def rectify_distortion(K, dist, image_size):
    # TUMHCTLAZ This only works for shift along one axis, not arbitrary R, T
    #R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, dist1, K2, dist2, image_size, R, T, alpha=alpha=0.0)
    
    P, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, P, image_size, cv2.CV_32FC1)

    return map1, map2


# --- PATCH: Rectified-Preview mit horizontalen Hilfslinien ------------------
# Nutze die bereits existierenden Maps (map1_l, map2_l, map1_r, map2_r)
def save_rectified_previews_with_guides(
    pairs,                 # Liste[(left_path, right_path)]
    K1, dist1, K2, dist2,  # Numpy-Arrays aus der Kalibrierung
    R, T,                  # Numpy-Arrays
    image_size,            # (w, h) aus der Kalibrierung
    out_dir,
    count=2,
    step=40,
    alpha=0.0,
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def to_u8(img):
        if img is None: return None
        if img.dtype == np.uint16 or img.dtype == np.float32 or img.dtype == np.float64:
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    def draw_guides(img8, step):
        c = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR) if img8.ndim == 2 else img8.copy()
        h, w = c.shape[:2]
        for y in range(0, h, step):
            cv2.line(c, (0, y), (w-1, y), (0, 255, 0), 1, cv2.LINE_AA)
        return c

    # Maps mit den Kalibrier-Parametern und image_size (w,h) berechnen
    w_cal, h_cal = image_size
    #R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, dist1, K2, dist2, (w_cal, h_cal), R, T, alpha=alpha)  # Only works for shift along one axis, not for general R/T
    #map1_l, map2_l = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, (w_cal, h_cal), cv2.CV_32FC1)
    #map1_r, map2_r = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, (w_cal, h_cal), cv2.CV_32FC1)
    
    P1, roi1 = cv2.getOptimalNewCameraMatrix(K1, dist1, (w_cal, h_cal), 1, (w_cal, h_cal))
    P2, roi2 = cv2.getOptimalNewCameraMatrix(K2, dist2, (w_cal, h_cal), 1, (w_cal, h_cal))
    R1 = R2 = None

    print(f"[dbg] roi1: {roi1}, roi2: {roi2}")

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, dist1, None, P1, (w_cal, h_cal), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, dist2, None, P2, (w_cal, h_cal), cv2.CV_32FC1)

    def valid_ratio_for_map(m1, m2, w, h):
        # Anteil der Koordinaten, die im Zielbild liegen
        x_ok = (m1 >= 0) & (m1 < w)
        y_ok = (m2 >= 0) & (m2 < h)
        ok = (x_ok & y_ok).astype(np.uint8)
        return float(ok.mean())

    # Debug: Map-Validität gegen (w_cal,h_cal) checken
    vrL = valid_ratio_for_map(map1_l, map2_l, w_cal, h_cal)
    vrR = valid_ratio_for_map(map1_r, map2_r, w_cal, h_cal)
    print(f"[dbg] map valid ratio vs cal size: left={vrL:.3f}, right={vrR:.3f}")

    for i, (lp, rp) in enumerate(pairs[:count]):
        L = cv2.imread(str(lp))
        Rimg = cv2.imread(str(rp))
        
        if L is None or Rimg is None:
            print(f"[warn] konnte Bilder nicht laden: {lp} / {rp}")
            continue
        print("[chk] raw L/R size:", L.shape[::-1], Rimg.shape[::-1])  # (w,h)
        print("[chk] img_size (calib):", image_size)  # (w,h)
        print("[chk] K1 cx,cy:", float(K1[0, 2]), float(K1[1, 2]))
        print("[chk] K2 cx,cy:", float(K2[0, 2]), float(K2[1, 2]))

        # Wenn die Bildgröße NICHT der Kalibriergröße entspricht, Maps on-the-fly neu bauen
        w_img, h_img = int(L.shape[1]), int(L.shape[0])
        if (w_img, h_img) != (w_cal, h_cal):
            print(f"[warn] image_size mismatch: cal=({w_cal},{h_cal}) vs img=({w_img},{h_img}) -> rebuild maps")
            R1i, R2i, P1i, P2i, Qi, _, _ = cv2.stereoRectify(K1, dist1, K2, dist2, (w_img, h_img), R, T, alpha=alpha)
            map1_l_i, map2_l_i = cv2.initUndistortRectifyMap(K1, dist1, R1i, P1i, (w_img, h_img), cv2.CV_32FC1)
            map1_r_i, map2_r_i = cv2.initUndistortRectifyMap(K2, dist2, R2i, P2i, (w_img, h_img), cv2.CV_32FC1)
        else:
            map1_l_i, map2_l_i = map1_l, map2_l
            map1_r_i, map2_r_i = map1_r, map2_r
        print("[dbg] map1_l range:", float(np.nanmin(map1_l)), float(np.nanmax(map1_l)))
        print("[dbg] map2_l range:", float(np.nanmin(map2_l)), float(np.nanmax(map2_l)))
        print("[dbg] map1_r range:", float(np.nanmin(map1_r)), float(np.nanmax(map1_r)))
        print("[dbg] map2_r range:", float(np.nanmin(map2_r)), float(np.nanmax(map2_r)))
        # Noch ein Check: wie viel der Maps liegt im gültigen Bereich dieser Bildgröße?
        vrLi = valid_ratio_for_map(map1_l_i, map2_l_i, w_img, h_img)
        vrRi = valid_ratio_for_map(map1_r_i, map2_r_i, w_img, h_img)
        print(f"[dbg] valid ratio vs image size: left={vrLi:.3f}, right={vrRi:.3f}")

        Lr = cv2.remap(L, map1_l_i, map2_l_i, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        Rr = cv2.remap(Rimg, map1_r_i, map2_r_i, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print(f"[dbg] rectified mean L/R: {float(Lr.mean()):.3f} / {float(Rr.mean()):.3f}")

        Lg = draw_guides(to_u8(Lr), step)
        Rg = draw_guides(to_u8(Rr), step)
        cv2.imwrite(str(out_dir / f"rectified_left_guides_{i:02d}.png"), Lg)
        cv2.imwrite(str(out_dir / f"rectified_right_guides_{i:02d}.png"), Rg)

def automatic_brightness_and_contrast(image: np.ndarray, clip_hist_percent: float = .1):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = [float(hist[0])]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i - 1] + float(hist[i]))
    maximum = accumulator[-1]
    clip = clip_hist_percent * (maximum / 100.0) / 2.0

    minimum_gray, maximum_gray = 0, hist_size - 1
    while minimum_gray < hist_size and accumulator[minimum_gray] < clip:
        minimum_gray += 1
    while maximum_gray > 0 and accumulator[maximum_gray] >= (maximum - clip):
        maximum_gray -= 1

    if maximum_gray == minimum_gray:
        # Fallback: keine sinnvolle Spreizung möglich
        return image.copy(), 1.0, 0.0

    alpha = 255.0 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta
 

def run_opencv_mono_calibration(img_paths: list[Path], pattern_size: tuple[int, int], square_size_mm: float, pixel_pitch_mm: tuple[float, float], verbose=False) -> CameraParamsIntrinsic:
    # Pixel pitch µm → mm
    px_mm = pixel_pitch_mm[0]
    py_mm = pixel_pitch_mm[1]

    # Corner Detection
    
    objp = build_object_points(pattern_size, float(square_size_mm))
    obj_points = []
    img_points = []
    img_size = None

    for i, path in enumerate(img_paths):
        
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[warn] IMG {i}: Could not load: {path}")
            continue
        
        img, _, _ = automatic_brightness_and_contrast(img, clip_hist_percent=0.5)
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # (nx, ny)

        corners = detect_corners(img, pattern_size, verbose=False)
        if corners is None:
            if verbose:
                print(f"[dbg] No corners found (i={i}).")
            continue

        obj_points.append(objp.copy())
        img_points.append(corners)

    if not obj_points:
        raise RuntimeError("No valid pattern detections. Check pattern size/lighting/images.")
    
    # Calibration for filter
    rms, K, dist, rvecs, tvecs = calibrate_mono(img_points, obj_points, img_size)

    print(f"[mono] RMS: {rms:.4f}")

    errs_mono_reproj_initial = compute_mono_reproj_errors(img_points, obj_points, rvecs, tvecs, K, dist)

    if verbose:
        print("[dbg] Mono reprojection (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj_initial))

    # Remove outliers based on mono reprojection error
    keep, drop, details = filter_by_mono_errors(obj_points, img_points, rvecs, tvecs, K, dist, k_mad=2.5, max_cap_px=1.0)  # k_mad 2.5..3.5 is common

    if len(drop):
        print(f"[filter-mono] Drop {len(drop)} / {len(obj_points)} (thr={details['thr_mean']:.3f}px)")

        obj_points = [obj_points[i] for i in keep]
        img_points = [img_points[i] for i in keep]
    else:
        print("[filter-mono] No images removed")

    # Final Calibration
    rms, K, dist, rvecs, tvecs = calibrate_mono(img_points, obj_points, img_size)

    print(f"[mono] RMS: {rms:.4f}")

    errs_mono_reproj = compute_mono_reproj_errors(img_points, obj_points, rvecs, tvecs, K, dist)

    if verbose:
        print("[dbg] Mono reprojection 2 (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj))

    # def estimate_magnification_from_chessboard(img_points, square_size_mm, pixel_pitch_mm):
    #     lengths_px = []
    #     for corners in img_points:
    #         pts = corners.reshape(-1, 2)
    #         # lokale Pixelabstände (horizontale Nachbarn)
    #         seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    #         lengths_px.append(np.mean(seg))
    #     mean_edge_px = float(np.mean(lengths_px))
    #     M = (mean_edge_px * pixel_pitch_mm) / float(square_size_mm)
    #     return M, mean_edge_px

    #M_est, edge_px = estimate_magnification_from_chessboard(img_points, square_size_mm, px_mm)
    #print(f"[scale] mean edge = {edge_px:.2f}px  →  M ≈ {M_est:.4f}")
    
    map1, map2 = rectify_distortion(K, dist, img_size)
    
    mono_calibration_result = CameraParamsIntrinsic(
        K=K,
        dist=dist,
        map1=map1,
        map2=map2,
        f_mm=np.mean([K[0,0]*px_mm, K[1,1]*py_mm]),
        nx=img_size[0],
        ny=img_size[1],
        px_mm=px_mm,
        py_mm=py_mm,
        W_mm=img_size[0]*px_mm,
        H_mm=img_size[1]*py_mm,
        cx=K[0,2],
        cy=K[1,2],
        
        statistics=MonoCalibrationStatistics(
            rms=rms,
            errs_mono_reproj_initial=errs_mono_reproj_initial,
            errs_mono_reproj=errs_mono_reproj,
            num_images=len(obj_points)
        )
    )

    return mono_calibration_result


def initialize_stereo_calibration_results(camera_params_0: CameraParams, camera_params_1: CameraParams, R: np.ndarray, t: np.ndarray, E: np.ndarray, F: np.ndarray) -> StereoCalibrationResults:
    camera_params_0.extrinsic.relative.origin = np.array([[0],[0],[0],[1]])
    camera_params_0.extrinsic.relative.x = np.array([[1],[0],[0],[1]])
    camera_params_0.extrinsic.relative.y = np.array([[0],[1],[0],[1]])
    camera_params_0.extrinsic.relative.z = np.array([[0],[0],[1],[1]])

    T = camera_params_1.extrinsic.relative.T
    T_R = camera_params_1.extrinsic.relative.T_R

    camera_params_1.extrinsic.relative.origin = T @ camera_params_0.extrinsic.relative.origin
    camera_params_1.extrinsic.relative.x = T_R @ camera_params_0.extrinsic.relative.x
    camera_params_1.extrinsic.relative.y = T_R @ camera_params_0.extrinsic.relative.y
    camera_params_1.extrinsic.relative.z = T_R @ camera_params_0.extrinsic.relative.z
    
    results = StereoCalibrationResults(
        camera_params_0=camera_params_0,
        camera_params_1=camera_params_1,
        R=R,
        t=t,
        E=E,
        F=F,
    )
    
    return results


def run_opencv_stereo_calibration(
    img_path_pairs: List[Tuple[Path, Path]],
    pattern_size: Tuple[int, int],  # cols, rows, inner corners only!
    square_size_mm: float,
    pixel_pitch_mm: Tuple[float, float],
    left_dir_mono = None,  # optional separate directory for mono calibration
    right_dir_mono = None,
    verbose: bool = False,
) -> StereoCalibrationResults:
    
    # Pixel pitch µm → mm
    px_mm = pixel_pitch_mm[0]
    py_mm = pixel_pitch_mm[1]
    
    # 2) Ecken detektieren
    objp = build_object_points(pattern_size, float(square_size_mm))
    obj_points: List[np.ndarray] = []
    img_points_l: List[np.ndarray] = []
    img_points_r: List[np.ndarray] = []

    img_size: Optional[Tuple[int, int]] = None

    for i, (lp, rp) in enumerate(img_path_pairs):
        imgl = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)
        imgr = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        if imgl is None or imgr is None:
            print(f"[warn] Paar {i}: Bild konnte nicht geladen werden: {lp} / {rp}")
            continue
        
        # autoadjust brightness
        imgl, _, _ = automatic_brightness_and_contrast(imgl, clip_hist_percent=0.5)
        imgr, _, _ = automatic_brightness_and_contrast(imgr, clip_hist_percent=0.5)
        if img_size is None:
            img_size = (imgl.shape[1], imgl.shape[0])  # (nx, ny)

        corners_l = detect_corners(imgl, pattern_size, verbose=False)
        corners_r = detect_corners(imgr, pattern_size, verbose=False)
        if corners_l is None or corners_r is None:
            if verbose:
                print(f"[dbg] Ecken nicht in beiden Bildern gefunden (i={i}).")
            continue
        # Orientierung an linkem Bild ausrichten:
        #cr = ensure_same_orientation(cl, cr, pattern_cols, pattern_rows)  # zla2fe deaktiviert, da sonst evtl. falsche Orientierung bei asymm. Mustern
        obj_points.append(objp.copy())
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    if not obj_points:
        raise RuntimeError("Keine gültigen Corner-Detektionen. Prüfe Mustergröße/Beleuchtung/Bilder.")

    # Split erzeugen (z.B. 20% Hold-out)
    # Hold-out erzeugen (z.B. 20%)
    calib_idx, hold_idx = make_holdout_indices(len(obj_points), holdout_ratio=0.2, seed=42)

    obj_points_cal, img_points_l_cal, img_points_r_cal = (
        [obj_points[i] for i in calib_idx] if calib_idx else obj_points,
        [img_points_l[i] for i in calib_idx] if calib_idx else img_points_l,
        [img_points_r[i] for i in calib_idx] if calib_idx else img_points_r,
    )
    obj_points_hold = [obj_points[i] for i in hold_idx]
    img_points_l_hold = [img_points_l[i] for i in hold_idx]
    img_points_r_hold = [img_points_r[i] for i in hold_idx]

    assert img_size is not None
    print(f"[info] Verwendete Bilder: {len(obj_points)}  | Bildgröße: {img_size} (nx,ny)")

    # 3) Monokalibrierung
    # Monokalibrierung (Calib-Set)
    rms_l_mono_st, K_l_mono_st, dist_l_mono_st, rvecs_l_mono_st, tvecs_l_mono_st = calibrate_mono(img_points_l_cal, obj_points_cal, img_size)
    rms_r_mono_st, K_r_mono_st, dist_r_mono_st, rvecs_r_mono_st, tvecs_r_mono_st = calibrate_mono(img_points_r_cal, obj_points_cal, img_size)

    print(f"[mono] RMS links:  {rms_l_mono_st:.4f}")
    print(f"[mono] RMS rechts: {rms_r_mono_st:.4f}")

    errs_mono_reproj_initial_l_st = compute_mono_reproj_errors(img_points_l_cal, obj_points_cal, rvecs_l_mono_st, tvecs_l_mono_st, K_l_mono_st, dist_l_mono_st)
    errs_mono_reproj_initial_r_st = compute_mono_reproj_errors(img_points_r_cal, obj_points_cal, rvecs_r_mono_st, tvecs_r_mono_st, K_r_mono_st, dist_r_mono_st)
    if verbose:
        print("[dbg] Monoreprojection Left (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj_initial_l_st))
        print("[dbg] Monoreprojection Right (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj_initial_r_st))


    # --- [A] Paare mit hohem Mono-Fehler entfernen ------------------------------
    keepA, dropA, detailsA = filter_pairs_by_mono_errors(
        obj_points_cal, img_points_l_cal, img_points_r_cal,
        rvecs_l_mono_st, tvecs_l_mono_st, K_l_mono_st, dist_l_mono_st,
        rvecs_r_mono_st, tvecs_r_mono_st, K_r_mono_st, dist_r_mono_st,
        k_mad=2.5, max_cap_px=1.0  # k_mad 2.5..3.5 ist üblich
    )
    if len(dropA):
        print(f"[filter-mono] drop {len(dropA)} / {len(obj_points_cal)} (thr={detailsA['thr_mean']:.3f}px)")
        obj_points_cal = [obj_points_cal[i] for i in keepA]
        img_points_l_cal = [img_points_l_cal[i] for i in keepA]
        img_points_r_cal = [img_points_r_cal[i] for i in keepA]
        rvecs_l_mono_st = [rvecs_l_mono_st[i] for i in keepA]
        tvecs_l_mono_st = [tvecs_l_mono_st[i] for i in keepA]
        rvecs_r_mono_st = [rvecs_r_mono_st[i] for i in keepA]
        tvecs_r_mono_st = [tvecs_r_mono_st[i] for i in keepA]
    else:
        print("[filter-mono] keine Paare entfernt")

    # Final Mono Calibration
    rms_l_mono_st, K_l_mono_st, dist_l_mono_st, rvecs_l_mono_st, tvecs_l_mono_st = calibrate_mono(img_points_l_cal, obj_points_cal, img_size)
    rms_r_mono_st, K_r_mono_st, dist_r_mono_st, rvecs_r_mono_st, tvecs_r_mono_st = calibrate_mono(img_points_r_cal, obj_points_cal, img_size)

    print(f"[mono2] RMS links:  {rms_l_mono_st:.4f}")
    print(f"[mono2] RMS rechts: {rms_r_mono_st:.4f}")

    errs_mono_reproj_l_st = compute_mono_reproj_errors(img_points_l_cal, obj_points_cal, rvecs_l_mono_st, tvecs_l_mono_st, K_l_mono_st, dist_l_mono_st)
    errs_mono_reproj_r_st = compute_mono_reproj_errors(img_points_r_cal, obj_points_cal, rvecs_r_mono_st, tvecs_r_mono_st, K_r_mono_st, dist_r_mono_st)
    if verbose:
        print("[dbg2] Monoreprojection Left 2 (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj_l_st))
        print("[dbg2] Monoreprojection Right 2 (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**errs_mono_reproj_r_st))

    map1_l_mono_st, map2_l_mono_st = rectify_distortion(K_l_mono_st, dist_l_mono_st, img_size)
    map1_r_mono_st, map2_r_mono_st = rectify_distortion(K_r_mono_st, dist_r_mono_st, img_size)

    K_l = K_l_mono_st
    dist_l = dist_l_mono_st
    map1_l = map1_l_mono_st
    map2_l = map2_l_mono_st
    errs_mono_reproj_l = errs_mono_reproj_l_st
    errs_mono_reproj_initial_l = errs_mono_reproj_initial_l_st

    K_r = K_r_mono_st
    dist_r = dist_r_mono_st
    map1_r = map1_r_mono_st
    map2_r = map2_r_mono_st
    errs_mono_reproj_r = errs_mono_reproj_r_st
    errs_mono_reproj_initial_r = errs_mono_reproj_initial_r_st

    # Optional: Separate Monokalibrierung mit eigenen Bildern (falls left_dir_mono / right_dir_mono angegeben)
    if left_dir_mono is not None and right_dir_mono is not None:
        print("[info] Separate Monokalibrierung mit eigenen Bildern...")
        rms_l_mono, K_l_mono, dist_l_mono, rvecs_l_mono, tvecs_l_mono, map1_l_mono, map2_l_mono, errs_mono_reproj_initial_l_mono, errs_mono_reproj_l_mono = run_opencv_mono_calibration(left_dir_mono, pattern_size, square_size_mm, px_mm, verbose=verbose)
        rms_r_mono, K_r_mono, dist_r_mono, rvecs_r_mono, tvecs_r_mono, map1_r_mono, map2_r_mono, errs_mono_reproj_initial_r_mono, errs_mono_reproj_r_mono = run_opencv_mono_calibration(right_dir_mono, pattern_size, square_size_mm, px_mm, verbose=verbose)
        print(f"[mono-separat] RMS links:  {rms_l_mono:.4f}")
        print(f"[mono-separat] RMS rechts: {rms_r_mono:.4f}")
        
        # TODO maybe depending on RMS?
        K_l = K_l_mono
        dist_l = dist_l_mono
        map1_l = map1_l_mono
        map2_l = map2_l_mono
        errs_mono_reproj_l = errs_mono_reproj_l_mono
        errs_mono_reproj_initial_l = errs_mono_reproj_initial_l_mono

        K_r = K_r_mono
        dist_r = dist_r_mono
        map1_r = map1_r_mono
        map2_r = map2_r_mono
        errs_mono_reproj_r = errs_mono_reproj_r_mono
        errs_mono_reproj_initial_r = errs_mono_reproj_initial_r_mono

    # 4) Stereokalibrierung
    # --- Stereo #1 (mit fixierten Intrinsics) -----------------------------------
    rms_stereo_1, K_l, dist_l, K_r, dist_r, R, t, E, F = calibrate_stereo(obj_points_cal, img_points_l_cal, img_points_r_cal, img_size, K_l, dist_l, K_r, dist_r)
    print(f"[stereo#1] RMS: {rms_stereo_1:.4f}")

    # --- [B] Paare mit hohem Sampson-Fehler entfernen ---------------------------
    samps = per_view_sampson(F, img_points_l_cal, img_points_r_cal)
    thr_samps = robust_threshold(samps, k=3.0)
    
    keepB = np.where(samps <= thr_samps)[0].tolist()
    dropB = np.where(samps > thr_samps)[0].tolist()
    
    if len(dropB):
        print(f"[filter-sampson] drop {len(dropB)} / {len(obj_points_cal)} (thr={thr_samps:.3f}px)")
        
        obj_points_cal = [obj_points_cal[i] for i in keepB]
        img_points_l_cal = [img_points_l_cal[i] for i in keepB]
        img_points_r_cal = [img_points_r_cal[i] for i in keepB]
        
        # Stereo #2 – final
        rms_stereo_2, K_l, dist_l, K_r, dist_r, R, t, E, F = calibrate_stereo(obj_points_cal, img_points_l_cal, img_points_r_cal, img_size, K_l, dist_l, K_r, dist_r)

        print(f"[stereo#2] RMS: {rms_stereo_2:.4f}  (vorher {rms_stereo_1:.4f})")
        rms_stereo = rms_stereo_2
    else:
        print("[filter-sampson] keine Paare entfernt")
        rms_stereo = rms_stereo_1

    # epi_hold = compute_stereo_epipolar_error(img_points_l_hold, img_points_r_hold, F) if hold_idx else None
    # rect_hold = vertical_disparity_stats(img_points_l_hold, img_points_r_hold, K_l, dist_l, K_r, dist_r, R, T, img_size) if hold_idx else None

    print(f"[stereo] RMS (refit, filtered & fix-intrinsic): {rms_stereo:.4f}")
    # if verbose:
    #     print("[dbg] Epipolar (Sampson) (px): mean={mean_px:.3f}, std={std_px:.3f}, max={max_px:.3f}".format(**epi_hold))

    # 5) Welt = linke Kamera
    R_wc_l = np.eye(3, dtype=float)
    t_wc_l = np.zeros((3, 1), dtype=float)
    C2_w = -R.T @ t.reshape(3, 1)
    R_wc_r = R.T
    t_wc_r = C2_w

    # 6) Rectify
    ## Vertical Disparity is disabled because cameras are not aligned
    # R1, R2, P1, P2, Q, roi1, roi2, map1_l, map2_l, map1_r, map2_r = stereo_rectify(
    #     K1, dist1, K2, dist2, img_size, R, T, alpha=float(rectify_alpha)
    # )

    # Type checks
    for M, name in [(K_r, "K_r"), (K_l, "K_l"), (dist_r, "dist_r"), (dist_l, "dist_l"), (R, "R"), (t, "t")]:
        assert isinstance(M, np.ndarray), f"{name} must be numpy.ndarray, got {type(M)}"
        assert M.dtype in (np.float32, np.float64), f"{name}.dtype must be float32/64, got {M.dtype}"

    ## Vertical Disparity is disabled because cameras are not aligned
    # Robustes Printing
    # def _sf(x):
    #     return f"{x:.3f}" if x is not None else "NaN"

    # if epi_hold:
    #     print(f"[hold-out] Sampson (px): mean={_sf(epi_hold['mean_px'])}, "
    #           f"std={_sf(epi_hold['std_px'])}, max={_sf(epi_hold['max_px'])}, n={len(epi_hold['per_view'])}")
    # if rect_hold and rect_hold.get("mean_abs_px") is not None:
    #     print(f"[hold-out] Vertikale Disparität |dy| (px): mean_abs={_sf(rect_hold['mean_abs_px'])}, "
    #           f"std={_sf(rect_hold['std_px'])}, max_abs={_sf(rect_hold['max_abs_px'])}, n={rect_hold['n']}")
    # else:
    #     print("[hold-out] Vertikale Disparität: keine gültigen Werte (keine Ecken/NaN)")
        
    # 7) CameraParams
    camera_params_0 = build_camparams_from_K(K_l, dist_l, map1_l, map2_l, img_size, R_wc_l, t_wc_l, (px_mm, py_mm), MonoCalibrationStatistics(errs_mono_reproj_initial=errs_mono_reproj_initial_l, errs_mono_reproj=errs_mono_reproj_l))
    camera_params_1 = build_camparams_from_K(K_r, dist_r, map1_r, map2_r, img_size, R_wc_r, t_wc_r, (px_mm, py_mm), MonoCalibrationStatistics(errs_mono_reproj_initial=errs_mono_reproj_initial_r, errs_mono_reproj=errs_mono_reproj_r))
    
    results = initialize_stereo_calibration_results(camera_params_0, camera_params_1, R, t, E, F)

    # 9) Beispiel: eine rektifizierte Stichprobe speichern
    # try:
    #     save_rectified_previews_with_guides(
    #         pairs=pairs,
    #         K1=K_l, dist1=dist_l, K2=K_r, dist2=dist_r, R=R, T=T,
    #         image_size=img_size,  # ACHTUNG: (w,h)!
    #         out_dir=out_dir,
    #         count=2, step=40, alpha=0.0
    #     )

    #     print("[ok] Beispiel-Rectify gespeichert (rectified_left_sample.png / rectified_right_sample.png)")
    # except Exception as e:
    #     print(f"[warn] Beispiel-Rectify fehlgeschlagen: {e}")

    return results

