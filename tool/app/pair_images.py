import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


# robust gegen "1746640.7596404." (trailing Punkt vor Dateiendung)
FNAME_RE = re.compile(r"frame_(\d+).*?timestamp_([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _parse_name(fname: str) -> Tuple[Optional[int], Optional[float]]:
    """
    Versucht frame-ID und Timestamp (float Sekunden, relativ) zu parsen.
    Gibt (frame_id, ts_sec) zurück, jeweils None falls nicht gefunden.
    """
    m = FNAME_RE.search(fname)
    if m:
        fid = int(m.group(1))
        # group(2) ist exakt ohne trailing '.', dank Regex
        ts = float(m.group(2))
        return fid, ts

    # Fallback: nur frame_(\d+) vorhanden?
    m2 = re.search(r"frame_(\d+)", fname, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), None
    return None, None

def _list_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def _extract_frame_id(name: str) -> Optional[int]:
    """
    Extrahiert die frame-ID aus einem Dateinamen.
    """
    FRAME_RE = re.compile(r"frame_(\d+)[^-]*-timestamp_", re.IGNORECASE)
    m = FRAME_RE.search(name)
    if m:
        return int(m.group(1))
    return None


def pair_stereo_images(left_dir: Path, right_dir: Path, verbose: bool=False) -> List[Tuple[Path, Path]]:
    left_imgs = _list_images(left_dir)
    right_imgs = _list_images(right_dir)
    left_map = {}
    right_map = {}
    for p in left_imgs:
        fid = _extract_frame_id(p.name)
        if fid is not None:
            left_map[fid] = p
        elif verbose:
            print(f"[dbg] left: Name passt nicht zum Pattern -> {p.name}")
    for p in right_imgs:
        fid = _extract_frame_id(p.name)
        if fid is not None:
            right_map[fid] = p
        elif verbose:
            print(f"[dbg] right: Name passt nicht zum Pattern -> {p.name}")
    common = sorted(set(left_map.keys()) & set(right_map.keys()))
    pairs = [(left_map[fid], right_map[fid]) for fid in common]
    if not pairs:
        raise RuntimeError("Keine passenden Bildpaare gefunden. Prüfe Dateinamen und Ordner." )
    missing_l = sorted(set(right_map.keys()) - set(left_map.keys()))
    missing_r = sorted(set(left_map.keys()) - set(right_map.keys()))
    if missing_l:
        print(f"[warn] {len(missing_l)} Frames nur rechts vorhanden (werden ignoriert), z.B.: {missing_l[:5]}")
    if missing_r:
        print(f"[warn] {len(missing_r)} Frames nur links vorhanden (werden ignoriert), z.B.: {missing_r[:5]}")
    print(f"[info] Gefundene Stereo-Paare: {len(pairs)}")
    return pairs


def pair_stereo_images_smart(
    left_dir: Path,
    right_dir: Path,
    max_dt_ms: float = 2.0,
    allow_cross_id: bool = True,
    verbose: bool = True,
) -> List[Tuple[Path, Path]]:
    """
    Bildpaare finden mit Timestamp-Toleranz:
    1) gleiche frame-ID + Δt ≤ max_dt_ms  (bevorzugt)
    2) wenn zu wenige Paare, optional Cross-ID 1:1-Matching mit minimalem Δt (≤ max_dt_ms)
    3) wenn keine Timestamps gefunden → Fallback: Frame-ID Matching ohne Timestamps

    Rückgabe: Liste [(left_path, right_path), ...], nach left frame-ID sortiert (falls vorhanden).
    """
    
    L = [(p, *_parse_name(p.name)) for p in _list_images(left_dir)]
    R = [(p, *_parse_name(p.name)) for p in _list_images(right_dir)]

    # Split nach: mit/ohne Timestamp
    L_ts = [(p, fid, ts) for (p, fid, ts) in L if ts is not None and fid is not None]
    R_ts = [(p, fid, ts) for (p, fid, ts) in R if ts is not None and fid is not None]
    L_id = {fid: p for (p, fid, ts) in L if fid is not None}
    R_id = {fid: p for (p, fid, ts) in R if fid is not None}

    pairs: List[Tuple[Path, Path]] = []
    used_right = set()
    dt_thresh = max_dt_ms / 1000.0  # in Sekunden

    # ---------- A) gleiche frame-ID, Δt <= max_dt_ms ----------
    if L_ts and R_ts:
        R_by_id = {}
        for p, fid, ts in R_ts:
            R_by_id.setdefault(fid, []).append((p, ts))
        kept_A = 0
        dropped_out_of_dt = 0
        for pL, fidL, tsL in L_ts:
            if fidL in R_by_id:
                # nimm rechte mit minimalem Δt
                cand = min(R_by_id[fidL], key=lambda x: abs(tsL - x[1]))
                pR, tsR = cand
                if abs(tsL - tsR) <= dt_thresh and pR not in used_right:
                    pairs.append((pL, pR))
                    used_right.add(pR)
                    kept_A += 1
                else:
                    dropped_out_of_dt += 1
        if verbose:
            print(f"[pair] gleiche frame-ID & Δt≤{max_dt_ms}ms: keep={kept_A}, drop_dt={dropped_out_of_dt}")

    # ---------- B) Cross-ID (optional), falls noch wenig Paare ----------
    # (z. B. wenn Kameras nicht exakt gemeinsam zählen)
    if allow_cross_id and L_ts and R_ts:
        # Rechte, die noch frei sind:
        free_R = [(p, fid, ts) for (p, fid, ts) in R_ts if p not in used_right]
        if verbose:
            print(f"[pair] cross-id phase: freie R={len(free_R)}; bereits gepaart={len(pairs)}")
        # Greedy: für jede linke wähle rechtes mit minimalem Δt, 1:1, falls unter Schwelle
        added_B = 0
        right_taken = set()
        for pL, fidL, tsL in L_ts:
            # wenn pL bereits gepaart? (über frame-ID passiert implizit nicht)
            # Suche freien R mit min Δt
            best = None
            best_dt = None
            for idx, (pR, fidR, tsR) in enumerate(free_R):
                if pR in used_right or idx in right_taken:
                    continue
                dt = abs(tsL - tsR)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, (idx, pR)
            if best is not None and best_dt <= dt_thresh:
                idx, pR = best
                if (pL, pR) not in pairs:
                    pairs.append((pL, pR))
                    used_right.add(pR)
                    right_taken.add(idx)
                    added_B += 1
        if verbose:
            print(f"[pair] cross-id ergänzt: +{added_B} Paare")
    
    # ---------- B2) YYYY_MM_DD_HH_MM_DD Suffix Matching (optional) ----------
    if allow_cross_id and not pairs:
        def _parse_datetime_suffix(name: str) -> Optional[str]:
            m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", name)
            if m:
                return m.group(1)
            return None

        L_dt = {}
        R_dt = {}
        for p, fid, ts in L:
            dt_suff = _parse_datetime_suffix(p.name)
            if dt_suff is not None:
                L_dt[dt_suff] = p
        for p, fid, ts in R:
            dt_suff = _parse_datetime_suffix(p.name)
            if dt_suff is not None:
                R_dt[dt_suff] = p
        common_dt = sorted(set(L_dt.keys()) & set(R_dt.keys()))
        added_B2 = 0
        for dt_suff in common_dt:
            pL = L_dt[dt_suff]
            pR = R_dt[dt_suff]
            if (pL, pR) not in pairs:
                pairs.append((pL, pR))
                added_B2 += 1
        if verbose and added_B2 > 0:
            print(f"[pair] datetime-suffix ergänzt: +{added_B2} Paare")

    # ---------- C) Fallback: kein Timestamp nutzbar → Frame-ID Matching ----------
    if not pairs:
        if verbose:
            print("[pair] Fallback: keine/nur spärliche Timestamps erkannt → Frame-ID Matching ohne Δt.")
        common = sorted(set(L_id.keys()) & set(R_id.keys()))
        for fid in common:
            pairs.append((L_id[fid], R_id[fid]))
        if verbose:
            print(f"[pair] Fallback-Paare: {len(pairs)}")

    # ---------- Aufräumen: sortiert zurückgeben ----------
    def key_fn(pLR: Tuple[Path, Path]) -> int:
        fid, _ = _parse_name(pLR[0].name)
        return fid if fid is not None else 0

    pairs_sorted = sorted(pairs, key=key_fn)

    # ---------- Reporting ----------
    if verbose:
        # Δt-Statistik nur für Paare mit Timestamp
        dts_ms = []
        for pL, pR in pairs_sorted:
            _, fidL, tsL = pL, *_parse_name(pL.name)
            _, fidR, tsR = pR, *_parse_name(pR.name)
            if tsL is not None and tsR is not None:
                dts_ms.append(abs(tsL - tsR) * 1000.0)
        if dts_ms:
            dts = np.array(dts_ms, float)
            print("[pair] Δt-Statistik (ms): mean={:.3f}, std={:.3f}, max={:.3f}, n={}".format(
                dts.mean(), dts.std(), dts.max(), dts.size
            ))
        print(f"[info] Gefundene Stereo-Paare: {len(pairs_sorted)} (smart pairing)")

    if not pairs_sorted:
        raise RuntimeError("Keine passenden Bildpaare gefunden (auch nicht mit Timestamp/Cross-ID).")

    return pairs_sorted