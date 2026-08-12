# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:50:13 2026

@author: JoanaCatarino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map video frames into imec time using blue LED TTL edges.

Pipeline
--------
1. Load camdata CSV (frame timestamps).
2. Load video TTL CSV (LED state / timestamps extracted from video) and detect edges.
3. Map each video LED edge to the nearest visible frame (NEXT frame by default).
4. Load aligned_ttls.pkl (nidq -> imec already done).
5. Use blue LED edges from aligned_ttls.pkl as imec anchors.
6. Sequentially pair video edges <-> imec edges.
7. Build a piecewise-linear mapper: video_time_ns -> imec_sample.
8. Convert every video frame to imec samples and seconds.
9. Save outputs + QC plots.

Outputs
-------
<save_dir>/
    video_frames_in_imec.csv
    video_led_edges_paired.csv
    video_imec_mapping_qc.png
    video_edge_pairing_qc.png
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# =============================================================================
#%% 0. Config
# =============================================================================

# --- input files -------------------------------------------------------------
camdata_path = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day3/999770_day3_camdata2025-12-06T10_04_09.csv'
)

ttl_path = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day3/999770_day32025-12-06T10_04_09.csv'
)


aligned_ttls_path = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/animals/999770/sessions/20251206_adaptivesensorimotortask_rec/sync/aligned_ttls.pkl'
)
# example:
# aligned_ttls_path = Path(
#     r"L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/999770_20251111/sync/aligned_ttls.pkl"
# )

save_dir = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/animals/999770/sessions/20251206_adaptivesensorimotortask_rec/video_imec_alignment'
)

# --- which imec anchor to use ------------------------------------------------
# Preferred: "blue_led" (analog XA0 mapped to imec)
# Alternative: "bit4"    (digital bit4 mapped to imec)
USE_IMEC_SOURCE = "blue_led"   # "blue_led" or "bit4"

# --- video edge -> frame assignment ------------------------------------------
# If LED changes between frames, it first becomes visible in the NEXT frame.
VIDEO_EDGE_TO_FRAME = "next"   # "next" or "previous" or "nearest"

# --- mismatch handling --------------------------------------------------------
# If counts differ, trim to shorter.
TRIM_TO_SHORTER = True

# --- plotting ----------------------------------------------------------------
DPI = 150


# =============================================================================
#%% 1. Helpers
# =============================================================================

def load_camdata(camdata_csv: Path) -> pd.DataFrame:
    """
    Load camera metadata CSV.

    Expects:
        Item2 = frame number
        Item3 = timestamp
    """
    cam = pd.read_csv(camdata_csv)

    cam["frame_ts"] = pd.to_datetime(cam["Item3"], utc=True, errors="coerce")
    cam = cam.dropna(subset=["frame_ts"]).reset_index(drop=True)

    out = pd.DataFrame({
        "frame_number": cam["Item2"].to_numpy(),
        "frame_ts": cam["frame_ts"].values,
    })
    out["frame_ts_ns"] = out["frame_ts"].astype("int64")
    return out


def load_video_ttl_edges(ttl_csv: Path) -> pd.DataFrame:
    """
    Load the video-derived TTL CSV and detect edges.

    Your original script detects an edge whenever Item3 changes.
    This function preserves that logic.
    """
    ttl = pd.read_csv(ttl_csv)

    ttl_ts = pd.to_datetime(ttl["Item3"], utc=True, errors="coerce")
    ttl_ts = ttl_ts.dropna().reset_index(drop=True)

    edge_ts = ttl_ts[ttl_ts.ne(ttl_ts.shift(1))].reset_index(drop=True)

    edges = pd.DataFrame({
        "edge_number": np.arange(len(edge_ts), dtype=int),
        "video_edge_time": edge_ts.values,
    })
    edges["video_edge_time_ns"] = edges["video_edge_time"].astype("int64")
    edges["edge_type"] = np.where(edges["edge_number"] % 2 == 0, "UP", "DOWN")
    return edges


def assign_edges_to_frames(edges: pd.DataFrame, cam: pd.DataFrame, mode="next") -> pd.DataFrame:
    """
    Assign each video edge to a frame.

    mode:
        "next"     -> first frame at or after edge time
        "previous" -> last frame at or before edge time
        "nearest"  -> nearest frame in time
    """
    frame_ts_ns = cam["frame_ts_ns"].to_numpy()
    frame_nums = cam["frame_number"].to_numpy()
    edge_ts_ns = edges["video_edge_time_ns"].to_numpy()

    if mode == "next":
        pos = np.searchsorted(frame_ts_ns, edge_ts_ns, side="left")
        pos = np.clip(pos, 0, len(frame_ts_ns) - 1)

    elif mode == "previous":
        pos = np.searchsorted(frame_ts_ns, edge_ts_ns, side="right") - 1
        pos = np.clip(pos, 0, len(frame_ts_ns) - 1)

    elif mode == "nearest":
        pos_right = np.searchsorted(frame_ts_ns, edge_ts_ns, side="left")
        pos_right = np.clip(pos_right, 0, len(frame_ts_ns) - 1)
        pos_left = np.clip(pos_right - 1, 0, len(frame_ts_ns) - 1)

        d_right = np.abs(frame_ts_ns[pos_right] - edge_ts_ns)
        d_left = np.abs(frame_ts_ns[pos_left] - edge_ts_ns)
        pos = np.where(d_left <= d_right, pos_left, pos_right)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = edges.copy()
    out["frame_number"] = frame_nums[pos]
    out["frame_time"] = pd.to_datetime(frame_ts_ns[pos], utc=True)
    out["frame_time_ns"] = frame_ts_ns[pos]
    out["edge_to_frame_delay_ms"] = (out["frame_time_ns"] - out["video_edge_time_ns"]) / 1e6
    return out


def get_imec_led_edges(aligned: dict, source="blue_led") -> pd.DataFrame:
    """
    Extract imec LED edges from aligned_ttls.pkl.

    source:
        "blue_led" -> aligned["blue_led"]
        "bit4"     -> aligned["bits"]["bit4"]
    """
    if source == "blue_led":
        if "blue_led" not in aligned:
            raise KeyError("aligned_ttls.pkl does not contain key 'blue_led'")
        src = aligned["blue_led"]

    elif source == "bit4":
        if "bits" not in aligned or "bit4" not in aligned["bits"]:
            raise KeyError("aligned_ttls.pkl does not contain aligned['bits']['bit4']")
        src = aligned["bits"]["bit4"]

    else:
        raise ValueError("source must be 'blue_led' or 'bit4'")

    up = np.asarray(src["up_imec"], dtype=np.int64)
    down = np.asarray(src["down_imec"], dtype=np.int64)
    n = min(len(up), len(down))

    # Flatten into one chronological edge list: UP0, DOWN0, UP1, DOWN1, ...
    imec_edges = pd.DataFrame({
        "edge_number": np.arange(2 * n, dtype=int),
        "edge_type": np.array(["UP", "DOWN"] * n),
        "imec_sample": np.ravel(np.column_stack([up[:n], down[:n]])).astype(np.int64),
    })

    return imec_edges


def pair_video_edges_to_imec(video_edges: pd.DataFrame,
                             imec_edges: pd.DataFrame,
                             trim_to_shorter=True) -> pd.DataFrame:
    """
    Sequentially pair video LED edges to imec LED edges.
    """
    n_video = len(video_edges)
    n_imec = len(imec_edges)

    if n_video != n_imec:
        msg = f"Edge count mismatch: video={n_video}, imec={n_imec}"
        if not trim_to_shorter:
            raise ValueError(msg)
        print(f"*** {msg} — trimming to shorter")

    n = min(n_video, n_imec)

    v = video_edges.iloc[:n].reset_index(drop=True).copy()
    i = imec_edges.iloc[:n].reset_index(drop=True).copy()

    paired = pd.DataFrame({
        "pair_index": np.arange(n, dtype=int),

        "video_edge_number": v["edge_number"].values,
        "video_edge_type": v["edge_type"].values,
        "video_edge_time": v["video_edge_time"].values,
        "video_edge_time_ns": v["video_edge_time_ns"].values,
        "frame_number": v["frame_number"].values,
        "frame_time": v["frame_time"].values,
        "frame_time_ns": v["frame_time_ns"].values,
        "edge_to_frame_delay_ms": v["edge_to_frame_delay_ms"].values,

        "imec_edge_number": i["edge_number"].values,
        "imec_edge_type": i["edge_type"].values,
        "imec_sample": i["imec_sample"].values,
    })

    type_match = paired["video_edge_type"].values == paired["imec_edge_type"].values
    if not np.all(type_match):
        mismatch_idx = np.where(~type_match)[0]
        print(f"*** WARNING: {len(mismatch_idx)} edge-type mismatches found")
        print(f"    first mismatches at pairs: {mismatch_idx[:10]}")

    paired["edge_type_match"] = type_match
    return paired


def build_video_to_imec_mapper(video_time_ns: np.ndarray,
                               imec_samples: np.ndarray):
    """
    Build a piecewise-linear interpolation:
        video timestamp (ns) -> imec sample
    """
    x = np.asarray(video_time_ns, dtype=np.float64)
    y = np.asarray(imec_samples, dtype=np.float64)

    # drop duplicate x if needed
    keep = np.concatenate([[True], np.diff(x) != 0])
    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        raise ValueError("Need at least 2 unique anchor points to build mapper")

    mapper = interp1d(
        x,
        y,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    return mapper


def map_all_frames(cam: pd.DataFrame, mapper, imec_fs: float) -> pd.DataFrame:
    """
    Map every camera frame timestamp into imec samples.
    """
    out = cam.copy()
    out["imec_sample"] = np.round(mapper(out["frame_ts_ns"].to_numpy())).astype(np.int64)
    out["imec_time_s"] = out["imec_sample"] / float(imec_fs)
    return out


def plot_pairing_qc(paired: pd.DataFrame, imec_fs: float, save_path: Path):
    """
    QC for edge pairing and monotonicity.
    """
    if len(paired) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x_rel_s = (paired["frame_time_ns"].to_numpy() - paired["frame_time_ns"].iloc[0]) / 1e9
    y_s = paired["imec_sample"].to_numpy() / imec_fs

    axes[0].scatter(x_rel_s, y_s, s=6, alpha=0.7)
    axes[0].set_xlabel("video edge frame time (relative s)")
    axes[0].set_ylabel("imec edge time (s)")
    axes[0].set_title("Video-edge time vs imec-edge time")

    c = np.polyfit(x_rel_s, y_s, 1)
    pred = np.polyval(c, x_rel_s)
    residual_ms = (y_s - pred) * 1000

    axes[1].plot(residual_ms, ".", ms=3)
    axes[1].axhline(0, color="gray", lw=0.5, ls="--")
    axes[1].set_xlabel("paired edge index")
    axes[1].set_ylabel("residual (ms)")
    axes[1].set_title(f"Pairing residuals — RMS={np.std(residual_ms):.2f} ms")

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_frame_mapping_qc(frames_imec: pd.DataFrame, imec_fs: float, save_path: Path):
    """
    QC for full-frame mapping.
    """
    if len(frames_imec) < 3:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))

    frame_ts_rel = (frames_imec["frame_ts_ns"].to_numpy() - frames_imec["frame_ts_ns"].iloc[0]) / 1e9
    imec_time_s = frames_imec["imec_sample"].to_numpy() / imec_fs

    axes[0].plot(frame_ts_rel, imec_time_s, lw=0.8)
    axes[0].set_xlabel("camera time (relative s)")
    axes[0].set_ylabel("imec time (s)")
    axes[0].set_title("All frames mapped into imec time")

    frame_dt_video_ms = np.diff(frames_imec["frame_ts_ns"].to_numpy()) / 1e6
    frame_dt_imec_ms = np.diff(frames_imec["imec_sample"].to_numpy()) / imec_fs * 1000

    axes[1].plot(frame_dt_video_ms, lw=0.7, label="video dt (ms)")
    axes[1].plot(frame_dt_imec_ms, lw=0.7, label="mapped imec dt (ms)")
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("frame interval (ms)")
    axes[1].set_title("Frame interval consistency")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
#%% 2. Load data
# =============================================================================

save_dir.mkdir(parents=True, exist_ok=True)

print("Loading camera frame timestamps...")
cam = load_camdata(camdata_path)
print(f"  Frames in camdata: {len(cam)}")
print(f"  First frame: {cam['frame_number'].iloc[0]}")
print(f"  Last frame : {cam['frame_number'].iloc[-1]}")

print("\nLoading video TTL edges...")
video_edges = load_video_ttl_edges(ttl_path)
print(f"  Video edges detected: {len(video_edges)}")

print(f"\nAssigning video edges to frames ({VIDEO_EDGE_TO_FRAME})...")
video_edges = assign_edges_to_frames(video_edges, cam, mode=VIDEO_EDGE_TO_FRAME)

with open(aligned_ttls_path, "rb") as f:
    aligned = pickle.load(f)

imec_fs = float(aligned["imec_sample_rate"])
nidq_fs = float(aligned["nidq_sample_rate"])

print(f"\nLoaded aligned_ttls.pkl")
print(f"  imec fs: {imec_fs} Hz")
print(f"  nidq fs: {nidq_fs} Hz")

print(f"\nExtracting imec LED edges from: {USE_IMEC_SOURCE}")
imec_edges = get_imec_led_edges(aligned, source=USE_IMEC_SOURCE)
print(f"  IMEC edges available: {len(imec_edges)}")


# =============================================================================
#%% 3. Pair anchors
# =============================================================================

paired = pair_video_edges_to_imec(
    video_edges=video_edges,
    imec_edges=imec_edges,
    trim_to_shorter=TRIM_TO_SHORTER,
)

print(f"\nPaired anchors: {len(paired)}")

n_type_match = int(paired["edge_type_match"].sum())
print(f"  Edge-type matches: {n_type_match}/{len(paired)}")

if len(paired) < 2:
    raise RuntimeError("Not enough paired edges to build mapper")

print("\nVideo edge -> frame delay:")
print(f"  median = {paired['edge_to_frame_delay_ms'].median():.2f} ms")
print(f"  min    = {paired['edge_to_frame_delay_ms'].min():.2f} ms")
print(f"  max    = {paired['edge_to_frame_delay_ms'].max():.2f} ms")


# =============================================================================
#%% 4. Build video_time -> imec mapper
# =============================================================================

# IMPORTANT:
# Use frame_time_ns, not raw video_edge_time_ns, because the LED is visible on the frame.
mapper_video_to_imec = build_video_to_imec_mapper(
    video_time_ns=paired["frame_time_ns"].to_numpy(),
    imec_samples=paired["imec_sample"].to_numpy(),
)

print("\nMapper built successfully")
print(f"  Anchor span (video): "
      f"{(paired['frame_time_ns'].iloc[-1] - paired['frame_time_ns'].iloc[0]) / 1e9:.2f} s")
print(f"  Anchor span (imec):  "
      f"{(paired['imec_sample'].iloc[-1] - paired['imec_sample'].iloc[0]) / imec_fs:.2f} s")


# =============================================================================
#%% 5. Map all video frames to imec
# =============================================================================

frames_imec = map_all_frames(cam, mapper_video_to_imec, imec_fs)

print("\nAll frames converted to imec")
print(f"  Mapped frames: {len(frames_imec)}")
print(f"  First imec sample: {frames_imec['imec_sample'].iloc[0]}")
print(f"  Last imec sample : {frames_imec['imec_sample'].iloc[-1]}")

# Optional frame interval summaries
video_dt_ms = np.diff(frames_imec["frame_ts_ns"].to_numpy()) / 1e6
imec_dt_ms = np.diff(frames_imec["imec_sample"].to_numpy()) / imec_fs * 1000

print("\nFrame interval summary:")
print(f"  Video dt  median={np.median(video_dt_ms):.3f} ms   std={np.std(video_dt_ms):.3f} ms")
print(f"  IMEC dt   median={np.median(imec_dt_ms):.3f} ms   std={np.std(imec_dt_ms):.3f} ms")


# =============================================================================
#%% 6. Save outputs
# =============================================================================

paired_csv = save_dir / "video_led_edges_paired.csv"
frames_csv = save_dir / "video_frames_in_imec.csv"
qc_pair_png = save_dir / "video_edge_pairing_qc.png"
qc_map_png = save_dir / "video_imec_mapping_qc.png"
mapper_pkl = save_dir / "video_to_imec_mapper_anchors.pkl"

paired.to_csv(paired_csv, index=False)
frames_imec.to_csv(frames_csv, index=False)

with open(mapper_pkl, "wb") as f:
    pickle.dump({
        "camdata_path": str(camdata_path),
        "ttl_path": str(ttl_path),
        "aligned_ttls_path": str(aligned_ttls_path),
        "imec_sample_rate": imec_fs,
        "nidq_sample_rate": nidq_fs,
        "imec_source_used": USE_IMEC_SOURCE,
        "video_edge_to_frame_mode": VIDEO_EDGE_TO_FRAME,
        "paired_edges": paired,
    }, f)

plot_pairing_qc(paired, imec_fs, qc_pair_png)
plot_frame_mapping_qc(frames_imec, imec_fs, qc_map_png)

print(f"\nSaved:")
print(f"  {paired_csv}")
print(f"  {frames_csv}")
print(f"  {mapper_pkl}")
print(f"  {qc_pair_png}")
print(f"  {qc_map_png}")


# =============================================================================
#%% 7. Example: how to use the frame table later
# =============================================================================
# For any behavior event already in imec samples:
#   - find nearest video frame
#
# Example:
# event_imec = beh.loc[123, "stim_imec_up"]
# idx = np.argmin(np.abs(frames_imec["imec_sample"].to_numpy() - event_imec))
# nearest_frame = frames_imec.iloc[idx]
# print(nearest_frame[["frame_number", "frame_ts", "imec_sample", "imec_time_s"]])