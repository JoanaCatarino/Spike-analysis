# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 18:05:23 2026

@author: JoanaCatarino
"""

import numpy as np
import pandas as pd


camdata_path= 'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day3/999770_day3_camdata2025-12-06T10_04_09.csv'
ttl_path= 'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day3/999770_day32025-12-06T10_04_09.csv'

camdata_path= 'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day1/999770_day1_1_camdata2025-11-11T14_46_38.csv'
ttl_path= 'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day1/999770_day1_12025-11-11T14_46_38.csv'


# --- load camera frames (ground truth) ---
cam = pd.read_csv(camdata_path)
cam["frame_ts"] = pd.to_datetime(cam["Item3"], utc=True, errors="coerce")
cam = cam.dropna(subset=["frame_ts"]).reset_index(drop=True)

# frame index should be 0..275699
frame_idx= cam["Item2"].to_numpy()
frame_ts= cam["frame_ts"].to_numpy(dtype="datetime64[ns]")

print("Frames in camdata:", len(cam))

# --- load TTL file and extract TTL edge timestamps ---
ttl = pd.read_csv(ttl_path)
ttl_ts = pd.to_datetime(ttl["Item3"], utc=True, errors="coerce")
ttl_ts = ttl_ts.dropna().reset_index(drop=True)

# edges = whenever Item3 changes
edge_ts = ttl_ts[ttl_ts.ne(ttl_ts.shift(1))].reset_index(drop=True)
edge_ts_np = edge_ts.to_numpy(dtype="datetime64[ns]")

print("TTL edges detected:", len(edge_ts))

# --- map each edge to the NEXT frame (ceiling) ---
# (if edge occurs between frames, it will be seen at the next frame)
pos = np.searchsorted(frame_ts, edge_ts_np, side="left")
pos = np.clip(pos, 0, len(frame_ts) - 1)

edges = pd.DataFrame({
    "edge_number": np.arange(len(edge_ts)),
    "ttl_edge_time": edge_ts.values,
    "frame_number": frame_idx[pos],
    "frame_time": pd.to_datetime(frame_ts[pos], utc=True),
})

# label alternating UP/DOWN (flip if needed)
edges["ttl_edge"] = np.where(edges["edge_number"] % 2 == 0, "UP", "DOWN")