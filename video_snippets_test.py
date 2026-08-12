# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 13:36:46 2026

@author: JoanaCatarino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# =============================================================================
#%% Config
# =============================================================================

video_path = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Video/999770/day3/999770_day32025-12-06T10_04_09.avi'
)

behavior_csv = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/videos/behavior_paired_with_video_frames.csv'
)

save_dir = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/videos'
)

save_dir.mkdir(parents=True, exist_ok=True)

# Example choices
TRIAL_NUMBER = 10
ALIGN_TO = "stim_frame"       # e.g. "stim_frame", "trial_start_frame", "lick_frame"
FRAMES_BEFORE = 30
FRAMES_AFTER = 120

# Alternative: use whole trial
USE_WHOLE_TRIAL = False


# =============================================================================
#%% Helpers
# =============================================================================

def get_video_info(video_file: Path):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, n_frames, width, height


def write_video_clip(video_file: Path, output_file: Path, start_frame: int, end_frame: int):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_frame))
    end_frame = min(n_frames - 1, int(end_frame))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    f = start_frame

    while f <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        f += 1

    writer.release()
    cap.release()


# =============================================================================
#%% Main
# =============================================================================

beh = pd.read_csv(behavior_csv)
row = beh.loc[beh["trial_number"] == TRIAL_NUMBER].iloc[0]

fps, n_frames, width, height = get_video_info(video_path)
print(f"Video: fps={fps:.3f}, n_frames={n_frames}, size={width}x{height}")

if USE_WHOLE_TRIAL:
    start_frame = int(row["trial_start_frame"])
    end_frame = int(row["trial_end_frame"])
    out_name = f"trial_{TRIAL_NUMBER:04d}_whole_trial.mp4"
else:
    center = int(row[ALIGN_TO])
    start_frame = center - FRAMES_BEFORE
    end_frame = center + FRAMES_AFTER
    out_name = f"trial_{TRIAL_NUMBER:04d}_{ALIGN_TO}_minus{FRAMES_BEFORE}_plus{FRAMES_AFTER}.mp4"

output_file = save_dir / out_name
write_video_clip(video_path, output_file, start_frame, end_frame)

print(f"Saved -> {output_file}")
print(f"Frames: {start_frame} to {end_frame}")