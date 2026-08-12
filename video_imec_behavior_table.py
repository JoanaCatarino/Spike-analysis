# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:15:10 2026

@author: JoanaCatarino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
#%% Config
# =============================================================================

behavior_csv = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/animals/999770/sessions/20251206_adaptivesensorimotortask_rec/behavior/behavior_paired.csv'
)

video_frames_csv = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Analysis/animals/999770/sessions/20251206_adaptivesensorimotortask_rec/video_imec_alignment/video_frames_in_imec.csv'
)

output_csv = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/videos/behavior_paired_with_video_frames.csv'
)


# =============================================================================
#%% Helpers
# =============================================================================

def nearest_frame_index(frame_samples: np.ndarray, event_sample: float) -> int:
    """Return index of nearest frame to an imec event sample."""
    return int(np.argmin(np.abs(frame_samples - event_sample)))


def map_event_column_to_frame(beh: pd.DataFrame,
                              frames: pd.DataFrame,
                              event_col: str,
                              prefix: str = None) -> pd.DataFrame:
    """
    For one imec event column, add:
        <prefix>_frame
        <prefix>_frame_imec
        <prefix>_frame_time_s
    """
    if prefix is None:
        prefix = event_col.replace("_imec_up", "").replace("_imec_down", "").replace("_imec", "")

    frame_samples = frames["imec_sample"].to_numpy()
    frame_numbers = frames["frame_number"].to_numpy()
    frame_time_s = frames["imec_time_s"].to_numpy()

    out_frame = np.full(len(beh), np.nan)
    out_frame_imec = np.full(len(beh), np.nan)
    out_frame_time_s = np.full(len(beh), np.nan)

    valid = beh[event_col].notna()

    for i in np.where(valid)[0]:
        ev = float(beh.iloc[i][event_col])
        idx = nearest_frame_index(frame_samples, ev)
        out_frame[i] = frame_numbers[idx]
        out_frame_imec[i] = frame_samples[idx]
        out_frame_time_s[i] = frame_time_s[idx]

    beh[f"{prefix}_frame"] = out_frame
    beh[f"{prefix}_frame_imec"] = out_frame_imec
    beh[f"{prefix}_frame_time_s"] = out_frame_time_s

    return beh


# =============================================================================
#%% Main
# =============================================================================

beh = pd.read_csv(behavior_csv)
frames = pd.read_csv(video_frames_csv)

events_to_map = [
    ("blue_led_imec_up", "trial_start"),
    ("blue_led_imec_down", "trial_end"),
    ("stim_imec_up", "stim"),
    ("stim_imec_down", "stim_off"),
    ("reward_imec_up", "reward"),
    ("reward_imec_down", "reward_off"),
    ("punishment_imec_up", "punishment"),
    ("punishment_imec_down", "punishment_off"),
    ("lick_imec", "lick"),
]

for event_col, prefix in events_to_map:
    if event_col in beh.columns:
        beh = map_event_column_to_frame(beh, frames, event_col, prefix=prefix)
        print(f"Mapped {event_col} -> {prefix}_frame")
    else:
        print(f"Skipping {event_col} (not found)")

beh.to_csv(output_csv, index=False)
print(f"\nSaved -> {output_csv}")