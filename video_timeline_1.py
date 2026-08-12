# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:09:33 2026

@author: JoanaCatarino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a presentation-ready multi-trial video with:
  - original video on top
  - event timeline underneath
  - moving cursor showing current frame
  - event markers/labels for each trial

Output:
  one MP4 containing several trial snippets back-to-back

Requirements:
  pip install opencv-python pandas numpy
"""

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

output_path = Path(
    r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/videos/trials_timeline_movie.mp4'
)


output_path.parent.mkdir(parents=True, exist_ok=True)

# choose trials to render
TRIALS_TO_RENDER = [10, 11, 12, 55, 30, 31, 126]

# snippet choice
USE_WHOLE_TRIAL = False
ALIGN_TO = "stim_frame"   # or "trial_start_frame", "lick_frame", ...
FRAMES_BEFORE = 30
FRAMES_AFTER = 120

# playback
SPEED_FACTOR = 0.25      # 1.0 = normal, 0.5 = half-speed output fps
PAUSE_FRAMES_BETWEEN_TRIALS = 30

# video layout
TIMELINE_HEIGHT = 190
HEADER_HEIGHT = 40
SPACER_HEIGHT = 40   # try 30–80 depending on taste
FONT = cv2.FONT_HERSHEY_SIMPLEX

# colors (BGR because OpenCV)
BG_COLOR = (245, 245, 245)
TEXT_COLOR = (30, 30, 30)
SUBTLE_TEXT = (110, 110, 110)
CURSOR_COLOR = (20, 20, 20)
BASELINE_COLOR = (120, 120, 120)

EVENT_COLORS = {
    "Blue LED ON":  (255, 180, 50),
    "Blue LED OFF": (170, 120, 40),
    "Stim ON":      (80, 220, 255),
    "Reward":       (70, 220, 90),
    "Punishment":   (70, 70, 255),
    "Lick":         (220, 220, 220),
}

# timeline geometry
LEFT_PAD = 80
RIGHT_PAD = 40
Y_BASE = 105
LABEL_Y_TOP = [38, 60, 20]
LABEL_Y_BOTTOM = [145, 165, 125]


# =============================================================================
#%% Helpers
# =============================================================================

def draw_text(img, text, xy, scale=0.55, color=(255, 255, 255), thickness=1):
    cv2.putText(img, str(text), xy, FONT, scale, color, thickness, cv2.LINE_AA)


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


def get_trial_row(beh: pd.DataFrame, trial_number: int) -> pd.Series:
    x = beh.loc[beh["trial_number"] == trial_number]
    if len(x) == 0:
        raise ValueError(f"trial_number {trial_number} not found")
    return x.iloc[0]


def get_snippet_bounds(row: pd.Series):
    if USE_WHOLE_TRIAL:
        if pd.isna(row["trial_start_frame"]) or pd.isna(row["trial_end_frame"]):
            raise ValueError(f"Trial {row['trial_number']}: missing trial start/end frames")
        return int(row["trial_start_frame"]), int(row["trial_end_frame"])

    if ALIGN_TO not in row.index or pd.isna(row[ALIGN_TO]):
        raise ValueError(f"Trial {row['trial_number']}: missing {ALIGN_TO}")

    center = int(row[ALIGN_TO])
    return center - FRAMES_BEFORE, center + FRAMES_AFTER


def frame_to_x(frame_num: int, start_frame: int, end_frame: int, width: int) -> int:
    usable = width - LEFT_PAD - RIGHT_PAD
    if end_frame <= start_frame:
        return LEFT_PAD
    frac = (frame_num - start_frame) / (end_frame - start_frame)
    frac = np.clip(frac, 0, 1)
    return int(LEFT_PAD + frac * usable)


def add_event_if_valid(events, row, col, label):
    if col in row.index and pd.notna(row[col]):
        events.append({
            "label": label,
            "frame": int(row[col]),
        })


def build_trial_events(row: pd.Series):
    """
    Event list in frame units for drawing timeline markers.
    """
    events = []
    add_event_if_valid(events, row, "trial_start_frame", "Blue LED ON")
    add_event_if_valid(events, row, "trial_end_frame", "Blue LED OFF")
    add_event_if_valid(events, row, "stim_frame", "Stim ON")
    add_event_if_valid(events, row, "reward_frame", "Reward")
    add_event_if_valid(events, row, "punishment_frame", "Punishment")
    add_event_if_valid(events, row, "lick_frame", "Lick")

    events = sorted(events, key=lambda x: x["frame"])
    return events


def trial_metadata_text(row: pd.Series):
    parts = [f"Trial {int(row['trial_number'])}"]

    if "block" in row.index and pd.notna(row["block"]):
        parts.append(f"Block: {row['block']}")

    if "reward" in row.index and row["reward"] == 1:
        outcome = "Reward"
    elif "punishment" in row.index and row["punishment"] == 1:
        outcome = "Punishment"
    elif "omission" in row.index and row["omission"] == 1:
        outcome = "Omission"
    else:
        outcome = None

    if outcome is not None:
        parts.append(f"Outcome: {outcome}")

    if "8KHz" in row.index and row["8KHz"] == 1:
        parts.append("Stim: 8kHz")
    elif "16KHz" in row.index and row["16KHz"] == 1:
        parts.append("Stim: 16kHz")

    return "   |   ".join(parts)


def draw_event_marker(panel, x, label, color, idx):
    """
    Alternate labels above/below baseline to reduce overlap.
    """
    if idx % 2 == 0:
        y_label = LABEL_Y_TOP[(idx // 2) % len(LABEL_Y_TOP)]
        cv2.line(panel, (x, Y_BASE - 18), (x, y_label + 8), color, 1)
    else:
        y_label = LABEL_Y_BOTTOM[(idx // 2) % len(LABEL_Y_BOTTOM)]
        cv2.line(panel, (x, Y_BASE + 18), (x, y_label - 14), color, 1)

    cv2.line(panel, (x, Y_BASE - 18), (x, Y_BASE + 18), color, 2)
    cv2.circle(panel, (x, Y_BASE), 4, color, -1)

    text_x = max(5, x - 42)
    draw_text(panel, label, (text_x, y_label), scale=0.45, color=color, thickness=1)


def draw_timeline_panel(width, cur_frame, start_frame, end_frame, row, events, fps):
    panel = np.zeros((TIMELINE_HEIGHT, width, 3), dtype=np.uint8)
    panel[:] = BG_COLOR

    # title
    draw_text(panel, "Behavior events", (20, 24), scale=0.62, color=TEXT_COLOR, thickness=1)

    # baseline
    cv2.line(panel, (LEFT_PAD, Y_BASE), (width - RIGHT_PAD, Y_BASE), BASELINE_COLOR, 2)

    # start/end labels
    draw_text(panel, f"{start_frame}", (LEFT_PAD - 12, Y_BASE + 28), scale=0.42, color=SUBTLE_TEXT)
    draw_text(panel, f"{end_frame}", (width - RIGHT_PAD - 25, Y_BASE + 28), scale=0.42, color=SUBTLE_TEXT)

    # event markers
    for i, ev in enumerate(events):
        x = frame_to_x(ev["frame"], start_frame, end_frame, width)
        color = EVENT_COLORS.get(ev["label"], TEXT_COLOR)
        draw_event_marker(panel, x, ev["label"], color, i)

    # cursor
    x_cur = frame_to_x(cur_frame, start_frame, end_frame, width)
    cv2.line(panel, (x_cur, 10), (x_cur, TIMELINE_HEIGHT - 15), CURSOR_COLOR, 2)

    # current relative time
    t_rel = (cur_frame - start_frame) / fps
    draw_text(panel, f"t = {t_rel:+.3f} s", (20, TIMELINE_HEIGHT - 18), scale=0.55, color=TEXT_COLOR, thickness=1)

    return panel


def draw_header(width, row):
    hdr = np.zeros((HEADER_HEIGHT, width, 3), dtype=np.uint8)
    hdr[:] = (10, 10, 10)
    draw_text(hdr, trial_metadata_text(row), (18, 27), scale=0.60, color=TEXT_COLOR, thickness=1)
    return hdr


def write_pause_frames(writer, frame, n_pause):
    for _ in range(n_pause):
        writer.write(frame)


# =============================================================================
#%% Main renderer
# =============================================================================

def render_multi_trial_movie(
    video_path: Path,
    behavior_csv: Path,
    output_path: Path,
    trial_numbers,
):
    beh = pd.read_csv(behavior_csv)
    fps, n_frames, width, height = get_video_info(video_path)

    fps_out = fps * SPEED_FACTOR
    if fps_out <= 0:
        raise ValueError("fps_out must be > 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    out_w = width
    out_h = HEADER_HEIGHT + height + SPACER_HEIGHT + TIMELINE_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (out_w, out_h))

    print(f"Video: fps={fps:.3f}, n_frames={n_frames}, size={width}x{height}")
    print(f"Output: {output_path}")

    for trial_number in trial_numbers:
        row = get_trial_row(beh, trial_number)
        events = build_trial_events(row)

        start_frame, end_frame = get_snippet_bounds(row)
        start_frame = max(0, start_frame)
        end_frame = min(n_frames - 1, end_frame)

        print(f"Rendering trial {trial_number}: frames {start_frame} -> {end_frame}")

        header = draw_header(width, row)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cur_frame = start_frame

        last_composite = None

        while cur_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            timeline = draw_timeline_panel(
                width=width,
                cur_frame=cur_frame,
                start_frame=start_frame,
                end_frame=end_frame,
                row=row,
                events=events,
                fps=fps,
            )

            spacer = np.ones((SPACER_HEIGHT, width, 3), dtype=np.uint8) * 255
            composite = np.vstack([header, frame, spacer, timeline])
            writer.write(composite)
            last_composite = composite
            cur_frame += 1

        # brief hold between trials
        if last_composite is not None and PAUSE_FRAMES_BETWEEN_TRIALS > 0:
            write_pause_frames(writer, last_composite, PAUSE_FRAMES_BETWEEN_TRIALS)

    writer.release()
    cap.release()
    print(f"Saved -> {output_path}")


# =============================================================================
#%% Run
# =============================================================================

if __name__ == "__main__":
    render_multi_trial_movie(
        video_path=video_path,
        behavior_csv=behavior_csv,
        output_path=output_path,
        trial_numbers=TRIALS_TO_RENDER,
    )