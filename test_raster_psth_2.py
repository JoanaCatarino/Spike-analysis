# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:09:04 2026

@author: JoanaCatarino
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO


# ============================================================
# USER PATHS
# ============================================================

animal = 999770
day = 1
probe = "imec1"

nwb_file = rf"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/{animal}/{animal}_day{day}_g0_{probe}/{animal}_day{day}_{probe}_filtered_spikes.nwb"

# behavior file
behavior_csv = rf"L:/dmclab/Joana/PFC-Str_behavior_project/Behavior/Cohort_1/999770/Behavior/20251111/old/AdaptSensorimotor_999770_20251111_143934_boxEphys_old.csv"
# change this if needed

# TTL-aligned event file
ttl_align_dir = rf"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/{animal}/{animal}_day{day}_g0_{probe}/TTL_alignment"
ttl_file = os.path.join(ttl_align_dir, f"{animal}_day{day}_{probe}_ttl_aligned_imec_time.npz")


# ============================================================
# COLUMN MAPPING
# Edit here if your CSV uses slightly different names
# ============================================================

COL_TRIAL = "trial_number"
COL_BLOCK = "block"
COL_STIM = "stim"

COL_IS_8K = "8KHz"
COL_IS_16K = "16KHz"

COL_LICK = "lick"
COL_LEFT = "left_spout"
COL_RIGHT = "right_spout"
COL_LICK_TIME = "lick_time"

COL_REWARD = "reward"
COL_PUNISH = "punishment"
COL_OMISSION = "omission"
COL_EARLY = "early_lick"

# optional columns if present
COL_TRIAL_START = "trial_start"
COL_RW_START = "RW_start"
COL_TRIAL_END = "trial_end"
COL_ITI = "ITI"

# block naming
SOUND_BLOCK_LABELS = ["sound", "S", "Sound"]
ACTION_RIGHT_LABELS = ["action_right", "AR", "action-right", "Action Right"]
ACTION_LEFT_LABELS = ["action_left", "AL", "action-left", "Action Left"]


# ============================================================
# TTL NPZ KEYS
# ============================================================

events_map = {
    "Reward": "ttl_rew_d_imec",
    "Punishment": "ttl_pun_d_imec",
    "Stim": "ttl_stim_d_imec",
    "Blue LED": "ttl_led_d_imec",
}

# optional if you later have explicit choice times
# otherwise first lick is estimated from lick_time relative to stim
CHOICE_EVENT_NAME = "Choice"


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

T_PRE = 1.0
T_POST = 1.5
BIN = 0.010
SMOOTH_BINS = 3

MAX_UNITS_TO_PLOT = 12

SWITCH_WINDOW_PRE = 15
SWITCH_WINDOW_POST = 25

PLOT_SINGLE_UNITS = True
PLOT_POPULATION = True
PLOT_SWITCH_ANALYSIS = True


# ============================================================
# LOADERS
# ============================================================

def load_nwb_units(nwb_path):
    if not os.path.exists(nwb_path):
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    with NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        if nwbfile.units is None:
            raise ValueError("No units table found in NWB file.")
        units_df = nwbfile.units.to_dataframe()

    units_dict = {}
    quality_map = {}

    for unit_id, row in units_df.iterrows():
        units_dict[int(unit_id)] = np.asarray(row["spike_times"], dtype=float)
        quality_map[int(unit_id)] = str(row["quality"]) if "quality" in units_df.columns else "unknown"

    return units_dict, quality_map


def load_ttl_events(ttl_npz_path, events_map):
    if not os.path.exists(ttl_npz_path):
        raise FileNotFoundError(f"TTL file not found: {ttl_npz_path}")

    z = np.load(ttl_npz_path, allow_pickle=True)
    out = {}

    for label, key in events_map.items():
        if key not in z.files:
            print(f"[WARN] Missing TTL key: {key} (skipping {label})")
            out[label] = np.array([], dtype=float)
            continue

        arr = np.asarray(z[key])
        if arr.size == 0:
            out[label] = np.array([], dtype=float)
        elif arr.ndim == 1:
            out[label] = arr.astype(float)
        else:
            out[label] = arr[:, 0].astype(float)

    return out


def load_behavior_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Behavior CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


# ============================================================
# BEHAVIOR HELPERS
# ============================================================

def block_to_rule(block_value):
    s = str(block_value).strip()

    if s in SOUND_BLOCK_LABELS:
        return "sound"
    if s in ACTION_RIGHT_LABELS:
        return "action_right"
    if s in ACTION_LEFT_LABELS:
        return "action_left"

    s_low = s.lower()
    if "sound" in s_low:
        return "sound"
    if "action" in s_low and "right" in s_low:
        return "action_right"
    if "action" in s_low and "left" in s_low:
        return "action_left"
    if s_low == "ar":
        return "action_right"
    if s_low == "al":
        return "action_left"
    if s_low == "s":
        return "sound"

    return s_low


def add_task_columns(df):
    df = df.copy()

    # rule / block type
    df["rule"] = df[COL_BLOCK].apply(block_to_rule)

    # tone
    if COL_IS_8K in df.columns and COL_IS_16K in df.columns:
        df["tone"] = np.where(df[COL_IS_8K] == 1, "8k",
                      np.where(df[COL_IS_16K] == 1, "16k", "unknown"))
    else:
        stim_str = df[COL_STIM].astype(str).str.lower()
        df["tone"] = np.where(stim_str.str.contains("8"), "8k",
                      np.where(stim_str.str.contains("16"), "16k", "unknown"))

    # choice
    df["choice"] = np.where(df[COL_LEFT] == 1, "left",
                    np.where(df[COL_RIGHT] == 1, "right", "none"))

    # outcome
    df["outcome"] = np.where(df[COL_EARLY] == 1, "early",
                     np.where(df[COL_OMISSION] == 1, "omission",
                     np.where(df[COL_REWARD] == 1, "reward",
                     np.where(df[COL_PUNISH] == 1, "punishment", "unknown"))))

    # correctness
    df["is_correct"] = (df[COL_REWARD] == 1).astype(int)
    df["is_error"] = (df[COL_PUNISH] == 1).astype(int)

    # correct side
    # sound block: 8k/16k map to side must be edited if needed
    # IMPORTANT: adjust if in your task 8k->left and 16k->right is reversed
    df["correct_side"] = "unknown"
    df.loc[(df["rule"] == "sound") & (df["tone"] == "8k"), "correct_side"] = "left"
    df.loc[(df["rule"] == "sound") & (df["tone"] == "16k"), "correct_side"] = "right"
    df.loc[df["rule"] == "action_right", "correct_side"] = "right"
    df.loc[df["rule"] == "action_left", "correct_side"] = "left"

    # trial index within block
    block_change = df["rule"] != df["rule"].shift(1)
    df["block_id"] = block_change.cumsum()
    df["trial_in_block"] = df.groupby("block_id").cumcount()

    # switch-centered trial index
    df["is_switch_trial"] = 0
    switch_rows = np.where(block_change)[0]
    df.loc[switch_rows, "is_switch_trial"] = 1

    return df


def get_switch_indices(df):
    block_change = df["rule"] != df["rule"].shift(1)
    switch_ix = np.where(block_change)[0]
    if len(switch_ix) > 0 and switch_ix[0] == 0:
        switch_ix = switch_ix[1:]
    return switch_ix


# ============================================================
# ALIGN BEHAVIOR TO TTL EVENTS
# ============================================================

def attach_trial_event_times(df, events_dict):
    """
    Assumes one event per trial in order.
    If lengths do not match exactly, uses minimum length and warns.
    """
    df = df.copy()

    for label in ["Blue LED", "Stim", "Reward", "Punishment"]:
        arr = np.asarray(events_dict.get(label, []), dtype=float)
        n = min(len(df), len(arr))
        if len(arr) != len(df):
            print(f"[WARN] {label}: behavior trials={len(df)}, TTL events={len(arr)}. Using first {n}.")
        new_col = f"{label.lower().replace(' ', '_')}_time"
        df[new_col] = np.nan
        df.loc[df.index[:n], new_col] = arr[:n]

    # estimate choice time from stim time + lick_time
    if "stim_time" in df.columns and COL_LICK_TIME in df.columns:
        df["choice_time"] = df["stim_time"] + pd.to_numeric(df[COL_LICK_TIME], errors="coerce")
    else:
        df["choice_time"] = np.nan

    # unified outcome_time
    df["outcome_time"] = np.nan
    reward_mask = df["outcome"] == "reward"
    punish_mask = df["outcome"] == "punishment"

    if "reward_time" in df.columns:
        df.loc[reward_mask, "outcome_time"] = df.loc[reward_mask, "reward_time"]
    if "punishment_time" in df.columns:
        df.loc[punish_mask, "outcome_time"] = df.loc[punish_mask, "punishment_time"]

    return df


# ============================================================
# SPIKE HELPERS
# ============================================================

def moving_average(x, w):
    if w is None or w <= 1:
        return x
    kernel = np.ones(int(w), dtype=float) / int(w)
    return np.convolve(x, kernel, mode="same")


def compute_psth(spike_times_s, event_times_s, t_pre, t_post, bin_s):
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    event_times_s = np.asarray(event_times_s, dtype=float)

    edges = np.arange(-t_pre, t_post + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    if spike_times_s.size == 0 or event_times_s.size == 0:
        return centers, np.zeros_like(centers), []

    spike_times_s = np.sort(spike_times_s)

    counts = np.zeros(len(centers), dtype=float)
    raster = []

    for t0 in event_times_s:
        if np.isnan(t0):
            continue
        lo = np.searchsorted(spike_times_s, t0 - t_pre, side="left")
        hi = np.searchsorted(spike_times_s, t0 + t_post, side="right")
        rel = spike_times_s[lo:hi] - t0
        raster.append(rel)
        h, _ = np.histogram(rel, bins=edges)
        counts += h

    n_events = sum(~np.isnan(event_times_s))
    if n_events == 0:
        return centers, np.zeros_like(centers), []

    rate = counts / (n_events * bin_s)
    return centers, rate, raster


def zscore_rows(mat):
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return mat
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (mat - mean) / std


# ============================================================
# TRIAL SELECTION
# ============================================================

def get_event_times(df, mask, align_col):
    return df.loc[mask, align_col].dropna().to_numpy(dtype=float)


def make_masks(df):
    masks = {}

    # basic
    masks["sound_8k"] = (df["rule"] == "sound") & (df["tone"] == "8k")
    masks["sound_16k"] = (df["rule"] == "sound") & (df["tone"] == "16k")

    masks["ar_8k"] = (df["rule"] == "action_right") & (df["tone"] == "8k")
    masks["ar_16k"] = (df["rule"] == "action_right") & (df["tone"] == "16k")

    masks["al_8k"] = (df["rule"] == "action_left") & (df["tone"] == "8k")
    masks["al_16k"] = (df["rule"] == "action_left") & (df["tone"] == "16k")

    masks["left_choice"] = df["choice"] == "left"
    masks["right_choice"] = df["choice"] == "right"

    masks["reward"] = df["outcome"] == "reward"
    masks["punishment"] = df["outcome"] == "punishment"

    masks["correct"] = df["is_correct"] == 1
    masks["error"] = df["is_error"] == 1

    masks["sound_correct"] = (df["rule"] == "sound") & (df["is_correct"] == 1)
    masks["sound_error"] = (df["rule"] == "sound") & (df["is_error"] == 1)

    return masks


# ============================================================
# SINGLE UNIT PLOTS
# ============================================================

def plot_single_unit_raster_psth(unit_id, spike_times, event_times, title,
                                 t_pre=T_PRE, t_post=T_POST, bin_s=BIN,
                                 smooth_bins=SMOOTH_BINS):
    centers, rate, raster = compute_psth(spike_times, event_times, t_pre, t_post, bin_s)
    rate_sm = moving_average(rate, smooth_bins)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    for i, rel in enumerate(raster):
        if rel.size:
            ax1.vlines(rel, i + 0.5, i + 1.5)
    ax1.axvline(0, linestyle="--", color="red")
    ax1.set_xlim(-t_pre, t_post)
    ax1.set_ylabel("Trials")
    ax1.set_title(f"Unit {unit_id} — {title}")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(centers, rate_sm)
    ax2.axvline(0, linestyle="--", color="red")
    ax2.set_xlim(-t_pre, t_post)
    ax2.set_ylabel("Hz")
    ax2.set_xlabel("Time from event (s)")

    fig.tight_layout()
    plt.show()


# ============================================================
# POPULATION PLOTS
# ============================================================

def plot_population_psth(units_dict, event_times, title,
                         t_pre=T_PRE, t_post=T_POST, bin_s=BIN,
                         smooth_bins=SMOOTH_BINS):
    all_rates = []

    for spikes in units_dict.values():
        centers, rate, _ = compute_psth(spikes, event_times, t_pre, t_post, bin_s)
        all_rates.append(rate)

    all_rates = np.asarray(all_rates)
    mean_rate = np.mean(all_rates, axis=0)
    sem_rate = np.std(all_rates, axis=0) / np.sqrt(all_rates.shape[0])

    mean_rate = moving_average(mean_rate, smooth_bins)
    sem_rate = moving_average(sem_rate, smooth_bins)

    plt.figure(figsize=(8, 4))
    plt.plot(centers, mean_rate)
    plt.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3)
    plt.axvline(0, linestyle="--", color="red")
    plt.xlim(-t_pre, t_post)
    plt.xlabel("Time from event (s)")
    plt.ylabel("Hz")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_population_heatmap(units_dict, event_times, title,
                            t_pre=T_PRE, t_post=T_POST, bin_s=BIN,
                            smooth_bins=SMOOTH_BINS, zscore=True, sort_by_peak=False):
    rate_matrix = []
    unit_ids = list(units_dict.keys())

    for unit_id in unit_ids:
        centers, rate, _ = compute_psth(units_dict[unit_id], event_times, t_pre, t_post, bin_s)
        rate_matrix.append(moving_average(rate, smooth_bins))

    rate_matrix = np.asarray(rate_matrix)
    plot_mat = zscore_rows(rate_matrix) if zscore else rate_matrix

    if sort_by_peak:
        order = np.argsort(np.argmax(plot_mat, axis=1))
        plot_mat = plot_mat[order, :]

    plt.figure(figsize=(9, 6))
    plt.imshow(
        plot_mat,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, plot_mat.shape[0]]
    )
    plt.axvline(0, linestyle="--", color="red")
    plt.colorbar(label="z-score" if zscore else "Hz")
    plt.xlabel("Time from event (s)")
    plt.ylabel("Units")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_difference_heatmap(units_dict, ev_a, ev_b, title,
                            t_pre=T_PRE, t_post=T_POST, bin_s=BIN,
                            smooth_bins=SMOOTH_BINS):
    diffs = []

    for unit_id in units_dict:
        centers, ra, _ = compute_psth(units_dict[unit_id], ev_a, t_pre, t_post, bin_s)
        _, rb, _ = compute_psth(units_dict[unit_id], ev_b, t_pre, t_post, bin_s)
        diffs.append(moving_average(ra - rb, smooth_bins))

    diffs = np.asarray(diffs)
    vmax = np.max(np.abs(diffs)) if diffs.size else 1

    plt.figure(figsize=(9, 6))
    plt.imshow(
        diffs,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, diffs.shape[0]],
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.axvline(0, linestyle="--", color="black")
    plt.colorbar(label="Difference (Hz)")
    plt.xlabel("Time from event (s)")
    plt.ylabel("Units")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# SWITCH ANALYSIS
# ============================================================

def compute_switch_matrix(df, units_dict, align_col="stim_time",
                          pre_trials=15, post_trials=25,
                          t_pre=0.5, t_post=1.0, bin_s=0.02):
    switch_ix = get_switch_indices(df)
    unit_ids = list(units_dict.keys())

    trial_axis = np.arange(-pre_trials, post_trials + 1)
    unit_trial_matrix = np.full((len(unit_ids), len(trial_axis)), np.nan, dtype=float)

    for u, unit_id in enumerate(unit_ids):
        spikes = np.asarray(units_dict[unit_id], dtype=float)

        for s in switch_ix:
            rows = np.arange(s - pre_trials, s + post_trials + 1)
            valid = rows[(rows >= 0) & (rows < len(df))]

            temp = []
            for r in valid:
                t0 = df.iloc[r][align_col]
                if pd.isna(t0):
                    continue
                lo = np.searchsorted(spikes, t0 - t_pre, side="left")
                hi = np.searchsorted(spikes, t0 + t_post, side="right")
                nsp = hi - lo
                fr = nsp / (t_pre + t_post)
                temp.append((r - s, fr))

            for rel_trial, fr in temp:
                col = rel_trial + pre_trials
                if np.isnan(unit_trial_matrix[u, col]):
                    unit_trial_matrix[u, col] = fr
                else:
                    unit_trial_matrix[u, col] += fr

        # average over switches
        n_switch = max(len(switch_ix), 1)
        unit_trial_matrix[u, :] = unit_trial_matrix[u, :] / n_switch

    return trial_axis, unit_trial_matrix


def plot_switch_heatmap(df, units_dict, align_col="stim_time",
                        pre_trials=15, post_trials=25):
    trial_axis, mat = compute_switch_matrix(
        df, units_dict,
        align_col=align_col,
        pre_trials=pre_trials,
        post_trials=post_trials
    )
    mat = np.nan_to_num(mat, nan=0.0)
    mat = zscore_rows(mat)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[trial_axis[0], trial_axis[-1], 0, mat.shape[0]]
    )
    plt.axvline(0, linestyle="--", color="red")
    plt.colorbar(label="z-score")
    plt.xlabel("Trials from block switch")
    plt.ylabel("Units")
    plt.title(f"Population activity around unsignaled block switch ({align_col})")
    plt.tight_layout()
    plt.show()


def plot_switch_population_mean(df, units_dict, align_col="stim_time",
                                pre_trials=15, post_trials=25):
    trial_axis, mat = compute_switch_matrix(
        df, units_dict,
        align_col=align_col,
        pre_trials=pre_trials,
        post_trials=post_trials
    )
    mean_fr = np.nanmean(mat, axis=0)
    sem_fr = np.nanstd(mat, axis=0) / np.sqrt(mat.shape[0])

    plt.figure(figsize=(8, 4))
    plt.plot(trial_axis, mean_fr)
    plt.fill_between(trial_axis, mean_fr - sem_fr, mean_fr + sem_fr, alpha=0.3)
    plt.axvline(0, linestyle="--", color="red")
    plt.xlabel("Trials from block switch")
    plt.ylabel("Mean firing rate")
    plt.title(f"Mean population firing around switch ({align_col})")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("Loading NWB...")
    units_dict, quality_map = load_nwb_units(nwb_file)
    print(f"Units in NWB: {len(units_dict)}")

    print("Loading TTL...")
    events_dict = load_ttl_events(ttl_file, events_map)

    print("Loading behavior...")
    df = load_behavior_csv(behavior_csv)
    df = add_task_columns(df)
    df = attach_trial_event_times(df, events_dict)

    print("\nBehavior summary")
    print(df[["rule", "tone", "choice", "outcome"]].head())
    print(df["rule"].value_counts(dropna=False))
    print(df["tone"].value_counts(dropna=False))
    print(df["outcome"].value_counts(dropna=False))

    masks = make_masks(df)

    # --------------------------------------------------------
    # 1. SINGLE UNIT EXAMPLES
    # --------------------------------------------------------
    if PLOT_SINGLE_UNITS:
        example_units = list(units_dict.keys())[:MAX_UNITS_TO_PLOT]

        for unit_id in example_units:
            spikes = units_dict[unit_id]

            # tone aligned: same tone, different rule
            ev_sound_8 = get_event_times(df, masks["sound_8k"], "stim_time")
            ev_ar_8 = get_event_times(df, masks["ar_8k"], "stim_time")
            ev_al_8 = get_event_times(df, masks["al_8k"], "stim_time")

            if len(ev_sound_8) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_sound_8, "8 kHz in Sound block")
            if len(ev_ar_8) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_ar_8, "8 kHz in Action-right block")
            if len(ev_al_8) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_al_8, "8 kHz in Action-left block")

            # choice aligned
            ev_left = get_event_times(df, masks["left_choice"], "choice_time")
            ev_right = get_event_times(df, masks["right_choice"], "choice_time")

            if len(ev_left) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_left, "Left choices", t_pre=1.0, t_post=1.0)
            if len(ev_right) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_right, "Right choices", t_pre=1.0, t_post=1.0)

            # outcome aligned
            ev_reward = get_event_times(df, masks["reward"], "outcome_time")
            ev_pun = get_event_times(df, masks["punishment"], "outcome_time")

            if len(ev_reward) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_reward, "Reward", t_pre=1.0, t_post=1.5)
            if len(ev_pun) > 0:
                plot_single_unit_raster_psth(unit_id, spikes, ev_pun, "Punishment", t_pre=1.0, t_post=1.5)

    # --------------------------------------------------------
    # 2. POPULATION PSTHS / HEATMAPS
    # --------------------------------------------------------
    if PLOT_POPULATION:
        # Tone coding in sound blocks
        ev_sound_8 = get_event_times(df, masks["sound_8k"], "stim_time")
        ev_sound_16 = get_event_times(df, masks["sound_16k"], "stim_time")

        if len(ev_sound_8) > 0:
            plot_population_psth(units_dict, ev_sound_8, "Population PSTH — 8 kHz in Sound block")
            plot_population_heatmap(units_dict, ev_sound_8, "Population heatmap — 8 kHz in Sound block", sort_by_peak=True)

        if len(ev_sound_16) > 0:
            plot_population_psth(units_dict, ev_sound_16, "Population PSTH — 16 kHz in Sound block")
            plot_population_heatmap(units_dict, ev_sound_16, "Population heatmap — 16 kHz in Sound block", sort_by_peak=True)

        # Same tone, different rules
        ev_ar_8 = get_event_times(df, masks["ar_8k"], "stim_time")
        ev_al_8 = get_event_times(df, masks["al_8k"], "stim_time")

        if len(ev_sound_8) > 0:
            plot_population_psth(units_dict, ev_sound_8, "8 kHz — Sound")
        if len(ev_ar_8) > 0:
            plot_population_psth(units_dict, ev_ar_8, "8 kHz — Action Right")
        if len(ev_al_8) > 0:
            plot_population_psth(units_dict, ev_al_8, "8 kHz — Action Left")

        ev_ar_16 = get_event_times(df, masks["ar_16k"], "stim_time")
        ev_al_16 = get_event_times(df, masks["al_16k"], "stim_time")

        if len(ev_sound_16) > 0:
            plot_population_psth(units_dict, ev_sound_16, "16 kHz — Sound")
        if len(ev_ar_16) > 0:
            plot_population_psth(units_dict, ev_ar_16, "16 kHz — Action Right")
        if len(ev_al_16) > 0:
            plot_population_psth(units_dict, ev_al_16, "16 kHz — Action Left")

        # Choice coding
        ev_left = get_event_times(df, masks["left_choice"], "choice_time")
        ev_right = get_event_times(df, masks["right_choice"], "choice_time")

        if len(ev_left) > 0:
            plot_population_psth(units_dict, ev_left, "Choice-aligned — Left choices", t_pre=1.0, t_post=1.0)
            plot_population_heatmap(units_dict, ev_left, "Choice heatmap — Left", t_pre=1.0, t_post=1.0, sort_by_peak=True)

        if len(ev_right) > 0:
            plot_population_psth(units_dict, ev_right, "Choice-aligned — Right choices", t_pre=1.0, t_post=1.0)
            plot_population_heatmap(units_dict, ev_right, "Choice heatmap — Right", t_pre=1.0, t_post=1.0, sort_by_peak=True)

        # Outcome coding
        ev_reward = get_event_times(df, masks["reward"], "outcome_time")
        ev_pun = get_event_times(df, masks["punishment"], "outcome_time")

        if len(ev_reward) > 0:
            plot_population_psth(units_dict, ev_reward, "Outcome-aligned — Reward")
            plot_population_heatmap(units_dict, ev_reward, "Outcome heatmap — Reward", sort_by_peak=True)

        if len(ev_pun) > 0:
            plot_population_psth(units_dict, ev_pun, "Outcome-aligned — Punishment")
            plot_population_heatmap(units_dict, ev_pun, "Outcome heatmap — Punishment", sort_by_peak=True)

        if len(ev_reward) > 0 and len(ev_pun) > 0:
            plot_difference_heatmap(units_dict, ev_reward, ev_pun, "Reward minus Punishment")

        # Correct vs error
        ev_correct = get_event_times(df, masks["correct"], "choice_time")
        ev_error = get_event_times(df, masks["error"], "choice_time")

        if len(ev_correct) > 0:
            plot_population_psth(units_dict, ev_correct, "Choice-aligned — Correct trials", t_pre=1.0, t_post=1.0)
        if len(ev_error) > 0:
            plot_population_psth(units_dict, ev_error, "Choice-aligned — Error trials", t_pre=1.0, t_post=1.0)

        if len(ev_correct) > 0 and len(ev_error) > 0:
            plot_difference_heatmap(units_dict, ev_correct, ev_error, "Correct minus Error (choice aligned)", t_pre=1.0, t_post=1.0)

    # --------------------------------------------------------
    # 3. SWITCH ANALYSIS
    # --------------------------------------------------------
    if PLOT_SWITCH_ANALYSIS:
        if "stim_time" in df.columns:
            plot_switch_population_mean(
                df, units_dict,
                align_col="stim_time",
                pre_trials=SWITCH_WINDOW_PRE,
                post_trials=SWITCH_WINDOW_POST
            )
            plot_switch_heatmap(
                df, units_dict,
                align_col="stim_time",
                pre_trials=SWITCH_WINDOW_PRE,
                post_trials=SWITCH_WINDOW_POST
            )

        if "choice_time" in df.columns:
            plot_switch_population_mean(
                df, units_dict,
                align_col="choice_time",
                pre_trials=SWITCH_WINDOW_PRE,
                post_trials=SWITCH_WINDOW_POST
            )
            plot_switch_heatmap(
                df, units_dict,
                align_col="choice_time",
                pre_trials=SWITCH_WINDOW_PRE,
                post_trials=SWITCH_WINDOW_POST
            )


if __name__ == "__main__":
    main()