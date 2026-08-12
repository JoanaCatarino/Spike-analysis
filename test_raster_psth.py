# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:43:13 2026

@author: JoanaCatarino
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO


# ============================================================
# File paths
# ============================================================

animal = 999770
day = 1
probe = "imec1"

nwb_file = rf"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/{animal}/{animal}_day{day}_g0_{probe}/{animal}_day{day}_{probe}_filtered_spikes.nwb"


ttl_align_dir = rf"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/{animal}/{animal}_day{day}_g0_{probe}/TTL_alignment"
ttl_file = os.path.join(ttl_align_dir, f"{animal}_day{day}_{probe}_ttl_aligned_imec_time.npz")


# General PSTH settings
T_PRE = 1.0
T_POST = 1.5
BIN = 0.010
SMOOTH_BINS = 3
MAKE_RASTER = True

# Example trial settings
TRIAL_T_PRE = 0.5
TRIAL_T_POST = 2.0
EXAMPLE_TRIAL_INDEX = 0

# Plot controls
PLOT_INDIVIDUAL_UNITS = True
MAX_UNITS_TO_PLOT = 20

PLOT_POPULATION_MEAN = True
PLOT_POPULATION_HEATMAP = True
PLOT_SORTED_HEATMAP = True
SORT_HEATMAP_EVENT = "Stim"        # can be "Stim", "Reward", "Punishment", "Blue LED"

PLOT_REWARD_VS_PUNISHMENT_DIFF = True

PLOT_EXAMPLE_TRIAL_REWARD = True
PLOT_EXAMPLE_TRIAL_PUNISHMENT = True

PLOT_PER_TRIAL_HEATMAP_SINGLE_UNIT = True
UNIT_FOR_TRIAL_HEATMAP = None      # None = first unit in file
TRIAL_HEATMAP_ALIGN_EVENT = "Stim" # can be "Stim", "Reward", "Punishment", "Blue LED"

# Map labels to keys inside your TTL npz file
events_map = {
    "Reward": "ttl_rew_d_imec",
    "Punishment": "ttl_pun_d_imec",
    "Stim": "ttl_stim_d_imec",
    "Blue LED": "ttl_led_d_imec",
    "Laser": "ttl_laser_a_imec",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_nwb_units(nwb_path):
    """
    Load units from NWB.
    Returns:
        units_dict: {unit_id: spike_times_in_seconds}
        quality_map: {unit_id: quality}
    """
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
        if "quality" in units_df.columns:
            quality_map[int(unit_id)] = str(row["quality"])
        else:
            quality_map[int(unit_id)] = "unknown"

    return units_dict, quality_map


def load_ttl_events(ttl_npz_path, events_map):
    """
    Load TTL event times from npz.
    If arrays are Nx2, first column is assumed to be event onset.
    """
    if not os.path.exists(ttl_npz_path):
        raise FileNotFoundError(f"TTL file not found: {ttl_npz_path}")

    z = np.load(ttl_npz_path, allow_pickle=True)

    out = {}
    for label, key in events_map.items():
        if key not in z.files:
            print(f"[WARN] Missing TTL key: {key} (skipping {label})")
            continue

        arr = np.asarray(z[key])

        if arr.size == 0:
            out[label] = np.array([], dtype=float)
        elif arr.ndim == 1:
            out[label] = arr.astype(float)
        else:
            out[label] = arr[:, 0].astype(float)

    return out


# ============================================================
# HELPERS
# ============================================================

def moving_average(x, w):
    if w is None or w <= 1:
        return x
    kernel = np.ones(int(w), dtype=float) / int(w)
    return np.convolve(x, kernel, mode="same")


def compute_psth(spike_times_s, event_times_s, t_pre, t_post, bin_s):
    """
    Standard PSTH.
    Returns:
        centers : bin centers
        rate    : firing rate in Hz
        raster  : list of relative spike times for each event
    """
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
        lo = np.searchsorted(spike_times_s, t0 - t_pre, side="left")
        hi = np.searchsorted(spike_times_s, t0 + t_post, side="right")
        rel = spike_times_s[lo:hi] - t0
        raster.append(rel)
        h, _ = np.histogram(rel, bins=edges)
        counts += h

    rate = counts / (len(event_times_s) * bin_s)
    return centers, rate, raster


def compute_trial_rate_matrix(spike_times_s, event_times_s, t_pre, t_post, bin_s):
    """
    For one unit and one event type:
    rows = trials
    cols = time bins
    values = firing rate (Hz) in each trial
    """
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    event_times_s = np.asarray(event_times_s, dtype=float)

    edges = np.arange(-t_pre, t_post + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    if spike_times_s.size == 0 or event_times_s.size == 0:
        return centers, np.zeros((0, len(centers)))

    spike_times_s = np.sort(spike_times_s)
    mat = []

    for t0 in event_times_s:
        lo = np.searchsorted(spike_times_s, t0 - t_pre, side="left")
        hi = np.searchsorted(spike_times_s, t0 + t_post, side="right")
        rel = spike_times_s[lo:hi] - t0
        counts, _ = np.histogram(rel, bins=edges)
        mat.append(counts / bin_s)

    return centers, np.asarray(mat)


def zscore_rows(mat):
    """
    Z-score each row independently.
    Helpful for heatmaps where units have very different baseline firing rates.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.size == 0:
        return mat
    mean = mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (mat - mean) / std


def get_trial_sequence_times(events_dict, trial_index=0, outcome="reward"):
    led_times = np.asarray(events_dict.get("Blue LED", []), dtype=float)
    stim_times = np.asarray(events_dict.get("Stim", []), dtype=float)

    if outcome.lower() == "reward":
        out_label = "Reward"
    elif outcome.lower() == "punishment":
        out_label = "Punishment"
    else:
        raise ValueError("outcome must be 'reward' or 'punishment'")

    outcome_times = np.asarray(events_dict.get(out_label, []), dtype=float)

    if led_times.size == 0:
        raise ValueError("No Blue LED events found.")
    if trial_index >= led_times.size:
        raise IndexError(f"trial_index={trial_index} but only {led_times.size} Blue LED events available.")

    t_led = led_times[trial_index]
    t_stim = stim_times[trial_index] if trial_index < stim_times.size else None
    t_out = outcome_times[trial_index] if trial_index < outcome_times.size else None

    return t_led, t_stim, t_out, out_label


# ============================================================
# 1) SINGLE-UNIT RASTER + PSTH
# Why: first QC step; shows whether a unit responds to events.
# ============================================================

def plot_unit_psth_show(unit_id, spike_times_s, events_dict,
                        t_pre=1.0, t_post=1.5, bin_s=0.01, smooth_bins=3,
                        make_raster=True, quality="unknown"):

    event_labels = [k for k, v in events_dict.items() if np.asarray(v).size > 0]
    if len(event_labels) == 0:
        print(f"[SKIP] Unit {unit_id}: no events available.")
        return

    n = len(event_labels)
    fig_h = 3.2 * n if make_raster else 2.4 * n
    fig = plt.figure(figsize=(12, fig_h))
    fig.suptitle(f"Unit {unit_id} ({quality}) — raster + PSTH", y=0.995)

    for i, label in enumerate(event_labels, start=1):
        ev = np.asarray(events_dict[label], dtype=float)
        centers, rate, raster = compute_psth(spike_times_s, ev, t_pre, t_post, bin_s)
        rate_sm = moving_average(rate, smooth_bins)

        if make_raster:
            ax_r = fig.add_subplot(n, 2, (i - 1) * 2 + 1)
            for k, rel in enumerate(raster):
                if rel.size:
                    ax_r.vlines(rel, k + 0.5, k + 1.5)
            ax_r.axvline(0, linestyle="--", color="red")
            ax_r.set_xlim(-t_pre, t_post)
            ax_r.set_ylabel("Trials")
            ax_r.set_title(f"{label} — raster (n={len(ev)})")
            if i == n:
                ax_r.set_xlabel("Time from event (s)")
            else:
                ax_r.set_xticklabels([])

            ax_p = fig.add_subplot(n, 2, (i - 1) * 2 + 2)
        else:
            ax_p = fig.add_subplot(n, 1, i)

        ax_p.plot(centers, rate_sm)
        ax_p.axvline(0, linestyle="--", color="red")
        ax_p.set_xlim(-t_pre, t_post)
        ax_p.set_ylabel("Hz")
        ax_p.set_title(f"{label} — PSTH")
        if i == n:
            ax_p.set_xlabel("Time from event (s)")
        else:
            ax_p.set_xticklabels([])

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    plt.show()


# ============================================================
# 2) POPULATION MEAN PSTH
# Why: summarizes average response across all units.
# ============================================================

def plot_population_psth(units_dict, events_dict,
                         t_pre=1.0, t_post=1.5, bin_s=0.01, smooth_bins=3):
    event_labels = [k for k, v in events_dict.items() if np.asarray(v).size > 0]
    if len(event_labels) == 0:
        return

    n = len(event_labels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False)
    axes = axes.flatten()

    for ax, label in zip(axes, event_labels):
        ev = np.asarray(events_dict[label], dtype=float)
        all_rates = []

        for spike_times_s in units_dict.values():
            centers, rate, _ = compute_psth(spike_times_s, ev, t_pre, t_post, bin_s)
            all_rates.append(rate)

        all_rates = np.asarray(all_rates)
        mean_rate = np.mean(all_rates, axis=0)
        sem_rate = np.std(all_rates, axis=0) / np.sqrt(all_rates.shape[0])

        mean_rate = moving_average(mean_rate, smooth_bins)
        sem_rate = moving_average(sem_rate, smooth_bins)

        ax.plot(centers, mean_rate)
        ax.fill_between(centers, mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3)
        ax.axvline(0, linestyle="--", color="red")
        ax.set_xlim(-t_pre, t_post)
        ax.set_ylabel("Hz")
        ax.set_title(f"{label} — population mean PSTH (n_units={len(units_dict)})")

    axes[-1].set_xlabel("Time from event (s)")
    fig.tight_layout()
    plt.show()


# ============================================================
# 3) POPULATION HEATMAP
# Why: very informative for seeing population dynamics.
# ============================================================

def plot_population_heatmap(units_dict, events_dict,
                            t_pre=1.0, t_post=1.5, bin_s=0.01, smooth_bins=3,
                            zscore=True):
    event_labels = [k for k, v in events_dict.items() if np.asarray(v).size > 0]
    if len(event_labels) == 0:
        return

    unit_ids = list(units_dict.keys())

    for label in event_labels:
        ev = np.asarray(events_dict[label], dtype=float)
        rate_matrix = []

        for unit_id in unit_ids:
            centers, rate, _ = compute_psth(units_dict[unit_id], ev, t_pre, t_post, bin_s)
            rate_matrix.append(moving_average(rate, smooth_bins))

        rate_matrix = np.asarray(rate_matrix)
        plot_mat = zscore_rows(rate_matrix) if zscore else rate_matrix

        plt.figure(figsize=(10, 6))
        plt.imshow(
            plot_mat,
            aspect="auto",
            origin="lower",
            extent=[centers[0], centers[-1], 0, len(unit_ids)]
        )
        plt.axvline(0, linestyle="--", color="red")
        plt.colorbar(label="z-scored rate" if zscore else "Hz")
        plt.xlabel("Time from event (s)")
        plt.ylabel("Units")
        plt.title(f"{label} — population heatmap")
        plt.show()


# ============================================================
# 4) HEATMAP SORTED BY PEAK RESPONSE TIME
# Why: excellent for revealing neural sequences.
# ============================================================

def plot_sorted_population_heatmap(units_dict, event_times_s, event_label,
                                   t_pre=1.0, t_post=1.5, bin_s=0.01,
                                   smooth_bins=3, zscore=True):
    if len(event_times_s) == 0:
        print(f"[SKIP] No events for {event_label}")
        return

    unit_ids = list(units_dict.keys())
    rate_matrix = []

    for unit_id in unit_ids:
        centers, rate, _ = compute_psth(units_dict[unit_id], event_times_s, t_pre, t_post, bin_s)
        rate_matrix.append(moving_average(rate, smooth_bins))

    rate_matrix = np.asarray(rate_matrix)
    plot_mat = zscore_rows(rate_matrix) if zscore else rate_matrix

    peak_idx = np.argmax(plot_mat, axis=1)
    sort_order = np.argsort(peak_idx)
    sorted_mat = plot_mat[sort_order, :]

    plt.figure(figsize=(10, 6))
    plt.imshow(
        sorted_mat,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, sorted_mat.shape[0]]
    )
    plt.axvline(0, linestyle="--", color="red")
    plt.colorbar(label="z-scored rate" if zscore else "Hz")
    plt.xlabel("Time from event (s)")
    plt.ylabel("Units (sorted by peak time)")
    plt.title(f"{event_label} — heatmap sorted by response latency")
    plt.show()


# ============================================================
# 5) REWARD VS PUNISHMENT DIFFERENCE HEATMAP
# Why: highlights valence/outcome coding.
# ============================================================

def plot_reward_vs_punishment_difference(units_dict, events_dict,
                                         t_pre=1.0, t_post=1.5, bin_s=0.01,
                                         smooth_bins=3, zscore=False):
    rew = np.asarray(events_dict.get("Reward", []), dtype=float)
    pun = np.asarray(events_dict.get("Punishment", []), dtype=float)

    if rew.size == 0 or pun.size == 0:
        print("[SKIP] Need both Reward and Punishment events for difference heatmap.")
        return

    unit_ids = list(units_dict.keys())
    diff_matrix = []

    for unit_id in unit_ids:
        centers, rate_rew, _ = compute_psth(units_dict[unit_id], rew, t_pre, t_post, bin_s)
        _, rate_pun, _ = compute_psth(units_dict[unit_id], pun, t_pre, t_post, bin_s)

        rate_rew = moving_average(rate_rew, smooth_bins)
        rate_pun = moving_average(rate_pun, smooth_bins)

        diff_matrix.append(rate_rew - rate_pun)

    diff_matrix = np.asarray(diff_matrix)
    if zscore:
        diff_matrix = zscore_rows(diff_matrix)

    plt.figure(figsize=(10, 6))
    vmax = np.max(np.abs(diff_matrix)) if diff_matrix.size else 1
    plt.imshow(
        diff_matrix,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, len(unit_ids)],
        vmin=-vmax,
        vmax=vmax,
        cmap="bwr"
    )
    plt.axvline(0, linestyle="--", color="black")
    plt.colorbar(label="Reward - Punishment")
    plt.xlabel("Time from outcome (s)")
    plt.ylabel("Units")
    plt.title("Reward vs Punishment difference heatmap")
    plt.show()


# ============================================================
# 6) EXAMPLE TRIAL RASTER ACROSS ALL UNITS
# Why: nice intuitive view of one trial sequence.
# ============================================================

def plot_trial_sequence_raster(units_dict, events_dict,
                               trial_index=0, t_pre=0.5, t_post=2.0,
                               outcome="reward"):
    t_led, t_stim, t_out, out_label = get_trial_sequence_times(
        events_dict, trial_index=trial_index, outcome=outcome
    )

    unit_ids = list(units_dict.keys())
    t0 = t_led - t_pre
    t1 = t_led + t_post

    plt.figure(figsize=(12, 8))

    for i, unit_id in enumerate(unit_ids):
        spikes = np.asarray(units_dict[unit_id], dtype=float)
        mask = (spikes >= t0) & (spikes <= t1)
        rel = spikes[mask] - t_led
        if rel.size:
            plt.vlines(rel, i + 0.5, i + 1.5)

    plt.axvline(0, linestyle="--", color="blue", label="Blue LED")
    if t_stim is not None:
        plt.axvline(t_stim - t_led, linestyle="--", color="green", label="Stim")
    if t_out is not None:
        plt.axvline(
            t_out - t_led,
            linestyle="--",
            color="red" if out_label == "Reward" else "orange",
            label=out_label
        )

    plt.xlabel("Time from Blue LED (s)")
    plt.ylabel("Units")
    plt.title(f"Example trial {trial_index}: Blue LED → Stim → {out_label}")
    plt.ylim(0, len(unit_ids) + 1)
    plt.legend()
    plt.show()


# ============================================================
# 7) EXAMPLE TRIAL HEATMAP ACROSS ALL UNITS
# Why: cleaner than raster for large populations.
# ============================================================

def plot_trial_sequence_heatmap(units_dict, events_dict,
                                trial_index=0, t_pre=0.5, t_post=2.0,
                                bin_s=0.01, smooth_bins=3, outcome="reward",
                                zscore=True):
    t_led, t_stim, t_out, out_label = get_trial_sequence_times(
        events_dict, trial_index=trial_index, outcome=outcome
    )

    unit_ids = list(units_dict.keys())
    edges = np.arange(-t_pre, t_post + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    rate_matrix = []

    for unit_id in unit_ids:
        spikes = np.sort(np.asarray(units_dict[unit_id], dtype=float))
        lo = np.searchsorted(spikes, t_led - t_pre, side="left")
        hi = np.searchsorted(spikes, t_led + t_post, side="right")
        rel = spikes[lo:hi] - t_led
        counts, _ = np.histogram(rel, bins=edges)
        rate = moving_average(counts / bin_s, smooth_bins)
        rate_matrix.append(rate)

    rate_matrix = np.asarray(rate_matrix)
    plot_mat = zscore_rows(rate_matrix) if zscore else rate_matrix

    plt.figure(figsize=(12, 8))
    plt.imshow(
        plot_mat,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, len(unit_ids)]
    )
    plt.axvline(0, linestyle="--", color="blue", label="Blue LED")
    if t_stim is not None:
        plt.axvline(t_stim - t_led, linestyle="--", color="green", label="Stim")
    if t_out is not None:
        plt.axvline(
            t_out - t_led,
            linestyle="--",
            color="red" if out_label == "Reward" else "orange",
            label=out_label
        )
    plt.colorbar(label="z-scored rate" if zscore else "Hz")
    plt.xlabel("Time from Blue LED (s)")
    plt.ylabel("Units")
    plt.title(f"Example trial {trial_index}: Blue LED → Stim → {out_label} heatmap")
    plt.legend()
    plt.show()


# ============================================================
# 8) PER-TRIAL HEATMAP FOR ONE UNIT
# Why: shows variability across trials instead of averaging.
# ============================================================

def plot_single_unit_trial_heatmap(unit_id, spike_times_s, event_times_s, event_label,
                                   t_pre=1.0, t_post=1.5, bin_s=0.01,
                                   smooth_bins=3):
    centers, mat = compute_trial_rate_matrix(spike_times_s, event_times_s, t_pre, t_post, bin_s)

    if mat.shape[0] == 0:
        print(f"[SKIP] No trials for unit {unit_id}, event {event_label}")
        return

    if smooth_bins > 1:
        mat = np.array([moving_average(row, smooth_bins) for row in mat])

    plt.figure(figsize=(10, 6))
    plt.imshow(
        mat,
        aspect="auto",
        origin="lower",
        extent=[centers[0], centers[-1], 0, mat.shape[0]]
    )
    plt.axvline(0, linestyle="--", color="red")
    plt.colorbar(label="Hz")
    plt.xlabel("Time from event (s)")
    plt.ylabel("Trials")
    plt.title(f"Unit {unit_id} — per-trial heatmap aligned to {event_label}")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    print("Looking for NWB at:")
    print(nwb_file)
    print("Exists?", os.path.exists(nwb_file))

    print("\nLooking for TTL file at:")
    print(ttl_file)
    print("Exists?", os.path.exists(ttl_file))

    # Load data
    units_dict, quality_map = load_nwb_units(nwb_file)
    events_dict = load_ttl_events(ttl_file, events_map)

    # Summary
    print("\n[NWB] Total units in file:", len(units_dict))
    qualities = list(quality_map.values())
    print("[NWB] good units:", qualities.count("good"))
    print("[NWB] MUA units:", qualities.count("MUA"))
    print("[NWB] other labels:", len(qualities) - qualities.count("good") - qualities.count("MUA"))

    print("\n[TTL] event counts:")
    for k, v in events_dict.items():
        print(f"  {k}: {len(v)}")

    # 1) Single-unit raster + PSTH
    if PLOT_INDIVIDUAL_UNITS:
        unit_ids = list(units_dict.keys())[:MAX_UNITS_TO_PLOT]
        for i, unit_id in enumerate(unit_ids, start=1):
            st_unit = units_dict[unit_id]
            if st_unit.size == 0:
                continue
            print(f"[PLOT UNIT] {i}/{len(unit_ids)} unit={unit_id}, spikes={st_unit.size}")
            plot_unit_psth_show(
                unit_id=unit_id,
                spike_times_s=st_unit,
                events_dict=events_dict,
                t_pre=T_PRE,
                t_post=T_POST,
                bin_s=BIN,
                smooth_bins=SMOOTH_BINS,
                make_raster=MAKE_RASTER,
                quality=quality_map.get(unit_id, "unknown")
            )

    # 2) Population mean PSTH
    if PLOT_POPULATION_MEAN:
        plot_population_psth(
            units_dict=units_dict,
            events_dict=events_dict,
            t_pre=T_PRE,
            t_post=T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS
        )

    # 3) Population heatmap
    if PLOT_POPULATION_HEATMAP:
        plot_population_heatmap(
            units_dict=units_dict,
            events_dict=events_dict,
            t_pre=T_PRE,
            t_post=T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS,
            zscore=True
        )

    # 4) Sorted heatmap
    if PLOT_SORTED_HEATMAP:
        if SORT_HEATMAP_EVENT in events_dict:
            plot_sorted_population_heatmap(
                units_dict=units_dict,
                event_times_s=np.asarray(events_dict[SORT_HEATMAP_EVENT], dtype=float),
                event_label=SORT_HEATMAP_EVENT,
                t_pre=T_PRE,
                t_post=T_POST,
                bin_s=BIN,
                smooth_bins=SMOOTH_BINS,
                zscore=True
            )
        else:
            print(f"[SKIP] SORT_HEATMAP_EVENT not found: {SORT_HEATMAP_EVENT}")

    # 5) Reward vs punishment difference
    if PLOT_REWARD_VS_PUNISHMENT_DIFF:
        plot_reward_vs_punishment_difference(
            units_dict=units_dict,
            events_dict=events_dict,
            t_pre=T_PRE,
            t_post=T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS,
            zscore=False
        )

    # 6) Example reward trial
    if PLOT_EXAMPLE_TRIAL_REWARD:
        plot_trial_sequence_raster(
            units_dict=units_dict,
            events_dict=events_dict,
            trial_index=EXAMPLE_TRIAL_INDEX,
            t_pre=TRIAL_T_PRE,
            t_post=TRIAL_T_POST,
            outcome="reward"
        )
        plot_trial_sequence_heatmap(
            units_dict=units_dict,
            events_dict=events_dict,
            trial_index=EXAMPLE_TRIAL_INDEX,
            t_pre=TRIAL_T_PRE,
            t_post=TRIAL_T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS,
            outcome="reward",
            zscore=True
        )

    # 7) Example punishment trial
    if PLOT_EXAMPLE_TRIAL_PUNISHMENT:
        plot_trial_sequence_raster(
            units_dict=units_dict,
            events_dict=events_dict,
            trial_index=EXAMPLE_TRIAL_INDEX,
            t_pre=TRIAL_T_PRE,
            t_post=TRIAL_T_POST,
            outcome="punishment"
        )
        plot_trial_sequence_heatmap(
            units_dict=units_dict,
            events_dict=events_dict,
            trial_index=EXAMPLE_TRIAL_INDEX,
            t_pre=TRIAL_T_PRE,
            t_post=TRIAL_T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS,
            outcome="punishment",
            zscore=True
        )

    # 8) Per-trial heatmap for one selected unit
    if PLOT_PER_TRIAL_HEATMAP_SINGLE_UNIT:
        unit_ids = list(units_dict.keys())
        chosen_unit = unit_ids[0] if UNIT_FOR_TRIAL_HEATMAP is None else UNIT_FOR_TRIAL_HEATMAP

        if chosen_unit not in units_dict:
            print(f"[SKIP] UNIT_FOR_TRIAL_HEATMAP={chosen_unit} not found in NWB.")
        elif TRIAL_HEATMAP_ALIGN_EVENT not in events_dict:
            print(f"[SKIP] TRIAL_HEATMAP_ALIGN_EVENT={TRIAL_HEATMAP_ALIGN_EVENT} not found.")
        else:
            plot_single_unit_trial_heatmap(
                unit_id=chosen_unit,
                spike_times_s=units_dict[chosen_unit],
                event_times_s=np.asarray(events_dict[TRIAL_HEATMAP_ALIGN_EVENT], dtype=float),
                event_label=TRIAL_HEATMAP_ALIGN_EVENT,
                t_pre=T_PRE,
                t_post=T_POST,
                bin_s=BIN,
                smooth_bins=SMOOTH_BINS
            )


if __name__ == "__main__":
    main()