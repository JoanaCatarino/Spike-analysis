# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:09:14 2026

@author: JoanaCatarino

file_path = Path(r'L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/NWB/999770_20251111_2probes.nwb') 
"""

#%% Check what is inside the nwb file

from pathlib import Path
import pandas as pd
from pynwb import NWBHDF5IO

# ============================================================
# 1. File path
# ============================================================
file_path = Path('L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/NWB/999770_20251111_2probes_2.nwb')

if not file_path.exists():
    raise FileNotFoundError(f"NWB file not found: {file_path}")

# ============================================================
# 2. Open NWB file
# ============================================================
io = NWBHDF5IO(str(file_path), "r")
nwbfile = io.read()

# ============================================================
# 3. Basic info
# ============================================================
identifier = nwbfile.identifier
session_description = nwbfile.session_description
session_start_time = nwbfile.session_start_time
session_id = nwbfile.session_id

print("Identifier:", identifier)
print("Session description:", session_description)
print("Session start time:", session_start_time)
print("Session ID:", session_id)

# ============================================================
# 4. Subject info
# ============================================================
subject = nwbfile.subject

subject_id = None
species = None
sex = None
age = None

if subject is not None:
    subject_id = subject.subject_id
    species = subject.species
    sex = subject.sex
    age = subject.age

    print("\nSubject info")
    print("Subject ID:", subject_id)
    print("Species:", species)
    print("Sex:", sex)
    print("Age:", age)

# ============================================================
# 5. Processing modules
# ============================================================
processing_keys = list(nwbfile.processing.keys())
print("\nProcessing modules:", processing_keys)

# ============================================================
# 6. Electrodes table
# ============================================================
electrodes_df = None

if nwbfile.electrodes is not None:
    electrodes_df = nwbfile.electrodes.to_dataframe()
    print("\nElectrodes table loaded")
    print("Number of electrodes:", len(electrodes_df))
    print(electrodes_df.head())

# ============================================================
# 7. Trials table
# ============================================================
trials_df = None

if nwbfile.trials is not None:
    trials_df = nwbfile.trials.to_dataframe()
    print("\nTrials table loaded")
    print("Number of trials:", len(trials_df))
    print(trials_df.head())

# ============================================================
# 8. Units from probe 0
# ============================================================
units_probe0 = None
units_probe0_df = None
spike_times_probe0 = None

if "units_probe0" in nwbfile.processing:
    units_probe0 = nwbfile.processing["units_probe0"]["units_probe0"]
    units_probe0_df = units_probe0.to_dataframe()
    spike_times_probe0 = units_probe0_df["spike_times"]

    print("\nProbe 0 units loaded")
    print("Number of units:", len(units_probe0_df))
    print("Columns:", list(units_probe0_df.columns))
    print(units_probe0_df.head())

# ============================================================
# 9. Units from probe 1
# ============================================================
units_probe1 = None
units_probe1_df = None
spike_times_probe1 = None

if "units_probe1" in nwbfile.processing:
    units_probe1 = nwbfile.processing["units_probe1"]["units_probe1"]
    units_probe1_df = units_probe1.to_dataframe()
    spike_times_probe1 = units_probe1_df["spike_times"]

    print("\nProbe 1 units loaded")
    print("Number of units:", len(units_probe1_df))
    print("Columns:", list(units_probe1_df.columns))
    print(units_probe1_df.head())

# ============================================================
# 10. Example trial variable
# ============================================================
stim_on_times = None

if trials_df is not None and "imec_stim_on" in trials_df.columns:
    stim_on_times = trials_df["imec_stim_on"]
    print("\nStim onset times variable created: stim_on_times")

# ============================================================
# 11. Optional quick summaries
# ============================================================
probe0_firing_rate = None
probe1_firing_rate = None

if units_probe0_df is not None and "firing_rate" in units_probe0_df.columns:
    probe0_firing_rate = units_probe0_df["firing_rate"]

if units_probe1_df is not None and "firing_rate" in units_probe1_df.columns:
    probe1_firing_rate = units_probe1_df["firing_rate"]

# ============================================================
# 12. Keep file open while exploring variables in Spyder
# ============================================================
print("\nDone. Variables should now appear in the Spyder Variable Explorer.")
print("When you are completely finished, run: io.close()")

#%% plot example raster 

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import transforms
from pynwb import NWBHDF5IO


# =============================================================================
# USER SETTINGS
# =============================================================================

NWB_PATH = Path(r'L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/NWB/999770_20251111_2probes_2.nwb') 
OUTPUT_DIR = Path(r'L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/neuro_plots')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_COLOR_MAP = {
    "ACAd": "#40A666",
    "MOs":  "#1F9D5A",
    "PL":   "#2FA850",
    "ILA":  "#59B363",
    "ACB":  "#80CDF8",
    "CP":   "#98D6F9",
    "GPe":  "#8599CC",
    "SI":   "#A2B1D8",
    "SSp-ul5": "#188064",
    "SSp-ul6a": "#188064",
    "MOp5": "#1F9D5A",
    "scwm": "#cccccc",
    "NA": "#B0B0B0",
}
DEFAULT_REGION_COLOR = "#B0B0B0"

SORT_ASCENDING_BY_DEPTH = True
SHOW_REGION_LABELS = True
LABEL_SIDE = "left"
BACKGROUND_COLOR = "white"

CONTINUOUS_WINDOW_SEC = 8.0
USE_MIDDLE_PORTION_ONLY = True
MIDDLE_START_FRAC = 0.25
MIDDLE_STOP_FRAC = 0.75

BIN_SIZE = 0.02
T_BEFORE = 1.0
T_AFTER = 2.0
SMOOTHING_SIGMA_BINS = 1.0

EXAMPLE_REGIONS = ["ACAd", "MOs", "PL", "ACB", "CP"]

SAVE_PNG = True
SAVE_SVG = False
DPI = 300


# =============================================================================
# HELPERS
# =============================================================================

def maybe_save(fig, name):
    if SAVE_PNG:
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    if SAVE_SVG:
        fig.savefig(OUTPUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())


def gaussian_smooth_1d(x, sigma_bins=1.0):
    if sigma_bins is None or sigma_bins <= 0:
        return x.copy()
    radius = int(np.ceil(4 * sigma_bins))
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs ** 2) / (2 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def zscore_rows(mat):
    out = mat.astype(float).copy()
    means = np.nanmean(out, axis=1, keepdims=True)
    stds = np.nanstd(out, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (out - means) / stds


def collapse_probe_electrodes_to_sites(electrodes_df, probe_label):
    probe_elec = electrodes_df[electrodes_df["probe_label"] == probe_label].copy()

    if len(probe_elec) == 0:
        raise ValueError(f"No electrodes found for {probe_label}")
    if "site_index" not in probe_elec.columns:
        raise ValueError(f"{probe_label}: missing 'site_index'")

    sort_cols = [c for c in ["site_index", "channel_id", "pad_side"] if c in probe_elec.columns]
    probe_elec = probe_elec.sort_values(sort_cols).reset_index(drop=True)

    keep_cols = [
        "probe_label", "site_index", "distance_to_tip", "depth_um",
        "ap_coords_vox", "dv_coords_vox", "ml_coords_vox",
        "structure_id", "acronym", "brain_region", "dist_to_structure", "location"
    ]
    keep_cols = [c for c in keep_cols if c in probe_elec.columns]

    rows = []
    for site_idx, g in probe_elec.groupby("site_index", sort=True):
        row = {}
        for col in keep_cols:
            row[col] = g.iloc[0][col]

        if "pad_side" in g.columns and "channel_id" in g.columns:
            left_rows = g[g["pad_side"] == "left"]
            right_rows = g[g["pad_side"] == "right"]
            row["channel_id_left"] = left_rows.iloc[0]["channel_id"] if len(left_rows) > 0 else np.nan
            row["channel_id_right"] = right_rows.iloc[0]["channel_id"] if len(right_rows) > 0 else np.nan

        row["n_pad_rows"] = len(g)
        rows.append(row)

    sites_df = pd.DataFrame(rows).sort_values("site_index").reset_index(drop=True)
    sites_df["site_order"] = np.arange(len(sites_df))
    return sites_df


def map_units_to_sites(units_df, sites_df, region_color_map):
    units_mapped = units_df.copy().reset_index(drop=False)
    units_mapped.rename(columns={"index": "unit_index"}, inplace=True)

    mapped_site_index = []
    mapped_depth_um = []
    mapped_distance_to_tip = []
    mapped_structure_id = []
    mapped_acronym = []
    mapped_brain_region = []
    mapped_region_color = []

    n_sites = len(sites_df)

    for _, row in units_mapped.iterrows():
        peak_ch = row["peak_channel_id"]

        if pd.isna(peak_ch):
            mapped_site_index.append(np.nan)
            mapped_depth_um.append(np.nan)
            mapped_distance_to_tip.append(np.nan)
            mapped_structure_id.append(np.nan)
            mapped_acronym.append("NA")
            mapped_brain_region.append("NA")
            mapped_region_color.append(DEFAULT_REGION_COLOR)
            continue

        peak_ch = int(peak_ch)

        if 0 <= peak_ch < n_sites:
            site_row = sites_df.iloc[peak_ch]
            acronym = str(site_row["acronym"]) if "acronym" in site_row.index else "NA"
            brain_region = str(site_row["brain_region"]) if "brain_region" in site_row.index else "NA"
            sid = site_row["structure_id"] if "structure_id" in site_row.index else np.nan
            if pd.notna(sid):
                sid = int(sid)

            mapped_site_index.append(site_row["site_index"] if "site_index" in site_row.index else np.nan)
            mapped_depth_um.append(site_row["depth_um"] if "depth_um" in site_row.index else np.nan)
            mapped_distance_to_tip.append(site_row["distance_to_tip"] if "distance_to_tip" in site_row.index else np.nan)
            mapped_structure_id.append(sid)
            mapped_acronym.append(acronym)
            mapped_brain_region.append(brain_region)
            mapped_region_color.append(region_color_map.get(acronym, DEFAULT_REGION_COLOR))
        else:
            mapped_site_index.append(np.nan)
            mapped_depth_um.append(np.nan)
            mapped_distance_to_tip.append(np.nan)
            mapped_structure_id.append(np.nan)
            mapped_acronym.append("NA")
            mapped_brain_region.append("NA")
            mapped_region_color.append(DEFAULT_REGION_COLOR)

    units_mapped["mapped_site_index"] = mapped_site_index
    units_mapped["depth_um"] = mapped_depth_um
    units_mapped["distance_to_tip"] = mapped_distance_to_tip
    units_mapped["structure_id"] = mapped_structure_id
    units_mapped["acronym"] = mapped_acronym
    units_mapped["brain_region"] = mapped_brain_region
    units_mapped["region_color"] = mapped_region_color
    return units_mapped


def sort_units_for_raster(units_mapped_df, ascending=True):
    sort_cols = []
    if "depth_um" in units_mapped_df.columns:
        sort_cols.append("depth_um")
    if "peak_channel_id" in units_mapped_df.columns:
        sort_cols.append("peak_channel_id")

    if len(sort_cols) == 0:
        out = units_mapped_df.copy().reset_index(drop=True)
    else:
        out = units_mapped_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    out["plot_y"] = np.arange(1, len(out) + 1)
    return out


def get_region_blocks(units_sorted_df):
    blocks = []
    if units_sorted_df is None or len(units_sorted_df) == 0:
        return blocks

    acronyms = units_sorted_df["acronym"].fillna("NA").tolist()
    colors = units_sorted_df["region_color"].fillna(DEFAULT_REGION_COLOR).tolist()
    yvals = units_sorted_df["plot_y"].tolist()

    start_idx = 0
    for i in range(1, len(acronyms)):
        if acronyms[i] != acronyms[start_idx]:
            blocks.append({
                "acronym": acronyms[start_idx],
                "color": colors[start_idx],
                "y_start": yvals[start_idx],
                "y_end": yvals[i - 1],
                "y_center": 0.5 * (yvals[start_idx] + yvals[i - 1]),
            })
            start_idx = i

    blocks.append({
        "acronym": acronyms[start_idx],
        "color": colors[start_idx],
        "y_start": yvals[start_idx],
        "y_end": yvals[-1],
        "y_center": 0.5 * (yvals[start_idx] + yvals[-1]),
    })
    return blocks


def count_spikes_in_window(units_df, t_start, t_stop):
    total = 0
    for spikes in units_df["spike_times"]:
        spikes = np.asarray(spikes, dtype=float)
        total += np.sum((spikes >= t_start) & (spikes <= t_stop))
    return int(total)


def extract_aligned_counts(spike_times, event_times, t_before, t_after, bin_size):
    rel_bins = np.arange(-t_before, t_after + bin_size, bin_size)
    n_bins = len(rel_bins) - 1
    counts = np.zeros((len(event_times), n_bins), dtype=float)

    for i, ev in enumerate(event_times):
        rel_spikes = spike_times - ev
        hist, _ = np.histogram(rel_spikes, bins=rel_bins)
        counts[i, :] = hist

    bin_centers = rel_bins[:-1] + 0.5 * bin_size
    return counts, bin_centers


def choose_example_unit(units_df_region, event_times, t_before, t_after, bin_size):
    best_idx = None
    best_score = -np.inf

    for idx, row in units_df_region.iterrows():
        spike_times = np.asarray(row["spike_times"], dtype=float)
        counts, bin_centers = extract_aligned_counts(spike_times, event_times, t_before, t_after, bin_size)
        rates = counts / bin_size
        mean_rate = rates.mean(axis=0)
        pre = mean_rate[bin_centers < 0]
        post = mean_rate[bin_centers >= 0]
        if len(pre) == 0 or len(post) == 0:
            continue
        score = abs(np.nanmean(post) - np.nanmean(pre))
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def get_valid_times_from_trials(df, time_col):
    if time_col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[time_col], errors="coerce")
    vals = vals[(vals >= 0) & (~vals.isna())]
    return vals.to_numpy(dtype=float)


# =============================================================================
# LOAD DATA
# =============================================================================

if not NWB_PATH.exists():
    raise FileNotFoundError(f"NWB file not found: {NWB_PATH}")

io = NWBHDF5IO(str(NWB_PATH), "r")
nwbfile = io.read()

trials_df = nwbfile.trials.to_dataframe() if nwbfile.trials is not None else None
electrodes_df = nwbfile.electrodes.to_dataframe() if nwbfile.electrodes is not None else None
units_probe0_df = nwbfile.processing["units_probe0"]["units_probe0"].to_dataframe() if "units_probe0" in nwbfile.processing else None
units_probe1_df = nwbfile.processing["units_probe1"]["units_probe1"].to_dataframe() if "units_probe1" in nwbfile.processing else None

if trials_df is None or len(trials_df) == 0:
    raise ValueError("No trials found.")
if electrodes_df is None or len(electrodes_df) == 0:
    raise ValueError("No electrodes found.")
if units_probe0_df is None or units_probe1_df is None:
    raise ValueError("Both probe unit tables are required.")

probe0_sites_df = collapse_probe_electrodes_to_sites(electrodes_df, "probe0")
probe1_sites_df = collapse_probe_electrodes_to_sites(electrodes_df, "probe1")

units_probe0_mapped_df = map_units_to_sites(units_probe0_df, probe0_sites_df, REGION_COLOR_MAP)
units_probe1_mapped_df = map_units_to_sites(units_probe1_df, probe1_sites_df, REGION_COLOR_MAP)

units_probe0_sorted_df = sort_units_for_raster(units_probe0_mapped_df, ascending=SORT_ASCENDING_BY_DEPTH)
units_probe1_sorted_df = sort_units_for_raster(units_probe1_mapped_df, ascending=SORT_ASCENDING_BY_DEPTH)

probe0_region_blocks = get_region_blocks(units_probe0_sorted_df)
probe1_region_blocks = get_region_blocks(units_probe1_sorted_df)

# Trial subsets
rewarded_trials_df = trials_df[trials_df["reward"] == True].copy()
punished_trials_df = trials_df[trials_df["punishment"] == True].copy()

# Event sets
align_sets = {
    "stim_on_all": get_valid_times_from_trials(trials_df, "imec_stim_on"),
    "stim_on_rewarded": get_valid_times_from_trials(rewarded_trials_df, "imec_stim_on"),
    "stim_on_punished": get_valid_times_from_trials(punished_trials_df, "imec_stim_on"),
    "reward_on": get_valid_times_from_trials(rewarded_trials_df, "imec_reward_on"),
    "punishment_on": get_valid_times_from_trials(punished_trials_df, "imec_punishment_on"),
    "blue_led_on": get_valid_times_from_trials(trials_df, "imec_blue_led_on"),
    "first_lick_rewarded": get_valid_times_from_trials(rewarded_trials_df, "imec_lick"),
    "first_lick_punished": get_valid_times_from_trials(punished_trials_df, "imec_lick"),
}

for key, vals in align_sets.items():
    print(f"{key}: {len(vals)} events")

event_colors = {
    "imec_blue_led_on": "deepskyblue",
    "imec_blue_led_off": "steelblue",
    "imec_stim_on": "orange",
    "imec_stim_off": "darkorange",
    "imec_reward_on": "limegreen",
    "imec_reward_off": "green",
    "imec_punishment_on": "red",
    "imec_punishment_off": "darkred",
    "imec_lick": "black",
}

event_labels = {
    "imec_blue_led_on": "LED on",
    "imec_blue_led_off": "LED off",
    "imec_stim_on": "stim on",
    "imec_stim_off": "stim off",
    "imec_reward_on": "reward on",
    "imec_reward_off": "reward off",
    "imec_punishment_on": "punishment on",
    "imec_punishment_off": "punishment off",
    "imec_lick": "lick",
}

imec_event_columns = list(event_colors.keys())

# =============================================================================
# PART A — CONTINUOUS 8 s WINDOW WITH MOST SPIKES
# =============================================================================

session_start = float(trials_df["start_time"].min())
session_stop = float(trials_df["stop_time"].max())
session_duration = session_stop - session_start

step = CONTINUOUS_WINDOW_SEC / 4

if USE_MIDDLE_PORTION_ONLY:
    scan_start = session_start + MIDDLE_START_FRAC * session_duration
    scan_stop = session_start + MIDDLE_STOP_FRAC * session_duration
else:
    scan_start = session_start
    scan_stop = session_stop

candidate_windows = []
t0s = np.arange(scan_start, max(scan_start, scan_stop - CONTINUOUS_WINDOW_SEC), step)

for t0 in t0s:
    t1 = t0 + CONTINUOUS_WINDOW_SEC
    n0 = count_spikes_in_window(units_probe0_sorted_df, t0, t1)
    n1 = count_spikes_in_window(units_probe1_sorted_df, t0, t1)
    candidate_windows.append({
        "window_start": t0,
        "window_stop": t1,
        "spikes_probe0": n0,
        "spikes_probe1": n1,
        "spikes_total": n0 + n1,
    })

continuous_windows_df = pd.DataFrame(candidate_windows).sort_values("spikes_total", ascending=False).reset_index(drop=True)
selected_window = continuous_windows_df.iloc[0]
window_start = float(selected_window["window_start"])
window_stop = float(selected_window["window_stop"])
window_duration = window_stop - window_start

window_event_times_relative = {}
for col in imec_event_columns:
    if col in trials_df.columns:
        vals = pd.to_numeric(trials_df[col], errors="coerce")
        vals = vals[(vals >= window_start) & (vals <= window_stop)]
        vals = np.sort(vals.dropna().to_numpy(dtype=float))
        if len(vals) > 0:
            window_event_times_relative[col] = vals - window_start


def plot_continuous_raster(units_sorted_df, region_blocks, probe_name):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    for _, row in units_sorted_df.iterrows():
        spikes = np.asarray(row["spike_times"], dtype=float)
        spikes_in = spikes[(spikes >= window_start) & (spikes <= window_stop)] - window_start
        y = row["plot_y"]
        if spikes_in.size > 0:
            ax.vlines(spikes_in, y - 0.38, y + 0.38, color="black", linewidth=0.45)

    used_labels = set()
    for event_name, times in window_event_times_relative.items():
        color = event_colors.get(event_name, "gray")
        label = event_labels.get(event_name, event_name)
        for i, t in enumerate(times):
            if i == 0 and label not in used_labels:
                ax.axvline(t, color=color, linewidth=1.2, alpha=0.85, label=label)
                used_labels.add(label)
            else:
                ax.axvline(t, color=color, linewidth=1.2, alpha=0.85)

    ax.set_xlim(0, window_duration)
    ax.set_ylim(0.5, len(units_sorted_df) + 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Time from window start (s)")
    ax.set_ylabel("Units")
    ax.set_title(f"{probe_name} - continuous {window_duration:.1f}s window")
    ax.set_yticks([])

    text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    band_x = -0.09 * window_duration
    band_w = 0.035 * window_duration
    text_x_axes = -0.08

    for block in region_blocks:
        rect = Rectangle(
            (band_x, block["y_start"] - 0.5),
            band_w,
            block["y_end"] - block["y_start"] + 1.0,
            facecolor=block["color"],
            edgecolor="none",
            alpha=0.95,
            clip_on=False,
        )
        ax.add_patch(rect)
        if SHOW_REGION_LABELS:
            ax.text(text_x_axes, block["y_center"], block["acronym"],
                    transform=text_transform, va="center", ha="right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.tight_layout()
    maybe_save(fig, f"{probe_name}_continuous_raster_8s_best")
    plt.show()


plot_continuous_raster(units_probe0_sorted_df, probe0_region_blocks, "probe0")
plot_continuous_raster(units_probe1_sorted_df, probe1_region_blocks, "probe1")

# =============================================================================
# PART B — HEATMAPS FOR ALL REQUESTED ALIGNMENTS
# =============================================================================

def make_population_heatmap(units_sorted_df, probe_name, event_times, align_name):
    if len(event_times) == 0:
        print(f"{probe_name} {align_name}: no events, skipping heatmap")
        return None, None, None

    unit_mean_rates = []
    bin_centers = None

    for _, row in units_sorted_df.iterrows():
        spike_times = np.asarray(row["spike_times"], dtype=float)
        counts, bin_centers = extract_aligned_counts(spike_times, event_times, T_BEFORE, T_AFTER, BIN_SIZE)
        rates = counts / BIN_SIZE
        mean_rate = rates.mean(axis=0)
        mean_rate = gaussian_smooth_1d(mean_rate, SMOOTHING_SIGMA_BINS)
        unit_mean_rates.append(mean_rate)

    heatmap = np.vstack(unit_mean_rates)
    heatmap_z = zscore_rows(heatmap)

    fig = plt.figure(figsize=(10, 8), dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.08, 1.0], wspace=0.05)
    ax_band = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    for _, row in units_sorted_df.iterrows():
        y = row["plot_y"]
        ax_band.add_patch(Rectangle((0, y - 0.5), 1, 1, color=row["region_color"], ec="none"))
    ax_band.set_xlim(0, 1)
    ax_band.set_ylim(0.5, len(units_sorted_df) + 0.5)
    ax_band.invert_yaxis()
    ax_band.set_xticks([])
    ax_band.set_yticks([])
    ax_band.set_frame_on(False)

    im = ax.imshow(
        heatmap_z,
        aspect="auto",
        origin="upper",
        extent=[bin_centers[0], bin_centers[-1], len(units_sorted_df) + 0.5, 0.5],
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
    )
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(f"Time from {align_name} (s)")
    ax.set_ylabel("Units")
    ax.set_title(f"{probe_name} - {align_name}")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("z-scored firing")

    plt.tight_layout()
    maybe_save(fig, f"{probe_name}_heatmap_{align_name}")
    plt.show()

    return heatmap, heatmap_z, bin_centers


heatmap_results = {}

for probe_name, units_df in [("probe0", units_probe0_sorted_df), ("probe1", units_probe1_sorted_df)]:
    for align_name, event_times in align_sets.items():
        heatmap_results[(probe_name, align_name)] = make_population_heatmap(
            units_df, probe_name, event_times, align_name
        )

# =============================================================================
# PART C — EXAMPLE UNIT PLOTS FOR ALL REQUESTED REGIONS/EVENTS
# =============================================================================

def plot_example_unit(unit_row, event_times, region_name, probe_name, align_name):
    if len(event_times) == 0:
        return

    spike_times = np.asarray(unit_row["spike_times"], dtype=float)
    counts, bin_centers = extract_aligned_counts(
        spike_times, event_times, T_BEFORE, T_AFTER, BIN_SIZE
    )
    mean_rate = (counts / BIN_SIZE).mean(axis=0)
    mean_rate = gaussian_smooth_1d(mean_rate, SMOOTHING_SIGMA_BINS)

    fig = plt.figure(figsize=(7, 6), dpi=DPI)
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
    ax_raster = fig.add_subplot(gs[0, 0])
    ax_psth = fig.add_subplot(gs[1, 0], sharex=ax_raster)

    for i, ev in enumerate(event_times):
        rel_spikes = spike_times - ev
        rel_spikes = rel_spikes[(rel_spikes >= -T_BEFORE) & (rel_spikes <= T_AFTER)]
        if rel_spikes.size > 0:
            ax_raster.vlines(rel_spikes, i + 0.6, i + 1.4, color="black", linewidth=0.5)

    ax_raster.axvline(0, color="red", linewidth=1.2)
    ax_raster.set_ylabel("Trials")
    ax_raster.set_title(f"{probe_name} | {region_name} | {align_name}")
    ax_raster.invert_yaxis()

    ax_psth.plot(bin_centers, mean_rate, color=unit_row["region_color"], linewidth=2)
    ax_psth.axvline(0, color="red", linewidth=1.2)
    ax_psth.set_xlabel(f"Time from {align_name} (s)")
    ax_psth.set_ylabel("Hz")

    plt.tight_layout()
    maybe_save(fig, f"{probe_name}_{region_name}_{align_name}_example_unit")
    plt.show()


for probe_name, units_df in [("probe0", units_probe0_sorted_df), ("probe1", units_probe1_sorted_df)]:
    for region in EXAMPLE_REGIONS:
        region_units = units_df[units_df["acronym"] == region]
        if len(region_units) == 0:
            print(f"{probe_name}: no units in {region}")
            continue

        for align_name, event_times in align_sets.items():
            if len(event_times) == 0:
                print(f"{probe_name} {region} {align_name}: no events")
                continue

            idx = choose_example_unit(region_units, event_times, T_BEFORE, T_AFTER, BIN_SIZE)
            if idx is None:
                print(f"{probe_name} {region} {align_name}: no suitable example unit")
                continue

            plot_example_unit(region_units.loc[idx], event_times, region, probe_name, align_name)

# =============================================================================
# DONE
# =============================================================================

print("\nDone.")
print("Useful variables in Spyder:")
print("- trials_df")
print("- electrodes_df")
print("- units_probe0_sorted_df")
print("- units_probe1_sorted_df")
print("- rewarded_trials_df")
print("- punished_trials_df")
print("- align_sets")
print("- continuous_windows_df")
print("- selected_window")
print("- heatmap_results")
print("\nWhen finished, run:")
print("io.close()")