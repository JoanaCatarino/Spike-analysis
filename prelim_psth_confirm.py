# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:04:39 2026

@author: JoanaCatarino

Try to align some data

"""
#%% Step 1

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- File paths ----------

# Data to analyze
animal = 986170
day = 3
probe = "imec0"

# paths 

ks_dir = r"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/986170/986170_day3_g0_imec0/sorter_kilosort4/sorter_output"

ttl_align_dir = r"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/986170/986170_day3_g0_imec0/TTL_alignment"
ttl_file = os.path.join(ttl_align_dir, f"{animal}_day{day}_{probe}_ttl_aligned_imec_time.npz")

#out_dir = os.path.join(ttl_align_dir, "PSTH_good_units")
#os.makedirs(out_dir, exist_ok=True)

'''
Within the ks_dir path we expect to have:
    spike_times.npy - spike times in samples, not seconds
    spike_clusters.npy - cluster id for each spike
    params.py - contains things like sample rate
    cluster_group.tsv or cluster_KSLable.tsv - labels 'good', etc
Within the ttl_align_dir we expect to have:
    .npz file that contains arrays like ttl_rew_d_imec, etc
'''

#%% Step 2

# ---------- PSTH settings ----------

T_PRE = 1.0          # seconds before event
T_POST = 1.5         # seconds after event
BIN = 0.010          # seconds (10 ms)
SMOOTH_BINS = 3      # moving average smoothing over bins (0/1 disables)
MAKE_RASTER = True   # raster + psth if True, psth only if False


# Map "plot label" -> variable name saved in the TTL .npz
events_map = {
    "Reward": "ttl_rew_d_imec",
    "Punishment": "ttl_pun_d_imec",
    "Stim":   "ttl_stim_d_imec",
    "Blue LED": "ttl_led_d_imec",
    "Laser":  "ttl_laser_a_imec",
}


#%% Step 3

# ---------- Helpers ----------

# 1) Read sample rate from kilosort params.py

def load_sample_rate_from_params(ks_dir):
    params_path = os.path.join(ks_dir, "params.py")
    if not os.path.exists(params_path):
        return None
    with open(params_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sample_rate"):
                try:
                    return float(line.split("=", 1)[1])
                except Exception:
                    pass
    return None



# 2) Load spike times and cluster IDs

def load_kilosort_spikes(ks_dir):
    st_path = os.path.join(ks_dir, "spike_times.npy")
    sc_path = os.path.join(ks_dir, "spike_clusters.npy")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Missing {st_path}")
    if not os.path.exists(sc_path):
        raise FileNotFoundError(f"Missing {sc_path}")
    spike_times = np.load(st_path).squeeze().astype(np.int64)     # samples
    spike_clusters = np.load(sc_path).squeeze().astype(np.int64)  # cluster id
    return spike_times, spike_clusters



# 3) Find 'good' clusters from the TSV

def load_good_clusters(ks_dir):
    """
    Your files have columns: ['cluster_id', 'KSLabel'].
    We'll read either cluster_group.tsv or cluster_KSLabel.tsv (same format here)
    and return cluster_ids where KSLabel == 'good'.
    """
    for fname in ["cluster_group.tsv", "cluster_KSLabel.tsv"]:
        path = os.path.join(ks_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            if "cluster_id" not in df.columns or "KSLabel" not in df.columns:
                raise ValueError(f"{fname} columns are {df.columns.tolist()}, expected cluster_id and KSLabel")
            good = df.loc[df["KSLabel"].astype(str).str.lower() == "good", "cluster_id"].to_numpy(dtype=int)
            print(f"[KS] Using {fname}: good units = {good.size}")
            return good

    raise FileNotFoundError("Could not find cluster_group.tsv or cluster_KSLabel.tsv in Kilosort folder.")



# 4) Load TTL event times from the .npz file

def load_ttl_events(ttl_npz_path, events_map):
    if not os.path.exists(ttl_npz_path):
        raise FileNotFoundError(f"TTL file not found: {ttl_npz_path}")

    z = np.load(ttl_npz_path, allow_pickle=True)
    available = set(z.files)

    out = {}
    for label, key in events_map.items():
        if key not in available:
            print(f"[WARN] TTL key missing in NPZ: {key} (skipping '{label}')")
            continue
        pairs = z[key]
        out[label] = np.asarray(pairs)[:, 0].astype(float) if pairs.size else np.array([], dtype=float)  # rises
    return out



# 5) Moving Average smoothing

def moving_average(x, w):
    if w is None or w <= 1:
        return x
    w = int(w)
    k = np.ones(w, dtype=float) / w
    return np.convolve(x, k, mode="same")
# if SMOOTH_BINS = 3, each point becomes the average of itself + neighbors (simple smoothing)



# 6) Compute PSTH and raster data for one unit and one event type

def compute_psth(spike_times_s, event_times_s, t_pre, t_post, bin_s):
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    event_times_s = np.asarray(event_times_s, dtype=float)

    edges = np.arange(-t_pre, t_post + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    if event_times_s.size == 0 or spike_times_s.size == 0:
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

    rate = counts / (event_times_s.size * bin_s)
    return centers, rate, raster



# 7) Plotting one unit across all event types

def plot_unit_psth_show(unit_id, spike_times_s, events_dict,
                        t_pre=1.0, t_post=2.0, bin_s=0.01, smooth_bins=3,
                        make_raster=True):

    event_labels = [k for k, v in events_dict.items() if v.size > 0]
    if len(event_labels) == 0:
        print(f"[SKIP] Unit {unit_id}: no events available.")
        return

    n = len(event_labels)
    fig_h = 3.2 * n if make_raster else 2.4 * n
    fig = plt.figure(figsize=(12, fig_h))
    fig.suptitle(f"GOOD Unit {unit_id} — PSTH aligned to TTL events (IMEC time)", y=0.995)

    for i, label in enumerate(event_labels, start=1):
        ev = events_dict[label]
        centers, rate, raster = compute_psth(spike_times_s, ev, t_pre, t_post, bin_s)
        rate_sm = moving_average(rate, smooth_bins)

        if make_raster:
            ax_r = fig.add_subplot(n, 2, (i - 1) * 2 + 1)
            for k, rel in enumerate(raster):
                if rel.size:
                    ax_r.vlines(rel, k + 0.5, k + 1.5)
            ax_r.axvline(0, linestyle="--", color='red')
            ax_r.set_xlim(-t_pre, t_post)
            ax_r.set_ylim(0,300)
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
        ax_p.axvline(0, linestyle="--", color='red')
        ax_p.set_xlim(-t_pre, t_post)
        ax_p.set_ylabel("Hz")
        ax_p.set_title(f"{label} — PSTH (bin={bin_s*1000:.0f} ms, n={len(ev)})")
        if i == n:
            ax_p.set_xlabel("Time from event (s)")
        else:
            ax_p.set_xticklabels([])

    fig.tight_layout(rect=[0, 0, 1, 0.985])
    plt.show()



# ============================================================
# MAIN
# ============================================================

def main():
    # --- Spike times / clusters ---
    sr = load_sample_rate_from_params(ks_dir)
    if sr is None:
        raise RuntimeError("Could not read sample_rate from params.py in Kilosort folder.")

    spike_times_samp, spike_clusters = load_kilosort_spikes(ks_dir)
    spike_times_s = spike_times_samp / sr

    # --- Good units ---
    good_units = load_good_clusters(ks_dir)
    print("[KS] sample_rate:", sr)
    print("[KS] spikes:", spike_times_samp.size)
    print("[KS] good units:", good_units.size)

    # --- TTL events in IMEC seconds ---
    events_dict = load_ttl_events(ttl_file, events_map)

    print("[TTL] event counts:")
    for k, v in events_dict.items():
        print(f"  {k}: {v.size}")

    # --- Plot PSTH for each GOOD unit separately ---
    MAX_UNITS_TO_PLOT = 50  # change later

    for i, unit in enumerate(good_units[:MAX_UNITS_TO_PLOT], start=1):
        st_unit = spike_times_s[spike_clusters == unit]
        if st_unit.size == 0:
            continue
        print(f"[PLOT] {i}/{MAX_UNITS_TO_PLOT} unit={unit}, spikes={st_unit.size}")
        plot_unit_psth_show(
            unit_id=unit,
            spike_times_s=st_unit,
            events_dict=events_dict,
            t_pre=T_PRE,
            t_post=T_POST,
            bin_s=BIN,
            smooth_bins=SMOOTH_BINS,
            make_raster=MAKE_RASTER,
        )

    

if __name__ == "__main__":
    main()































