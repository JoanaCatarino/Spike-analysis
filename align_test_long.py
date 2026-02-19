# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:05:40 2026

@author: JoanaCatarino
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- File paths ----------

base_nidq = "L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Raw_data/999770/999770_day1_g0"
base_imec = base_nidq + "/999770_day1_1_g0_imec1"

# Nidaq files
path_nidq_meta = base_nidq + "/999770_day1_1_g0_t0.nidq.meta"
path_nidq_bin = base_nidq + "/999770_day1_1_g0_t0.nidq.bin"

# IMEC files
path_imec_ap_meta = base_imec + "/999770_day1_1_g0_t0.imec1.ap.meta"
path_imec_ap_bin = base_imec + "/999770_day1_1_g0_t0.imec1.ap.bin"
path_imec_lf_meta = base_imec + "/999770_day1_1_g0_t0.imec1.lf.meta"
path_imec_lf_bin = base_imec + "/999770_day1_1_g0_t0.imec1.lf.bin"


#%% 
# ---------- Load .meta files for nidq and imec ----------

def load_meta_df(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                rows.append((key, value))
    return pd.DataFrame(rows, columns=["key", "value"])

def get_meta_value(meta_df, key, cast=None, default=None):
    """Return value for `key` from a (key, value) meta DataFrame."""
    s = meta_df.loc[meta_df["key"] == key, "value"]
    if s.empty:
        return default
    val = s.iloc[0]
    if cast is not None:
        try:
            return cast(val)
        except (TypeError, ValueError):
            return default
    return val


nidq_meta = load_meta_df(path_nidq_meta)
imec_meta = load_meta_df(path_imec_ap_meta) 


#%% 
# ---------- Get number os channels save and sample rate ----------

nSavedChannels = get_meta_value(nidq_meta, "nSavedChans", cast=int) # number of channels saved in the nidq
totalChannels = get_meta_value(imec_meta, "nSavedChans", cast=int)

# sample rates 
nidq_fs = get_meta_value(nidq_meta, "niSampRate", cast=float)
if nidq_fs is None:
    nidq_fs = get_meta_value(nidq_meta, "imSampRate", cast=float)

imec_fs = get_meta_value(imec_meta, "imSampRate", cast=float)

print(f"[META] nidq_fs={nidq_fs}, imec_fs={imec_fs}, nidq_nCh={nSavedChannels}, imec_nCh={totalChannels}")


#%%
# ---------- Load .bin files (imec sync + nidq digital) --------

def load_spikeglx_bin_memmap(bin_path, meta_df, dtype=np.int16):
    n_channels = get_meta_value(meta_df, "nSavedChans", cast=int)
    if n_channels is None:
        raise KeyError("Key 'nSavedChans' not found in meta file.")

    # compute number of samples from file size (safer than guessing)
    file_bytes = os.path.getsize(bin_path)
    bytes_per_sample_allch = np.dtype(dtype).itemsize * n_channels

    if file_bytes % bytes_per_sample_allch != 0:
        raise ValueError(
            f"File size not divisible by (dtype * n_channels). "
            f"Check dtype ({dtype}) and nSavedChans ({n_channels})."
        )

    n_samples = file_bytes // bytes_per_sample_allch

    data = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_samples, n_channels))
    return data


nidq_bin = load_spikeglx_bin_memmap(path_nidq_bin, nidq_meta)
imec_ap_bin = load_spikeglx_bin_memmap(path_imec_ap_bin, imec_meta)


#%% 

#Extract IMEC SY + NIDQ analog/Digital 


#  ---------- IMEC AP: sync is typically the last saved channel ("SY") ----------
imec_sync = imec_ap_bin[:, -1]  # int16 vector
plt.plot(imec_sync[:1000000])  # Sanity check to quickly plot the sync channel and check is not empty


# ---------- NiDaq: split analog + digital ----------

# By convention for nidq, the last saved channel is a packed digital word(XD0) and the first 3 are analog channels (XA0-XA2)
nidq_dig_word = nidq_bin[:, -1].astype(np.uint16)     # (samples,)
nidq_analog_raw = nidq_bin[:, :-1].astype(np.float32) # (samples, n_analog)


# Scale analog int16 to volts (assuming NI range is [-10, 10] or similar)
ni_max = get_meta_value(nidq_meta, "niAiRangeMax", cast=float, default=10.0)
ni_min = get_meta_value(nidq_meta, "niAiRangeMin", cast=float, default=-10.0)


# int16 spans 65536 codes total; SpikeGLX/NI commonly maps full-scale to that span.
nidq_analog_volts = nidq_analog_raw * ((ni_max - ni_min) / 65536.0)


# Extract 16 digital lines (0-15) from the packed digital word
nidq_digital_bits = np.column_stack([(nidq_dig_word >> b) & 1 for b in range(16)]).astype(np.uint8)
# nidq_digital_bits[:, 0] = line 0, ..., nidq_digital_bits[:, 15] = line 15


#%%

# ---------- Channel mapping ----------

# Analog
ai_blue_led = nidq_analog_volts[:, 0]   # XA0
ai_sound    = nidq_analog_volts[:, 1]   # XA1
ai_laser    = nidq_analog_volts[:, 2]   # XA2

# Digital
di_blue_led = nidq_digital_bits[:, 4]   # DI4
di_stim     = nidq_digital_bits[:, 5]   # DI5
di_punish   = nidq_digital_bits[:, 6]   # DI6
di_reward   = nidq_digital_bits[:, 7]   # DI7


#%%

# ---------- TTL windows (rising + falling edges) ----------

# PART 1 - Extract TTL pulses

def ttl_windows(x, fs, thresh=0.5, min_dur_s=0.0):
    """
    Extract TTL pulses as:
      on_idx, off_idx, on_s, off_s, dur_s
    """
    x = np.asarray(x) # guarantees x is a NumPy array

    if x.dtype == bool:
        b = x
    else:
        b = x > thresh

    on = np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1 # we add + 1 because we detect the change at the next sample
    off = np.where((b[1:] == 0) & (b[:-1] == 1))[0] + 1

    # edge cases: starts high or ends high
    if b[0] == 1:
        on = np.r_[0, on]
    if b[-1] == 1:
        off = np.r_[off, len(b) - 1]


# - If the signal starts HIGH at sample 0, there was no rising edge detected, so we **manually add** onset at index 0.
# - If the signal ends HIGH, there was no falling edge detected, so we **manually add** an offset at the last sample.
# This makes sure each “high period” has a start and an end.

    # Make sure 'on' and 'off' pair properly
    n = min(len(on), len(off))
    on = on[:n]
    off = off[:n]

    # Remove incalid pairs (off must be after on)
    keep = off > on
    on, off = on[keep], off[keep]

    # Covert indices to seconds
    on_s = on / fs  # onset time in seconds
    off_s = off / fs # offset time in seconds
    dur_s = off_s - on_s # pulse duration in seconds

    # Optional: remove pulses shorter than 'min_dur_s' - currently set to 0s
    if min_dur_s > 0:
        k = dur_s >= min_dur_s
        on, off, on_s, off_s, dur_s = on[k], off[k], on_s[k], off_s[k], dur_s[k]

    return on, off, on_s, off_s, dur_s

# PART 2 - Put TTL pulses into a DataFrame

def ttl_df(name, x, fs, thresh=0.5, min_dur_s=0.0):
    on, off, on_s, off_s, dur_s = ttl_windows(x, fs, thresh=thresh, min_dur_s=min_dur_s)
    return pd.DataFrame({
        "signal": name, # Name of the TTL signal
        "on_idx": on,   # Onset sample index
        "off_idx": off, # Offset sample index
        "on_s": on_s,   # Onset time (seconds)
        "off_s": off_s, # Offset time (seconds)
        "dur_s": dur_s  # Duration (seconds)
    })


# PART 3 - Choose thresholds (Volts) for analog channels 

# Start with these; if Plot C shows they don't cross, adjust them.
thr_led_v   = 2
thr_sound_v = 0.5
thr_laser_v = 0.5

# Analogue is HIGH when voltage > 0.5 V

# PART 4 - Extract TTL windows for each signal
# Digital TTL windows
df_reward_d = ttl_df("DI7_reward", di_reward, nidq_fs, thresh=2)
df_punish_d = ttl_df("DI6_punish", di_punish, nidq_fs, thresh=2)
df_stim_d   = ttl_df("DI5_stim",   di_stim,   nidq_fs, thresh=2)
df_led_d    = ttl_df("DI4_led",    di_blue_led, nidq_fs, thresh=2)

# Analog TTL-like monitors (thresholded in volts)
df_led_a    = ttl_df("XA0_led_analog",   ai_blue_led, nidq_fs, thresh=thr_led_v)
df_sound_a  = ttl_df("XA1_sound_analog", ai_sound,    nidq_fs, thresh=thr_sound_v)
df_laser_a  = ttl_df("XA2_laser_analog", ai_laser,    nidq_fs, thresh=thr_laser_v)

# IMEC sync windows (SY int16, thresh=0 is fine)
df_imec_sync = ttl_df("IMEC_SY", imec_sync, imec_fs, thresh=2)
# IMEC SY is int16. Often 0 baseline and positive during pulses, so 'thresh=0' means “high whenever >0”.

print(f"[TTL] digital counts: reward={len(df_reward_d)}, punish={len(df_punish_d)}, stim={len(df_stim_d)}, led={len(df_led_d)}")
print(f"[TTL] analog  counts: led={len(df_led_a)}, sound={len(df_sound_a)}, laser={len(df_laser_a)}")
print(f"[TTL] imec SY count: {len(df_imec_sync)}")


#%%

# ---------- Align Nidq and Imec using sync onsets

def bin_edges_to_ms(edge_times_s, t_max_s, bin_ms=1.0):
    dt = bin_ms / 1000.0
    n = int(np.ceil(t_max_s / dt)) + 1
    y = np.zeros(n, dtype=np.float32)
    idx = (edge_times_s / dt).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    y[idx] = 1.0
    return y, dt

def best_nidq_sync_line(imec_edge_s, nidq_bits, nidq_fs, search_lines=range(16), t_search_s=60):
    """
    Find the NiDaq digital line whose rising-edge train matches IMEC SY best.
    """
    imec_edge_s = imec_edge_s[imec_edge_s <= t_search_s]
    if len(imec_edge_s) < 5:
        raise ValueError("Too few IMEC sync edges in the search window.")

    t_max = float(np.min([t_search_s, imec_edge_s[-1]]))
    imec_binned, dt = bin_edges_to_ms(imec_edge_s, t_max, bin_ms=1.0)

    best = {"line": None, "corr": -np.inf, "lag_s": None}
    n_samp_search = int(min(t_search_s * nidq_fs, nidq_bits.shape[0] - 1))

    for line in search_lines:
        nidq_line = nidq_bits[:n_samp_search, line].astype(bool)
        on_idx, _, on_s, _, _ = ttl_windows(nidq_line, nidq_fs, thresh=0.5)
        nidq_edge_s = on_s
        nidq_edge_s = nidq_edge_s[nidq_edge_s <= t_max]
        if len(nidq_edge_s) < 5:
            continue

        nidq_binned, _ = bin_edges_to_ms(nidq_edge_s, t_max, bin_ms=1.0)
        c = np.correlate(imec_binned, nidq_binned, mode="full")
        k = int(np.argmax(c))
        lag_bins = k - (len(nidq_binned) - 1)
        lag_s = lag_bins * dt

        if c[k] > best["corr"]:
            best = {"line": int(line), "corr": float(c[k]), "lag_s": float(lag_s)}

    if best["line"] is None:
        raise ValueError("Could not find a matching NiDaq sync line among the searched lines.")

    return best

def fit_time_mapping(nidq_edge_s, imec_edge_s, coarse_lag_s=0.0, n_fit=5000):
    """
    Fit linear mapping: t_imec ~= a * t_nidq + b
    using nearest-neighbor edge matching after a coarse lag shift.
    """
    nidq_shift = nidq_edge_s + coarse_lag_s
    imec = imec_edge_s

    j = np.searchsorted(imec, nidq_shift)
    j = np.clip(j, 1, len(imec) - 1)

    left = imec[j - 1]
    right = imec[j]
    choose_right = (np.abs(right - nidq_shift) < np.abs(nidq_shift - left))
    matched_imec = np.where(choose_right, right, left)

    n = min(len(nidq_edge_s), len(matched_imec), n_fit)
    x = nidq_edge_s[:n]
    y = matched_imec[:n]

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)

# IMEC SY onsets (rising edges)
imec_on_s = df_imec_sync["on_s"].values

# Find matching NiDaq sync line (digital)
best = best_nidq_sync_line(
    imec_edge_s=imec_on_s,
    nidq_bits=nidq_digital_bits,
    nidq_fs=nidq_fs,
    search_lines=range(16),
    t_search_s=60
)

nidq_sync_line = best["line"]
coarse_lag_s = best["lag_s"]

print(f"[ALIGN] Chosen NiDaq sync line: DI{nidq_sync_line} (coarse lag ~ {coarse_lag_s:.6f} s)")

# NiDaq sync onsets
nidq_sync_bool = nidq_digital_bits[:, nidq_sync_line].astype(bool)
nidq_sync_on_idx, _, nidq_sync_on_s, _, _ = ttl_windows(nidq_sync_bool, nidq_fs, thresh=0.5)

# Fit mapping
a, b = fit_time_mapping(nidq_sync_on_s, imec_on_s, coarse_lag_s=coarse_lag_s, n_fit=5000)
print(f"[ALIGN] Time mapping: t_imec ≈ {a:.12f} * t_nidq + {b:.6f}")

def nidq_time_to_imec_time(nidq_t_s):
    return a * np.asarray(nidq_t_s) + b

# Example: convert reward windows to IMEC time
df_reward_d["on_s_imec"]  = nidq_time_to_imec_time(df_reward_d["on_s"].values)
df_reward_d["off_s_imec"] = nidq_time_to_imec_time(df_reward_d["off_s"].values)
df_reward_d["dur_s_imec"] = df_reward_d["off_s_imec"] - df_reward_d["on_s_imec"]


#%% 

# ---------- Sanity check plots ----------


# ---- Plot settings ----
NIDQ_PLOT_SEC = 90
IMEC_PLOT_SEC = 90

nidq_n = int(NIDQ_PLOT_SEC * nidq_fs)
imec_n = int(IMEC_PLOT_SEC * imec_fs)

# Plot A: IMEC SY (first IMEC_PLOT_SEC seconds)
plt.figure(figsize=(12, 4))
plt.plot(np.arange(imec_n) / imec_fs, imec_sync[:imec_n])
plt.title(f"IMEC SY channel (first {IMEC_PLOT_SEC} s)")
plt.xlabel("Time (s)")
plt.ylabel("SY (int16)")
plt.tight_layout()
plt.show()

# Plot B: NiDaq digital TTLs stacked (first NIDQ_PLOT_SEC seconds)
t_nidq = np.arange(nidq_n) / nidq_fs
plt.figure(figsize=(12, 5))
plt.plot(t_nidq, di_blue_led[:nidq_n] + 0, label="DI4 LED")
plt.plot(t_nidq, di_stim[:nidq_n] + 2, label="DI5 STIM")
plt.plot(t_nidq, di_punish[:nidq_n] + 4, label="DI6 PUNISH")
plt.plot(t_nidq, di_reward[:nidq_n] + 6, label="DI7 REWARD")
plt.title(f"NiDaq digital TTLs (first {NIDQ_PLOT_SEC} s, stacked)")
plt.xlabel("Time (s)")
plt.yticks([])
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Plot C: Analog monitors + thresholds (first NIDQ_PLOT_SEC seconds)
plt.figure(figsize=(12, 4))
plt.plot(t_nidq, ai_blue_led[:nidq_n], label="XA0 LED analog")
plt.plot(t_nidq, np.full_like(t_nidq, thr_led_v), "--", label=f"thr={thr_led_v} V")
plt.title(f"Analog LED monitor (first {NIDQ_PLOT_SEC} s)")
plt.xlabel("Time (s)")
plt.ylabel("Volts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t_nidq, ai_sound[:nidq_n], label="XA1 Sound analog")
plt.plot(t_nidq, np.full_like(t_nidq, thr_sound_v), "--", label=f"thr={thr_sound_v} V")
plt.title(f"Analog Sound monitor (first {NIDQ_PLOT_SEC} s)")
plt.xlabel("Time (s)")
plt.ylabel("Volts")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(t_nidq, ai_laser[:nidq_n], label="XA2 Laser analog")
plt.plot(t_nidq, np.full_like(t_nidq, thr_laser_v), "--", label=f"thr={thr_laser_v} V")
plt.title(f"Analog Laser monitor (first {NIDQ_PLOT_SEC} s)")
plt.xlabel("Time (s)")
plt.ylabel("Volts")
plt.legend()
plt.tight_layout()
plt.show()

# Plot D: Alignment check — compare IMEC SY onsets vs mapped NiDaq sync onsets
nidq_on_s_in_imec = nidq_time_to_imec_time(nidq_sync_on_s)
m = min(300, len(imec_on_s), len(nidq_on_s_in_imec))

plt.figure(figsize=(12, 3.5))
plt.plot(imec_on_s[:m], np.zeros(m), "|", markersize=18, label="IMEC SY onsets")
plt.plot(nidq_on_s_in_imec[:m], np.ones(m), "|", markersize=18, label=f"NiDaq DI{nidq_sync_line} onsets (mapped)")
plt.title("Alignment sanity check (first pulses)")
plt.xlabel("IMEC time (s)")
plt.yticks([0, 1], ["IMEC", "NiDaq→IMEC"])
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Plot E: Alignment error (ms) for first pulses
err_ms = (nidq_on_s_in_imec[:m] - imec_on_s[:m]) * 1000
plt.figure(figsize=(12, 3.5))
plt.plot(err_ms)
plt.title("Sync onset timing error (NiDaq→IMEC) for first pulses")
plt.xlabel("Pulse #")
plt.ylabel("Error (ms)")
plt.tight_layout()
plt.show()

