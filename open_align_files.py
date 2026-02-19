# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 14:02:27 2025

@author: JoanaCatarino
"""
#%%# Step 1

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


#%% Step 2
# ---------- Load .meta files for nidq and imec ----------

def parse_meta(meta_path):
    """Parse a SpikeGLX .meta file into a dictionary."""
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, val = line.strip().split('=', maxsplit=1)
                meta[key] = val
    return meta

def meta_get(meta, key, cast=None, default=None):
    if key not in meta:
        return default
    v = meta[key]
    if cast is None:
        return v
    try:
        return cast(v)
    except Exception:
        return default


nidq_meta = parse_meta(path_nidq_meta)
imec_meta = parse_meta(path_imec_ap_meta)


#%% Step 3
# ---------- Get number os channels save and sample rate ----------

nSavedChannels = meta_get(nidq_meta, "nSavedChans", int) # number of channels saved in the nidq
totalChannels = meta_get(imec_meta, "nSavedChans", int)  # number of channels imec

# sample rates 
nidq_fs = meta_get(nidq_meta, "niSampRate", cast=float)
if nidq_fs is None:
    nidq_fs = meta_get(nidq_meta, "imSampRate", cast=float)

imec_fs = meta_get(imec_meta, "imSampRate", cast=float)


print(f"[META] nidq_fs={nidq_fs}, imec_fs={imec_fs}, nidq_nCh={nSavedChannels}, imec_nCh={totalChannels}")


#%% Step 4

# ---------- Load .bin files (imec sync + nidq digital) --------

def load_bin_memmap(bin_path, n_channels, dtype=np.int16):
    file_bytes = os.path.getsize(bin_path)
    bytes_per_sample = np.dtype(dtype).itemsize
    n_samples = file_bytes // (bytes_per_sample * n_channels)

    if file_bytes % (bytes_per_sample * n_channels) != 0:
        raise ValueError("File size not divisible by (dtype * n_channels). Check dtype/nSavedChans.")

    return np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_samples, n_channels))

nidq_bin = load_bin_memmap(path_nidq_bin, nSavedChannels)
imec_ap_bin = load_bin_memmap(path_imec_ap_bin, totalChannels)

print("[DATA] nidq shape:", nidq_bin.shape)
print("[DATA] imec shape:", imec_ap_bin.shape)


#%% Step 5

# Extract IMEC SY + Split NIDQ analog + Digital word


#  ---------- IMEC AP: sync is typically the last saved channel ("SY") ----------
imec_sync = imec_ap_bin[:, -1]  # int16 vector
plt.plot(imec_sync[:1000000])  # Sanity check to quickly plot the sync channel and check is not empty


# ---------- NiDaq: split analog + digital ----------

# By convention for nidq, the last saved channel is a packed digital word(XD0) and the first 3 are analog channels (XA0-XA2)
nidq_dig_word = nidq_bin[:, -1].astype(np.uint16)     # (samples,)
nidq_analog_raw = nidq_bin[:, :-1].astype(np.float32) # (samples, n_analog)


# Scale analog int16 to volts (assuming NI range is [-10, 10] or similar)
ni_max = meta_get(nidq_meta, "niAiRangeMax", cast=float, default=10.0)
ni_min = meta_get(nidq_meta, "niAiRangeMin", cast=float, default=-10.0)


# int16 spans 65536 codes total; SpikeGLX/NI commonly maps full-scale to that span.
nidq_analog_volts = nidq_analog_raw * ((ni_max - ni_min) / 65536.0)


#%% Step 6

# ---------- Channel mapping ----------

# Analog
ai_blue_led = nidq_analog_volts[:, 0]   # XA0
ai_sound    = nidq_analog_volts[:, 1]   # XA1
ai_laser    = nidq_analog_volts[:, 2]   # XA2

# Digital bits
BIT_led      = 4   # DI4
BIT_stim     = 5   # DI5
BIT_punish   = 6   # DI6
BIT_reward   = 7   # DI7


#%% Step 7

# ---------- Digital TTL pairs from packed word ----------

# PART 1 - Extract TTL pulses

def ttl_pairs_from_word(ttl_raw, ttl_ts, n_bits=16, active_low_bits=None, pad_unmatched=False):
    """
    Returns:
      ttls[bit] = array (N,2) with [rise_time, fall_time]
    """
    dw = np.asarray(ttl_raw).astype(np.uint16)
    ts = np.asarray(ttl_ts)
    assert dw.shape[0] == ts.shape[0], "ttl_raw and ttl_ts must have same length"

    bits = ((dw[:, None] >> np.arange(n_bits)) & 1).astype(np.int8)

    if active_low_bits is not None:
        for b in active_low_bits:
            if 0 <= b < n_bits:
                bits[:, b] = 1 - bits[:, b]

    ttls = {}
    edges = {}

    for b in range(n_bits):
        x = bits[:, b]
        d = np.diff(x, prepend=x[0])
        rise_idx = np.flatnonzero(d == 1)
        fall_idx = np.flatnonzero(d == -1)

        if fall_idx.size and (not rise_idx.size or fall_idx[0] < rise_idx[0]):
            fall_idx = fall_idx[1:]

        if pad_unmatched:
            R, F = rise_idx.size, fall_idx.size
            if R > F:
                fall_idx = np.concatenate([fall_idx, np.array([-1])])
            elif F > R:
                rise_idx = np.concatenate([rise_idx, np.array([-1])])

            rise_t = np.where(rise_idx >= 0, ts[rise_idx], np.nan)
            fall_t = np.where(fall_idx >= 0, ts[fall_idx], np.nan)
            pairs = np.column_stack([rise_t, fall_t])
        else:
            m = min(rise_idx.size, fall_idx.size)
            pairs = np.column_stack([ts[rise_idx[:m]], ts[fall_idx[:m]]]) if m > 0 else np.empty((0, 2))

        ttls[b] = pairs
        edges[b] = {
            "rise_idx": rise_idx,
            "fall_idx": fall_idx,
            "rise_t": ts[rise_idx] if rise_idx.size else np.empty(0),
            "fall_t": ts[fall_idx] if fall_idx.size else np.empty(0),
        }

    return ttls, edges

# timestamps in seconds for nidq samples
nidq_ts = np.arange(nidq_bin.shape[0]) / nidq_fs

ttls_daq, edges_daq = ttl_pairs_from_word(nidq_dig_word, nidq_ts, n_bits=16)

ttl_led_d   = ttls_daq[BIT_led]    # (N,2) [rise, fall]
ttl_stim_d  = ttls_daq[BIT_stim]
ttl_pun_d   = ttls_daq[BIT_punish]
ttl_rew_d   = ttls_daq[BIT_reward]

print("[TTL digital] DI4 LED:", ttl_led_d.shape[0],
      "DI5 STIM:", ttl_stim_d.shape[0],
      "DI6 PUN:", ttl_pun_d.shape[0],
      "DI7 REW:", ttl_rew_d.shape[0])

#%% Step 8

# ---------- Threshold windows for analog + imec SY ----------

def find_threshold_windows_np(a, threshold):
    a = np.asarray(a)
    above = a > threshold

    # transitions of boolean
    x = above.view(np.int8)
    changes = np.diff(x)

    starts = np.flatnonzero(changes == 1) + 1
    ends   = np.flatnonzero(changes == -1)

    if above.size and above[0]:
        starts = np.r_[0, starts]
    if above.size and above[-1]:
        ends = np.r_[ends, above.size - 1]

    if starts.size == 0 or ends.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    n = min(starts.size, ends.size)
    starts = starts[:n]
    ends = ends[:n]
    keep = ends >= starts
    starts = starts[keep]
    ends = ends[keep]

    out = np.empty((starts.size, 3), dtype=np.int32)
    out[:, 0] = (ends - starts + 1)
    out[:, 1] = starts
    out[:, 2] = ends
    return out

# thresholds (adjust if needed)
thr_led_v   = 2.0
thr_sound_v = 0.5
thr_laser_v = 0.5

# analog windows: [len, start_idx, end_idx] in NIDQ sample index space
win_led_a   = find_threshold_windows_np(ai_blue_led, thr_led_v)
win_sound_a = find_threshold_windows_np(ai_sound, thr_sound_v)
win_laser_a = find_threshold_windows_np(ai_laser, thr_laser_v)

# IMEC SY windows: threshold depends on your SY amplitude
thr_sy = 10
win_sy = find_threshold_windows_np(imec_sync, thr_sy)

print("[TTL analog windows] LED:", win_led_a.shape[0], "SOUND:", win_sound_a.shape[0], "LASER:", win_laser_a.shape[0])
print("[IMEC SY windows]", win_sy.shape[0])



#%% Step 9

# ---------- Align Nidq and Imec using sync rising edges

def bin_edges_to_ms(edge_times_s, t_max_s, bin_ms=1.0):
    dt = bin_ms / 1000.0
    n = int(np.ceil(t_max_s / dt)) + 1
    y = np.zeros(n, dtype=np.float32)
    idx = (edge_times_s / dt).astype(int)
    idx = idx[(idx >= 0) & (idx < n)]
    y[idx] = 1.0
    return y, dt

def best_nidq_sync_line_from_pairs(imec_rise_s, ttls_by_bit, search_bits=range(16), t_search_s=60):
    imec_rise_s = imec_rise_s[imec_rise_s <= t_search_s]
    if imec_rise_s.size < 5:
        raise ValueError("Too few IMEC SY pulses in search window.")

    t_max = float(min(t_search_s, imec_rise_s[-1]))
    imec_binned, dt = bin_edges_to_ms(imec_rise_s, t_max, bin_ms=1.0)

    best = {"bit": None, "corr": -np.inf, "lag_s": None}

    for b in search_bits:
        pairs = ttls_by_bit[b]
        nidq_rise_s = pairs[:, 0]
        nidq_rise_s = nidq_rise_s[nidq_rise_s <= t_max]
        if nidq_rise_s.size < 5:
            continue

        nidq_binned, _ = bin_edges_to_ms(nidq_rise_s, t_max, bin_ms=1.0)
        c = np.correlate(imec_binned, nidq_binned, mode="full")
        k = int(np.argmax(c))
        lag_bins = k - (len(nidq_binned) - 1)
        lag_s = lag_bins * dt

        if c[k] > best["corr"]:
            best = {"bit": int(b), "corr": float(c[k]), "lag_s": float(lag_s)}

    if best["bit"] is None:
        raise ValueError("Could not find a matching NIDQ sync bit.")
    return best


def fit_time_mapping(nidq_rise_s, imec_rise_s, coarse_lag_s=0.0, n_fit=5000):
    nidq_shift = nidq_rise_s + coarse_lag_s
    imec = imec_rise_s

    j = np.searchsorted(imec, nidq_shift)
    j = np.clip(j, 1, len(imec) - 1)

    left = imec[j - 1]
    right = imec[j]
    choose_right = (np.abs(right - nidq_shift) < np.abs(nidq_shift - left))
    matched_imec = np.where(choose_right, right, left)

    n = min(len(nidq_rise_s), len(matched_imec), n_fit)
    x = nidq_rise_s[:n]
    y = matched_imec[:n]

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


# Find sync bit automatically (like your earlier code, but using pairs dict)
best_sync = best_nidq_sync_line_from_pairs(imec_rise, ttls_daq, search_bits=range(16), t_search_s=60)
sync_bit = best_sync["bit"]
coarse_lag_s = best_sync["lag_s"]
print(f"[ALIGN] Best NIDQ sync bit: DI{sync_bit}, coarse lag {coarse_lag_s:.6f} s")

nidq_sync_rise = ttls_daq[sync_bit][:, 0]
a, b = fit_time_mapping(nidq_sync_rise, imec_rise, coarse_lag_s=coarse_lag_s, n_fit=5000)
print(f"[ALIGN] t_imec ≈ {a:.12f} * t_nidq + {b:.6f}")

def nidq_to_imec(t_s):
    return a * np.asarray(t_s) + b

#%% Step 10

# ---------- Align all nidq TTL pairs into Imec timebase ----------

def map_pairs_to_imec(pairs):
    if pairs.size == 0:
        return pairs.copy()
    out = pairs.copy().astype(np.float64)
    out[:, 0] = nidq_to_imec(out[:, 0])
    out[:, 1] = nidq_to_imec(out[:, 1])
    return out

ttl_led_d_imec   = map_pairs_to_imec(ttl_led_d)
ttl_stim_d_imec  = map_pairs_to_imec(ttl_stim_d)
ttl_pun_d_imec   = map_pairs_to_imec(ttl_pun_d)
ttl_rew_d_imec   = map_pairs_to_imec(ttl_rew_d)

ttl_led_a_imec   = map_pairs_to_imec(ttl_led_a)
ttl_sound_a_imec = map_pairs_to_imec(ttl_sound_a)
ttl_laser_a_imec = map_pairs_to_imec(ttl_laser_a)

print("[DONE] All NIDQ TTLs now have [rise, fall] in IMEC seconds.")


#%% Step 11

# ---------- Sanity check plots ----------


# ---- Plot settings ----
NIDQ_PLOT_SEC = 90
IMEC_PLOT_SEC = 90

nidq_n = int(NIDQ_PLOT_SEC * nidq_fs)
imec_n = int(IMEC_PLOT_SEC * imec_fs)

# IMEC SY
plt.figure(figsize=(12, 3))
plt.plot(np.arange(imec_n) / imec_fs, imec_sync[:imec_n])
plt.title("IMEC SY (first few seconds)")
plt.xlabel("Time (s)")
plt.ylabel("SY (int16)")
plt.tight_layout()
plt.show()

# Digital bits (DI4-7) stacked from decoded word (fast preview)
bits_preview = ((nidq_dig_word[:nidq_n, None] >> np.arange(16)) & 1).astype(np.int8)
t = np.arange(nidq_n) / nidq_fs

plt.figure(figsize=(12, 4))
plt.plot(t, bits_preview[:, BIT_LED] + 0, label="DI4 LED")
plt.plot(t, bits_preview[:, BIT_STIM] + 2, label="DI5 STIM")
plt.plot(t, bits_preview[:, BIT_PUN] + 4, label="DI6 PUN")
plt.plot(t, bits_preview[:, BIT_REW] + 6, label="DI7 REW")
plt.title("NIDQ digital lines (first 20 s, stacked)")
plt.yticks([])
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()

# Analog monitors + thresholds
plt.figure(figsize=(12, 3))
plt.plot(t, ai_blue_led[:nidq_n], label="XA0 LED")
plt.plot(t, np.full_like(t, thr_led_v), "--", label=f"thr={thr_led_v} V")
plt.title("XA0 LED monitor + threshold")
plt.xlabel("Time (s)")
plt.ylabel("V")
plt.legend()
plt.tight_layout()
plt.show()