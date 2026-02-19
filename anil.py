# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:05:08 2026

@author: JoanaCatarino
"""

def parse_meta(meta_path):
    """Parse a SpikeGLX .meta file into a dictionary."""
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            # print(line)
            if '=' in line:
                key, val = line.strip().split('=', maxsplit=1)
                meta[key] = val
    return meta



def ttl_pairs_from_word(ttl_raw, ttl_ts, n_bits=16, active_low_bits=None, pad_unmatched=False):
    """
    Parameters
    ----------
    ttl_raw : (T,) int array
        Raw digital word per sample (e.g., from nidq), will be cast to uint16.
    ttl_ts : (T,) float array
        Timestamps (seconds) for each sample.
    n_bits : int
        How many LSBs to decode.
    active_low_bits : iterable or None
        Bits that are active-low (invert those after decoding).
    pad_unmatched : bool
        If True, pad unmatched rise/fall with NaN instead of truncating.

    Returns
    -------
    ttls : dict[int, (N,2) float array]
        For each bit index, an array of [rise_time, fall_time] rows.
    edges : dict[int, dict[str, np.ndarray]]
        Optional: raw edge indices/time arrays for debugging.
    """
    dw = np.asarray(ttl_raw).astype(np.uint16)
    ts = np.asarray(ttl_ts)
    assert dw.shape[0] == ts.shape[0], "ttl_raw and ttl_ts must have the same length"

    # decode bits: bits[T, n_bits] in {0,1}
    bits = ((dw[:, None] >> np.arange(n_bits)) & 1).astype(np.int8)

    # invert any active-low lines
    if active_low_bits is not None:
        for b in active_low_bits:
            if 0 <= b < n_bits:
                bits[:, b] = 1 - bits[:, b]

    ttls = {}
    edges = {}

    for b in range(n_bits):
        x = bits[:, b]

        # transitions: +1 = rising, -1 = falling
        d = np.diff(x, prepend=x[0])
        rise_idx = np.flatnonzero(d == 1)
        fall_idx = np.flatnonzero(d == -1)

        # Pair rises/falls
        # If we start high, first transition could be a fall before any rise → drop it.
        if fall_idx.size and (not rise_idx.size or fall_idx[0] < rise_idx[0]):
            fall_idx = fall_idx[1:]

        # If we end high, there may be an extra rise with no fall.
        if pad_unmatched:
            # pad with NaN to keep counts equal
            R, F = rise_idx.size, fall_idx.size
            if R > F:
                fall_idx = np.concatenate([fall_idx, np.array([-1])])  # will map to NaN below
            elif F > R:
                rise_idx = np.concatenate([rise_idx, np.array([-1])])

            rise_t = np.where(rise_idx >= 0, ts[rise_idx], np.nan)
            fall_t = np.where(fall_idx >= 0, ts[fall_idx], np.nan)
            pairs = np.column_stack([rise_t, fall_t])
        else:
            m = min(rise_idx.size, fall_idx.size)
            pairs = np.column_stack([ts[rise_idx[:m]], ts[fall_idx[:m]]]) if m > 0 else np.empty((0,2))

        ttls[b] = pairs
        edges[b] = {
            "rise_idx": rise_idx,
            "fall_idx": fall_idx,
            "rise_t": ts[rise_idx] if rise_idx.size else np.empty(0),
            "fall_t": ts[fall_idx] if fall_idx.size else np.empty(0),
        }

    return ttls, edges



meta_daq = parse_meta(daq_meta)

ttls_daq, edges = ttl_pairs_from_word(ttl_daq_raw, np.arange(len(ttl_daq_raw)), n_bits=16)

ttl_windows_imec = utilso.find_threshold_windows(imec_sync, 10)