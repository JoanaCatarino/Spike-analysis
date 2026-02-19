#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:47:41 2025

@author: anil
"""

import numpy as np
from numba import njit

from scipy import interpolate



def find_windows_np(a_bool):
    """
    Given a boolean 1D array, returns windows where the value is True (1).
    Output: N x 3 array: [length, start_index, end_index]
    """
    # Fast int conversion using view avoids copy
    a_int = a_bool.view(np.int8)  # much faster than astype
    changes = np.diff(a_int)

    # Transitions
    starts = np.flatnonzero(changes == 1) + 1
    ends = np.flatnonzero(changes == -1)

    # Edge cases
    if a_bool[0]:
        starts = np.insert(starts, 0, 0)
    if a_bool[-1]:
        ends = np.append(ends, len(a_bool) - 1)

    # Use int32 directly to avoid float->int conversion
    result = np.empty((len(starts), 3), dtype=np.int32)
    result[:, 0] = ends - starts + 1  # lengths
    result[:, 1] = starts
    result[:, 2] = ends

    return result



@njit
def find_windows(a_bool):
    """
    Finds contiguous windows where a_bool is True.
    Returns an (N, 3) array: [length, start_idx, end_idx]
    """
    n = len(a_bool)
    windows = []
    inside = False
    start = 0

    for i in range(n):
        if a_bool[i]:
            if not inside:
                start = i
                inside = True
        else:
            if inside:
                end = i - 1
                windows.append((end - start + 1, start, end))
                inside = False

    # Handle case where signal ends while still inside a True segment
    if inside:
        end = n - 1
        windows.append((end - start + 1, start, end))

    result = np.empty((len(windows), 3), dtype=np.int32)
    for i, (length, start, end) in enumerate(windows):
        result[i, 0] = length
        result[i, 1] = start
        result[i, 2] = end

    return result


@njit
def find_threshold_windows(a, threshold):
    n = len(a)
    windows = []
    inside = False
    start = 0

    for i in range(n):
        if a[i] > threshold:
            if not inside:
                start = i
                inside = True
        else:
            if inside:
                end = i - 1
                windows.append((end - start + 1, start, end))
                inside = False

    if inside:
        end = n - 1
        windows.append((end - start + 1, start, end))

    result = np.empty((len(windows), 3), dtype=np.int32)
    for i, (length, start, end) in enumerate(windows):
        result[i, 0] = length
        result[i, 1] = start
        result[i, 2] = end

    return result




def interpolate_ttl(src_times, target_times, query_times):
    """Interpolate timestamps from src to target timebase."""
    f_interp = interpolate.interp1d(src_times, target_times, fill_value='extrapolate')
    return f_interp(query_times)




import numpy as np
import os


def load_bin(bin_path, n_channels, dtype=np.int16):
    """
    Memory-map a multichannel Neuropixels .bin file.
    
    Returns:
        memmap array of shape (n_samples, n_channels)
        n_samples
    """
    file_size_bytes = os.path.getsize(bin_path)
    bytes_per_sample = np.dtype(dtype).itemsize
    n_samples = file_size_bytes // (bytes_per_sample * n_channels)
    
    data = np.memmap(bin_path, dtype=dtype, mode='r', shape=(n_samples, n_channels))
    return data, n_samples


def load_single_channel(bin_path, channel_idx, n_channels, dtype=np.int16,
                        sampling_rate=None, start_sec=None, end_sec=None):
    """
    Load a single channel from a Neuropixels .bin file, optionally specifying a time window.

    Args:
        bin_path (str): Path to .bin file.
        channel_idx (int): Index of channel to load.
        n_channels (int): Total number of saved channels.
        dtype (np.dtype): Data type of binary file (default: np.int16).
        sampling_rate (float): Sampling rate in Hz (optional).
        start_sec (float): Start time in seconds (optional).
        end_sec (float): End time in seconds (optional).

    Returns:
        1D numpy array of the selected channel's data in the specified time range.
    """
    data, n_samples = load_bin(bin_path, n_channels, dtype=dtype)

    # Compute start and end indices
    if sampling_rate is not None and start_sec is not None:
        start_idx = int(start_sec * sampling_rate)
    else:
        start_idx = 0

    if sampling_rate is not None and end_sec is not None:
        end_idx = int(end_sec * sampling_rate)
    else:
        end_idx = n_samples

    return data[start_idx:end_idx, channel_idx]






import numpy as np
import os

def load_channel_chunk(
    bin_path,
    channel_idx,
    n_channels,
    sample_rate,
    start_sec=0,
    stop_sec=None,
    dtype=np.int16
):
    """
    Load a chunk of a single channel from a binary Neuropixels .bin file.

    Parameters:
    - bin_path: path to the .bin file
    - channel_idx: index of the channel to load (0-based)
    - n_channels: total number of interleaved channels
    - sample_rate: sampling rate in Hz
    - start_sec: start time in seconds (default = 0)
    - stop_sec: stop time in seconds (default = full length)
    - dtype: data type (default = int16)

    Returns:
    - time: time vector in seconds
    - signal: signal values for the specified chunk
    """
    bytes_per_sample = np.dtype(dtype).itemsize
    file_size_bytes = os.path.getsize(bin_path)
    total_samples = file_size_bytes // (bytes_per_sample * n_channels)

    start_idx = int(start_sec * sample_rate)
    if stop_sec is None:
        stop_idx = total_samples
    else:
        stop_idx = int(stop_sec * sample_rate)

    # Sanity check
    if stop_idx > total_samples:
        stop_idx = total_samples
    if start_idx >= stop_idx:
        raise ValueError("start_sec must be less than stop_sec and within recording bounds.")

    num_samples = stop_idx - start_idx

    # Map only the range we need
    data = np.memmap(
        bin_path,
        dtype=dtype,
        mode='r',
        shape=(total_samples, n_channels)
    )

    signal = data[start_idx:stop_idx, channel_idx]
    time = np.arange(start_idx, stop_idx) / sample_rate

    return time, signal














































