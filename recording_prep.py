# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:20:22 2025

@author: JoanaCatarino
"""

import spikeinterface.full as si
from spikeinterface_gui import run_mainwindow
from probeinterface import Probe
import numpy as np
import os
import importlib.util

# ---------- Paths -----------

# Raw_data used for Kilosort 4
raw_data_file = "C:/Users/JoanaCatarino/Joana/recording_tests/999770/Raw_data/999770_day2_g0_t0.imec0.ap.bin"

# folder with Kilosort results 
kilosort_folder = "C:/Users/JoanaCatarino/Joana/recording_tests/999770/kilosorting"  

# Folder where SpikeInterface metrics will be saved
analyzer_folder = "C:/Users/JoanaCatarino/Joana/recording_tests/999770/Spike_Interface/probe1"


# ---------- Load recording ----------

recording = si.read_binary(
    file_paths=raw_data_file,
    sampling_frequency=30000.0,
    num_channels=385,
    dtype="int16",
)


# ---------- Attach probe geometry ----------
map_path = os.path.join(kilosort_folder, "channel_map.npy")
pos_path = os.path.join(kilosort_folder, "channel_positions.npy")
shank_path = os.path.join(kilosort_folder, "channel_shanks.npy")

chan_map = np.load(map_path)
positions = np.load(pos_path)
shanks = np.load(shank_path)

# Fix mismatched channel indexing
chan_map = chan_map[chan_map < positions.shape[0]]

# Reorder positions and shanks to match binary ordering
positions_ordered = positions[chan_map]
shanks_ordered = shanks[chan_map]

# Create probe
probe = Probe(ndim=2)
probe.set_contacts(positions=positions_ordered, shank_ids=shanks_ordered)

# **THIS IS THE MISSING STEP**
probe.set_device_channel_indices(chan_map)

# Attach probe
recording = recording.set_probe(probe)

print("✔ Probe attached successfully with device channel indices")

# ---------- Load Kilosort4 sorting ----------
# If you want *all* units (not only "good") set keep_good_only=False.
sorting = si.read_kilosort(folder_path=kilosort_folder)

print("Units from Kilosort4:", sorting.get_unit_ids())


# ---------- Create SortingAnalyzer ----------

job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="3s")

sorting_analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="binary_folder",
    folder=analyzer_folder,
    **job_kwargs,
)


# ---------- Compute extensions and metrics for spikeinterface gui
sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
sorting_analyzer.compute("waveforms", **job_kwargs)
sorting_analyzer.compute("templates", **job_kwargs)
#sorting_analyzer.compute("noise_levels")
#sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
#sorting_analyzer.compute("isi_histograms")
#sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.)
#sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_global', whiten=True, **job_kwargs)
#sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
#sorting_analyzer.compute("template_similarity")
#sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

print("✔ Metrics for SpikeInterface computed")


# ---------- Open GUI automatically ----------
print("Launching SpikeInterface GUI…")
run_mainwindow(sorting_analyzer, mode="desktop", curation=True)
