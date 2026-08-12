# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:46:04 2026

@author: JoanaCatarino
"""

import numpy as np
import json
from pynwb import NWBFile, NWBHDF5IO
from datetime import datetime


# data to check

animal = "999770"
day = "day1"
probe = "imec1"

# -----------------------------
# Paths
# -----------------------------
spikes_path = r'L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/999770/999770_day1_g0_imec1/analyzer_kilosort4/sorting/spikes.npy'
curation_path = r'L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/999770/999770_day1_g0_imec1/analyzer_kilosort4/curation.json'
output_nwb = rf'L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/Sorted/999770/999770_day1_g0_imec1/{animal}_{day}_{probe}_filtered_spikes.nwb'


# -----------------------------
# Load curation
# -----------------------------
with open(curation_path, "r") as f:
    curation = json.load(f)

good_units = set()
quality_map = {}

for entry in curation["manual_labels"]:
    unit = entry["unit_id"]
    labels = entry.get("quality", [])
    if len(labels) > 0:
        quality_map[unit] = labels[0]
        if labels[0] in ["good", "MUA"]:
            good_units.add(unit)

print("Units kept:", len(good_units))

# -----------------------------
# Load spikes
# -----------------------------
spikes = np.load(spikes_path)

print("shape:", spikes.shape)
print("dtype:", spikes.dtype)
print("fields:", spikes.dtype.names)

# SpikeInterface spike vector fields
spike_samples = spikes["sample_index"]
unit_ids = spikes["unit_index"]

# Optional: convert samples to seconds
sampling_rate = 30000.0
spike_times_sec = spike_samples.astype(float) / sampling_rate

# -----------------------------
# Keep only curated units
# -----------------------------
mask = np.isin(unit_ids, list(good_units))

spike_times_sec = spike_times_sec[mask]
unit_ids = unit_ids[mask]

# -----------------------------
# Create NWB file
# -----------------------------
nwbfile = NWBFile(
    session_description="Filtered spike data containing only units labeled good or MUA",
    identifier="filtered_spikes",
    session_start_time=datetime.now(),
)

nwbfile.add_unit_column(
    name="quality",
    description="Manual curation quality label"
)

# -----------------------------
# Add units
# -----------------------------
for unit in np.unique(unit_ids):
    times = spike_times_sec[unit_ids == unit]
    quality = quality_map.get(int(unit), "unknown")

    nwbfile.add_unit(
        id=int(unit),
        spike_times=times,
        quality=quality
    )

# -----------------------------
# Write file
# -----------------------------
with NWBHDF5IO(output_nwb, "w") as io:
    io.write(nwbfile)

print("Saved:", output_nwb)