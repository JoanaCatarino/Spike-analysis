"""
config_psth.py

Edit this file to switch experiments. The plotting script reads all settings from here.
"""

# Path to a single .nwb file OR a directory containing multiple .nwb files.
data_path = r"L:/dmclab/Joana/PFC-Str_behavior_project/Recordings/NWB/999770_20251111_2probes_2.nwb"

# Event used for peri-event alignment
# Examples:
# event_path = "intervals/trials/lick_time"
# event_path = "intervals/trials/imec_stim_on"
# event_path = "intervals/trials/imec_reward_on"
# event_path = "intervals/trials/imec_punishment_on"
event_path = "intervals/trials/imec_blue_led_on"

# Peri-event window
win_start = -1.0
win_stop = 3.0
psth_bin = 0.005

# Z-score baseline: bins with t < baseline_stop are used as baseline
baseline_stop = 0.0

# Smoothing and display
smooth_sigma = 3.0       # Gaussian sigma in bins
zlim = 3.0               # color saturation for z-scored heatmap

# Unit filters
min_firing_rate = 0.04   # Hz, set to 0 to disable

# Brain regions to include. Empty list includes all regions.
# regions = ["MOs", "FRP", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv", "CP", "ACB"]
regions = []

# Optional custom colors, not used by this heatmap but kept for compatibility.
region_colors = []

# Figure saving. Empty string means display only.
save_path = r"L:/dmclab/Joana/PFC-Str_behavior_project/Nfn/neuro_plots/script_pierre"
save_format = "svg"
