# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 14:28:32 2025

@author: JoanaCatarino
"""

import spikeinterface.extractors as se
from spikeinterface.core import create_sorting_analyzer, load_sorting_analyzer

# path to the Kilosort4 output folder
ks_output_folder = "C:/Users/JoanaCatarino/Joana/recording_tests/999770/kilosorting"
sorting = se.read_kilosort(ks_output_folder)

recording = se.read_spikeglx("C:/Users/JoanaCatarino/Joana/recording_tests/999770/Raw_data/999770_day2_g0_t0.imec0",
                            stream_id='imec0.ap')

job_kwargs = dict(n_jobs=4, progress_bar=True, chunk_duration="1s")

sorting_analyzer = create_sorting_analyzer(sorting, recording, 
                                        folder="sorting_analyzer",
                                        format="binary_folder",
                                        **job_kwargs)

sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
sorting_analyzer.compute("waveforms", **job_kwargs)
sorting_analyzer.compute("templates", **job_kwargs)
sorting_analyzer.compute("noise_levels")
sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
sorting_analyzer.compute("isi_histograms")
sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.)
sorting_analyzer.compute("principal_components", n_components=3, mode='by_channel_global', whiten=True, **job_kwargs)
sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
sorting_analyzer.compute("template_similarity")
sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

from spikeinterface_gui import run_mainwindow

run_mainwindow(sorting_analyzer, mode="desktop")