# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 13:26:53 2026

@author: JoanaCatarino
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import json
import shutil

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
import spikeinterface.curation as sc
from spikeinterface.sortingcomponents.motion import interpolate_motion

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#%matplotlib widget
%matplotlib qt5

# %% parameters
#global_job_kwargs = dict(n_jobs=20, mp_context='fork', progress_bar=True)
global_job_kwargs = dict(n_jobs=20, mp_context='spawn', progress_bar=True) # windows version
si.set_global_job_kwargs(**global_job_kwargs)

# %% change these if needed
base_folder = Path('C:/Users/JoanaCatarino/Joana/recordings_local/')
spikeglx_folder = base_folder / '999770'
#working_folder = Path('/home/dmc/Documents/test_SI/Test_Joana/')
working_folder = Path('C:/Users/JoanaCatarino/Joana/recordings_local/999770/tests_si/') #Joana edit

# %% import
stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
raw_rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)

# %% check the probe layout
raw_rec.get_probe().to_dataframe()
si.set_default_plotter_backend("matplotlib")
fig, ax = plt.subplots(figsize=(10, 5))
si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=False)
ax.set_ylim(-150, 500)

# %% set the "ipywidgets" backend as default
#si.set_default_plotter_backend("ipywidgets")
#w = si.plot_traces(raw_rec, order_channel_by_depth=True, backend="ipywidgets")

# Spyder version
si.set_default_plotter_backend("matplotlib")

w = si.plot_traces(
    raw_rec,
    order_channel_by_depth=True,
    backend="matplotlib"
)



################## PREPROCESS #############################
# %% preprocess
raw_rec = raw_rec.astype('float32') 
shifted_recordings = spre.phase_shift(raw_rec)
referenced_recording = spre.common_reference(shifted_recordings, reference="global", operator="median")
filtered_recording = spre.bandpass_filter(referenced_recording, freq_min=300.0, freq_max=6000.0)
good_channels_recording = spre.detect_and_remove_bad_channels(filtered_recording)

    

# %% plot the preprocessed traces
#si.set_default_plotter_backend("ipywidgets")
#w = si.plot_traces(good_channels_recording, order_channel_by_depth=True, backend="ipywidgets")


#Spyder version
si.set_default_plotter_backend("matplotlib")
w = si.plot_traces(good_channels_recording, order_channel_by_depth=True, backend="matplotlib")


# %% Motion estimation, we test all presets here
# get all presets
presets = ("rigid_fast", "kilosort_like",  "nonrigid_accurate", "nonrigid_fast_and_accurate", "dredge_fast")
#si.get_motion_presets()
# compute motion with theses presets
for preset in presets:
    print("Computing with", preset)
    motion_folder = working_folder / f"motion_{preset}"
    if motion_folder.exists():
        print("Motion folder already exists for", preset)
        #shutil.rmtree(motion_folder)
    else:    
        motion, motion_info = si.compute_motion(
            good_channels_recording, preset=preset, folder=motion_folder, output_motion_info=True, overwrite=True, **global_job_kwargs
        )

# %% Plot all to decide
si.set_default_plotter_backend("matplotlib")
for preset in presets:
    # load
    motion_folder = working_folder / f"motion_{preset}"
    motion_info = si.load_motion_info(motion_folder)

    # and plot
    fig = plt.figure(figsize=(14, 8))
    si.plot_motion_info(
        motion_info, good_channels_recording,
        figure=fig,
        depth_lim=(400, 600),
        color_amplitude=True,
        amplitude_cmap="inferno",
        scatter_decimate=10,
    )

    fig.suptitle(f"{preset=}")

        
# %% apply motion correction
preset = "dredge_fast"
motion_info = si.load_motion_info(motion_folder)
motion = motion_info['motion']
interpolated_recording = interpolate_motion(recording=good_channels_recording, motion=motion)
interpolated_recording

# %% save preprocessed recording
recording_preprocessed = interpolated_recording
recording_preprocessed = recording_preprocessed.astype('int16') 

# compress with zarr, saves disk space
preprocessed_folder = working_folder / "preprocessed"
if preprocessed_folder.is_dir():
    print("Preprocessed recording is already saved")
else:
    recording = recording_preprocessed.save(folder=preprocessed_folder, format="binary")





################## SORTING #############################

# %% reload preprocessed recording
preprocess_folder = working_folder / "preprocessed"
if not preprocess_folder.is_dir():
    # in case the preprocessing is not done
    print("Preprocessed not done...")
else:
    print("Loading preprocessed recording...")    
    rec = si.load(preprocess_folder)
rec

# %% check available sorters
si.installed_sorters()

# %%
# Cut the 100s for sorting speed
sampling_rate = rec.get_sampling_frequency()
num_frames = int(100 * sampling_rate)  # 100 secondes en frames

recording_100s = rec.frame_slice(start_frame=0, end_frame=num_frames)

# %% Test Lupin
sorter_params = si.get_default_sorter_params('lupin')
sorting_lupin = si.run_sorter(
    'lupin', recording_100s,
    folder = working_folder / 'sorter_lupin_100s',
    verbose=True,
    remove_existing_folder=True,
    **sorter_params
)
# %% Test KS4
sorter_params = si.get_default_sorter_params('kilosort4')
sorting_KS4 = si.run_sorter(
    'kilosort4', recording_100s,
    folder = working_folder / 'sorter_KS4_100s',
    verbose=True,
    remove_existing_folder=True,
    **sorter_params
)

# %% Test Tridesclous2
sorter_params = si.get_default_sorter_params('tridesclous2')
sorting_TDC = si.run_sorter(
    'tridesclous2', recording_100s,
    folder = working_folder / 'sorter_TDC2_100s',
    verbose=True,
    remove_existing_folder=True,
    **sorter_params
)
# %% Test Spyking Circus
sorter_params = si.get_default_sorter_params('spykingcircus2')
sorting_SC2 = si.run_sorter(
    'spykingcircus2', recording_100s,
    folder = working_folder / 'sorter_SC2_100s',
    verbose=True,
    remove_existing_folder=True,
    **sorter_params
)   

# %% Reload sortings from folder (if needed)
sorting_lupin = si.load(working_folder / "sorter_lupin_100s")
sorting_KS4 = si.load(working_folder / "sorter_KS4_100s")
sorting_TDC = si.load(working_folder / "sorter_TDC2_100s")
sorting_SC2 = si.load(working_folder / "sorter_SC2_100s")

# %% visualize sorting results
#si.set_default_plotter_backend("ipywidgets")
si.set_default_plotter_backend("matplotlib")
si.plot_rasters(sorting_lupin, unit_ids=sorting_lupin.unit_ids[:60], time_range=(0, 5))
#si.plot_rasters(sorting_KS4, unit_ids=sorting_KS4.unit_ids[:60], time_range=(0, 5))
#si.plot_rasters(sorting_TDC, unit_ids=sorting_TDC.unit_ids[:60], time_range=(0, 5))
#si.plot_rasters(sorting_SC2, unit_ids=sorting_SC2.unit_ids[:60], time_range=(0, 5))


# %% Sorter comparison
multi_comp = si.compare_multiple_sorters(
    sorting_list=[sorting_KS4, sorting_TDC, sorting_SC2, sorting_lupin],
    name_list=['KS4', 'TDC2', 'SC2', 'LP'],
    spiketrain_mode='union',
    verbose=True
)
# %matplotlib inline
%matplotlib widget
w = si.plot_multicomparison_agreement(multi_comp)
w = si.plot_multicomparison_agreement_by_sorter(multi_comp)


################## ANALYSIS #############################
# %% make the analyzer
recording = si.load(working_folder / "preprocessed")
sorting = si.load(working_folder / f"sorter_{'KS4_100s'}")
analyzer_KS4 = si.create_sorting_analyzer(
    sorting=sorting, 
    recording=recording, 
    folder=working_folder / f"analyzer_{'KS4_100s'}",
    format="binary_folder",
    overwrite=True
)
# %% compute stuff
# just one
analyzer_KS4.compute("random_spikes", max_spikes_per_unit=1000)

# or lots: here we also specify some kwargs
analyzer_KS4.compute({
    "templates": {},
    "waveforms": {},
    "noise_levels": {},
    "correlograms": {},
    "noise_levels": {'method': 'std'},
    "spike_amplitudes": {},
    "spike_locations": {},
    "template_metrics": {'include_multi_channel_metrics': True},
    "unit_locations": {},
    "template_similarity": {'method': 'l1'},
    "quality_metrics": {},
    "principal_components": {},
})
# %% look at the quality metrics
quality_metrics = analyzer_KS4.get_extension("quality_metrics").get_data()
quality_metrics
# %% plot some stuff
%matplotlib widget
si.plot_amplitudes(analyzer_KS4, backend="ipywidgets")
si.plot_unit_waveforms(analyzer_KS4, backend='ipywidgets')
si.plot_all_amplitudes_distributions(analyzer_KS4)
si.plot_quality_metrics(analyzer_KS4, backend="ipywidgets")
# %% Export to pynapple
spikes = si.to_pynapple_tsgroup(analyzer_KS4)
spikes



################## CURATION #############################

# %%
%load_ext autoreload
%autoreload 2


# load SortingAnalyzer from previous step
#analyzer_folder = results_folder / f"analyzer_{sorter}"
analyzer_folder = working_folder / f"analyzer_{'lupin_100s'}"

if analyzer_folder.is_dir():
    print("Loading analyzer from disk")
    analyzer = si.load(analyzer_folder, load_extensions=True)
else:
    recording_preprocessed = si.load(f"preprocessed.zarr")
    sorting = si.load(f"sorting_{'lupin_100s'}")
    analyzer = si.create_sorting_analyzer(sorting, recording_preprocessed)
    extension_dict = {
        "random_spikes": {},
        "noise_levels": {},
        "waveforms": {},
        "templates": {},
        "template_similarity": {"method": "l1"},
        "template_metrics": {"include_multi_channel_metrics": True},
        "correlograms": {},
        "unit_locations": {},
        "spike_amplitudes": {},
        "spike_locations": {"method": "grid_convolution"},
        "quality_metrics": {},
        "principal_components": {},
    }
    analyzer.compute(extension_dict)
    analyzer.save_as(folder=analyzer_folder, format="binary_folder")
analyzer

# %% Remove redundant units 
sorting_clean = sc.remove_redundant_units(analyzer)
if len(sorting_clean.unit_ids) < len(analyzer.unit_ids):
    num_redundant = len(analyzer.unit_ids) - len(sorting_clean.unit_ids)
    print(f"Found {num_redundant} redundant units")
    analyzer = analyzer.select_units(sorting_clean.unit_ids)
    
num_uncurated_units = len(analyzer.unit_ids)
%matplotlib widget    
_ = sw.plot_unit_templates(analyzer, backend="ipywidgets")

# %% Quality metrics based curation
qm = analyzer.get_extension("quality_metrics").get_data()
print(f"Available metrics: {qm.columns.values}")

curation_query = "rp_contamination < 0.2 and snr > 5"
qm_filtered = qm.query(curation_query)
print(f"Number of units after curation: {len(qm_filtered)} / {len(qm)}")

# filter the units
units_to_keep = list(qm_filtered.index)
analyzer_qm_filt = analyzer.select_units(units_to_keep)
# we can also save the unit outputs 
passing_qc = np.zeros(len(analyzer.unit_ids), dtype=bool)
passing_qc[analyzer.sorting.ids_to_indices(units_to_keep)] = True
analyzer.set_sorting_property("passing_qc", passing_qc)
# and plot
%matplotlib widget
_ = sw.plot_unit_templates(analyzer_qm_filt, backend="ipywidgets")


# %% Automatic curation with UnitRefine

 # Apply the noise/neuron model
 noise_neuron_labels = sc.auto_label_units(
    sorting_analyzer=analyzer,
    repo_id="SpikeInterface/UnitRefine_noise_neural_classifier_lightweight",
    trust_model=True,
)

noise_units = noise_neuron_labels[noise_neuron_labels['prediction'] == 'noise']
analyzer_neural = analyzer.remove_units(list(noise_units.index))
analyzer_noise = analyzer.select_units(list(noise_units.index))

# Apply the sua/mua model
sua_mua_labels = sc.auto_label_units(
    sorting_analyzer=analyzer_neural,
    repo_id="SpikeInterface/UnitRefine_sua_mua_classifier_lightweight",
    trust_model=True,
)

unit_refine_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
sua_units = sua_mua_labels[sua_mua_labels['prediction'] == 'sua']
analyzer_sua = analyzer_neural.select_units(list(sua_units.index))
mua_units = sua_mua_labels[sua_mua_labels['prediction'] == 'mua']
analyzer_mua = analyzer_neural.select_units(list(mua_units.index))

# Let's see what we got
print(f"Noise units: {len(noise_units)} / {num_uncurated_units}")
print(f"MUA units: {len(mua_units)} / {num_uncurated_units}")
print(f"SUA units: {len(sua_units)} / {num_uncurated_units}")

# it is also convenient to store the label and the probability as properties of the sorting analyzer
analyzer.set_sorting_property("unitrefine_prediction", unit_refine_labels["prediction"])
analyzer.set_sorting_property("unitrefine_probability", unit_refine_labels["probability"])

# %%and plot
%matplotlib widget
_ = sw.plot_unit_templates(analyzer_noise, backend="ipywidgets")
_ = sw.plot_unit_templates(analyzer_mua, backend="ipywidgets")
_ = sw.plot_unit_templates(analyzer_sua, backend="ipywidgets")
analyzer_to_process = analyzer_neural

# %% Auto Merge units
#Algorithm to find and check potential merges between units.
#
#The merges are proposed based on a series of steps with different criteria:
#
#    * "num_spikes": enough spikes are found in each unit for computing the correlogram (`min_spikes`)
#    * "snr": the SNR of the units is above a threshold (`min_snr`)
#    * "remove_contaminated": each unit is not contaminated (by checking auto-correlogram - `contamination_thresh`)
#    * "unit_locations": estimated unit locations are close enough (`max_distance_um`)
#    * "correlogram": the cross-correlograms of the two units are similar to each auto-corrleogram (`corr_diff_thresh`)
#    * "template_similarity": the templates of the two units are similar (`template_diff_thresh`)
#    * "presence_distance": the presence of the units is complementary in time (`presence_distance_thresh`)
#    * "cross_contamination": the cross-contamination is not significant (`cc_thresh` and `p_value`)

potential_merges = sc.compute_merge_unit_groups(
    analyzer_neural,
    preset="similarity_correlograms",
    steps_params={"template_similarity": {"template_diff_thresh": 0.5}}
)
print(potential_merges)
# %%and plot
%matplotlib widget
sw.plot_potential_merges(analyzer_neural, potential_merges=potential_merges, backend="ipywidgets")



# %% How to save all of this and to use it for the final curation in the GUI. We will 
# use the curation module to save the curation info

# we will use unitrefine labels as pre-computed "quality"
unitrefine_labels = analyzer.get_sorting_property("unitrefine_prediction")
unitrefine_labels[unitrefine_labels == "sua"] = "good"
unitrefine_labels[unitrefine_labels == "mua"] = "MUA"

label_definitions = {
    "quality": dict(name="quality", label_options=["good", "MUA", "noise"], exclusive=True),
}
manual_labels = [
    {"unit_id": unit_id, "labels": {"quality": [unitrefine_labels[unit_index]]}}
    for unit_index, unit_id in enumerate(analyzer.unit_ids)
]

curation_dict = dict(
    format_version="2",
    unit_ids=analyzer.unit_ids,
    label_definitions=label_definitions, 
    merges=[
        dict(unit_ids=p)
        for p in potential_merges
    ],
    removed=analyzer_noise.unit_ids,
    manual_labels=manual_labels
)
# %% Now that we have a curation with all steps (remove, merge, label) we can apply it in one go!
from spikeinterface.curation import validate_curation_dict
validate_curation_dict(curation_dict)
# we can use the CurationModel for more secure serialization (save a json file)
from spikeinterface.curation.curation_model import CurationModel

curation = CurationModel(**curation_dict)
curation_file = Path("curation.json")
_ = curation_file.write_text(curation.model_dump_json(indent=4))

# we apply the curation to the analyzer and save a new analyzer
analyzer_curated = sc.apply_curation(analyzer, curation_dict, sparsity_overlap=0.5)
analyzer_sua.save_as(format='binary_folder', folder=working_folder / f"analyzer_{'lupin_100s'}_sua")

# %% Finally we can call the curation GUI to visualize and make the last decisions
from spikeinterface_gui import run_mainwindow
run_mainwindow(
    analyzer_qm_filt,
    displayed_unit_properties=["passing_qc", "unitrefine_prediction", "unitrefine_probability"],
    curation=True,
    mode="desktop"
)

#and you can compare again the results between sorters after curation!
#Just reload the analyzer as sorters with sua units only
import spikeinterface.full as si
sorting_KS4 = si.load(working_folder / "analyzer_KS4_sua").sorting
print(sorting_KS4)
sorting_SC2 = si.load(working_folder / "analyzer_SC2_sua").sorting
print(sorting_SC2)
sorting_TDC2 = si.load(working_folder / "analyzer_TDC2_sua").sorting
print(sorting_TDC2)
multi_comp = si.compare_multiple_sorters(
    sorting_list=[sorting_KS4, sorting_SC2, sorting_TDC2],
    name_list=['KS4', 'SC2', 'TDC2'],
    spiketrain_mode='union',
    verbose=True
)
# %% and plot
%matplotlib inline
w = si.plot_multicomparison_agreement(multi_comp)
w = si.plot_multicomparison_agreement_by_sorter(multi_comp)



























