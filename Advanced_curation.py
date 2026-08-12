#%%
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import json

import spikeinterface.full as si
import spikeinterface.curation as sc
from spikeinterface.curation.curation_model import CurationModel
from spikeinterface.curation import validate_curation_dict

#%% Define paths
# step 1
analyzer_folder = Path('C:/Users/JoanaCatarino/Joana/test_si/1021218_day1_imec1/analyzer_output')
curation_file = analyzer_folder / "curation.json"
unitrefine_file = analyzer_folder / "unitrefine_labels.json"


# %% load SortingAnalyzer and compute PCA
# step 2

if analyzer_folder.is_dir():
    print("Loading analyzer from disk")
    analyzer = si.load(analyzer_folder, load_extensions=True)

# Compute dependencies needed by UnitRefine
required_extensions = [
    "random_spikes",
    "waveforms",
    "templates",
    "noise_levels",
    "spike_amplitudes",
    "correlograms",
    "principal_components",
]

for ext in required_extensions:
    if not analyzer.has_extension(ext):
        print(f"Computing missing extension: {ext}")
        analyzer.compute(ext)

# Recompute quality metrics now that dependencies exist
if analyzer.has_extension("quality_metrics"):
    analyzer.delete_extension("quality_metrics")

analyzer.compute("quality_metrics")

qm = analyzer.get_extension("quality_metrics").get_data()
print(qm.columns.tolist())
#analyzer.compute("principal_components")

########################## Curation ##############################

# %% Remove redundant units 
# step 3
sorting_clean = sc.remove_redundant_units(analyzer)
if len(sorting_clean.unit_ids) < len(analyzer.unit_ids):
    num_redundant = len(analyzer.unit_ids) - len(sorting_clean.unit_ids)
    print(f"Found {num_redundant} redundant units")
    analyzer = analyzer.select_units(sorting_clean.unit_ids)
    
num_uncurated_units = len(analyzer.unit_ids)


# %% Quality metrics based curation
# step 4
qm = analyzer.get_extension("quality_metrics").get_data()
print(f"Available metrics: {qm.columns.values}")

 # From Siegle et al and Le Merre et al papers
curation_query = "amplitude_cutoff < 0.1 and presence_ratio >0.95 and isi_violations_ratio < 1"
qm_filtered = qm.query(curation_query)
print(f"Number of units after curation: {len(qm_filtered)} / {len(qm)}")

# filter the units
units_to_keep = list(qm_filtered.index)
analyzer_qm_filt = analyzer.select_units(units_to_keep)
# we can also save the unit outputs 
passing_qc = np.zeros(len(analyzer.unit_ids), dtype=bool)
passing_qc[analyzer.sorting.ids_to_indices(units_to_keep)] = True
analyzer.set_sorting_property("passing_qc", passing_qc)

# %% Automatic curation with UnitRefine
# step 5
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
analyzer.set_sorting_property(
    "unitrefine_prediction",
    np.array(unit_refine_labels["prediction"].reindex(analyzer.unit_ids).values, dtype="U10")
)
analyzer.set_sorting_property(
    "unitrefine_probability",
    np.array(unit_refine_labels["probability"].reindex(analyzer.unit_ids).values, dtype=float)
)
# %% Auto Merge units
# step 6
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


# %% How to save all of this and to use it for the final curation in the GUI. We will 
# step 7
# use the curation module to save the curation info

# we will use unitrefine labels as pre-computed "quality"
unitrefine_labels = analyzer.get_sorting_property("unitrefine_prediction")
unitrefine_labels = unitrefine_labels.astype(object)  # makes it writable + avoids truncation

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
#%%  Now that we have a curation with all steps (remove, merge, label) we can apply it in one go!
# step 8
validate_curation_dict(curation_dict)
# we can use the CurationModel for more secure serialization (save a json file)
curation = CurationModel(**curation_dict)
curation_file = analyzer_folder / "curation.json"
_ = curation_file.write_text(curation.model_dump_json(indent=4))

# we apply the curation to the analyzer and save a new analyzer
#analyzer_curated = sc.apply_curation(analyzer, curation_dict, sparsity_overlap=0.5)
#analyzer_sua.save_as(format='binary_folder', folder=working_folder / f"analyzer_{'lupin_100s'}_sua")


unitrefine_dict = {
    "unitrefine_prediction": {
        int(uid): pred
        for uid, pred in zip(
            analyzer.unit_ids,
            unit_refine_labels["prediction"].reindex(analyzer.unit_ids).values
        )
    },
    "unitrefine_probability": {
        int(uid): float(prob)
        for uid, prob in zip(
            analyzer.unit_ids,
            unit_refine_labels["probability"].reindex(analyzer.unit_ids).values
        )
    },
    "passing_qc": {
        int(uid): bool(val)
        for uid, val in zip(
            analyzer.unit_ids,
            analyzer.sorting.get_property("passing_qc")
        )
    },
}

kslabel = analyzer.sorting.get_property("KSLabel")
if kslabel is not None:
    unitrefine_dict["KSLabel"] = {
        int(uid): str(val)
        for uid, val in zip(analyzer.unit_ids, kslabel)
    }

unitrefine_file.write_text(json.dumps(unitrefine_dict, indent=4))

# %% Finally we can call the curation GUI to visualize and make the last decisions
# step 9
with open('C:/Users/JoanaCatarino/si_env/Lib/site-packages/spikeinterface_gui/settings/quality_settings.json', 'r') as f:
    settings_dict = json.load(f)
with open ('C:/Users/JoanaCatarino/si_env/Lib/site-packages/spikeinterface_gui/layouts/layout_quality.json', 'r') as f:
    layout_dict = json.load(f)

# %%  
# Step 10  
from spikeinterface_gui import run_mainwindow

extra_unit_properties = {
    "passing_qc": analyzer.sorting.get_property("passing_qc"),
    "unitrefine_prediction": np.array(analyzer.sorting.get_property("unitrefine_prediction"), dtype="U10"),
    "unitrefine_probability": analyzer.sorting.get_property("unitrefine_probability"),
}

kslabel = analyzer.sorting.get_property("KSLabel")
if kslabel is not None:
    extra_unit_properties["KSLabel"] = np.array(kslabel, dtype="U10")

run_mainwindow(
    analyzer,
    mode="desktop",
    extra_unit_properties=extra_unit_properties,
    curation=True,
    curation_dict=curation_dict,
    layout=layout_dict,
    user_settings=settings_dict
)

# %%
