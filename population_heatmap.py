"""
population_heatmap.py

Python version of Population_heatmap.jl for Spyder.

Dependencies:
    pip install numpy pandas matplotlib scipy h5py pynwb

Run:
    1. Put config_psth.py in the same folder as this script.
    2. Edit config_psth.py.
    3. Run this file in Spyder.
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

try:
    from pynwb import NWBHDF5IO
except ImportError as exc:
    raise ImportError(
        "pynwb is required. Install with: pip install pynwb"
    ) from exc

import config_psth as cfg


@dataclass
class Unit:
    spike_times: np.ndarray
    region: str
    depth_um: float
    session_id: str


# =============================================================================
# IO helpers
# =============================================================================

def _as_clean_str(x):
    """Convert bytes/objects from HDF5/NWB to a clean Python string."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip("\x00")
    return str(x).strip("\x00")


def _read_h5_vector(nwb_path, h5_path):
    """
    Read a numeric or string vector directly from the NWB/HDF5 path.

    This works for paths such as:
        intervals/trials/imec_blue_led_on
        intervals/trials/imec_stim_on
        general/extracellular_ephys/electrodes/location
    """
    h5_path = "/" + h5_path.strip("/")
    with h5py.File(nwb_path, "r") as f:
        if h5_path not in f:
            raise KeyError(f"Missing NWB path: {h5_path}")
        data = f[h5_path][()]
    return np.asarray(data)


def _nwb_files(data_path):
    """Return one or more NWB files from a file or directory path."""
    path = Path(data_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.nwb"))
    raise FileNotFoundError(f"Could not find data_path: {data_path}")


def load_events(nwb_path, event_path):
    """Load event timestamps and remove NaN / negative missing values."""
    x = _read_h5_vector(nwb_path, event_path).astype(float).ravel()
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    return x


def _load_electrode_metadata(nwb_path):
    """
    Load electrode locations and depths from common NWB electrode-table paths.
    Returns dictionaries keyed by electrode row index.
    """
    locations = None
    depths = None

    with h5py.File(nwb_path, "r") as f:
        loc_path = "/general/extracellular_ephys/electrodes/location"
        dep_path = "/general/extracellular_ephys/electrodes/depth_um"

        if loc_path in f:
            locations = [_as_clean_str(v) for v in np.asarray(f[loc_path][()]).ravel()]
        if dep_path in f:
            depths = np.asarray(f[dep_path][()]).astype(float).ravel()

    n = 0
    if locations is not None:
        n = max(n, len(locations))
    if depths is not None:
        n = max(n, len(depths))

    loc_by_idx = {i: locations[i] if locations is not None and i < len(locations) else "unknown"
                  for i in range(n)}
    dep_by_idx = {i: depths[i] if depths is not None and i < len(depths) else np.nan
                  for i in range(n)}
    return loc_by_idx, dep_by_idx


def _infer_unit_region_depth(unit_row, unit_i, loc_by_idx, dep_by_idx):
    """
    Try to map a unit to an electrode location/depth.

    Different NWB exports store the electrode/unit relation differently.
    This function tries common cases and falls back safely.
    """
    # Case 1: row contains an electrodes column with one or more electrode indices
    if "electrodes" in unit_row.index:
        electrodes = unit_row["electrodes"]
        try:
            arr = np.asarray(electrodes).astype(int).ravel()
            if len(arr) > 0:
                e = int(arr[0])
                return loc_by_idx.get(e, "unknown"), dep_by_idx.get(e, np.nan)
        except Exception:
            pass

    # Case 2: row contains a simple electrode id/index field
    for key in ("electrode", "electrode_id", "peak_channel_id", "channel_id"):
        if key in unit_row.index:
            try:
                e = int(unit_row[key])
                return loc_by_idx.get(e, "unknown"), dep_by_idx.get(e, np.nan)
            except Exception:
                pass

    # Case 3: use unit index as a fallback
    return loc_by_idx.get(unit_i, "unknown"), dep_by_idx.get(unit_i, np.nan)


def load_units_one_file(nwb_path):
    """Load units from one NWB file using PyNWB."""
    session_id = Path(nwb_path).stem
    loc_by_idx, dep_by_idx = _load_electrode_metadata(nwb_path)

    io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
    nwb = io.read()

    if nwb.units is None:
        io.close()
        raise ValueError(f"No units table found in {nwb_path}")

    df = nwb.units.to_dataframe()
    units = []

    for i, (_, row) in enumerate(df.iterrows()):
        spike_times = np.asarray(row["spike_times"], dtype=float)
        spike_times = spike_times[np.isfinite(spike_times)]

        region, depth = _infer_unit_region_depth(row, i, loc_by_idx, dep_by_idx)

        units.append(Unit(
            spike_times=np.sort(spike_times),
            region=region,
            depth_um=float(depth) if np.isfinite(depth) else np.nan,
            session_id=session_id,
        ))

    io.close()
    return units


def load_all_units(data_path):
    """Load units from one or more NWB files."""
    all_units = []
    for path in _nwb_files(data_path):
        all_units.extend(load_units_one_file(path))
    return all_units


def load_all_events(data_path, event_path):
    """Load event arrays from one or more NWB files, matching the file order."""
    return [load_events(path, event_path) for path in _nwb_files(data_path)]


# =============================================================================
# Analysis helpers
# =============================================================================

def filter_units(units, min_firing_rate=0.0, regions=None):
    """Filter units by mean firing rate and optional region list."""
    if regions is None:
        regions = []

    kept = []
    for u in units:
        if len(u.spike_times) < 2:
            continue

        duration = u.spike_times[-1] - u.spike_times[0]
        firing_rate = len(u.spike_times) / duration if duration > 0 else 0

        if min_firing_rate and firing_rate < min_firing_rate:
            continue

        if regions and u.region not in regions:
            continue

        kept.append(u)

    return kept


def population_psth_multi(units, events_by_file, bin_width, win_start, win_stop):
    """
    Compute PSTH matrix, rows = units, columns = peri-event bins.

    For multiple NWB files, units are matched to events by session_id/stem.
    """
    edges = np.arange(win_start, win_stop + bin_width, bin_width)
    t = edges[:-1] + bin_width / 2
    mat = np.zeros((len(units), len(t)), dtype=float)

    # Map session_id to events
    files = _nwb_files(cfg.data_path)
    session_to_events = {
        Path(path).stem: events_by_file[i]
        for i, path in enumerate(files)
    }

    for i, u in enumerate(units):
        events = session_to_events.get(u.session_id)
        if events is None or len(events) == 0:
            mat[i, :] = np.nan
            continue

        counts = np.zeros(len(t), dtype=float)

        for ev in events:
            lo = ev + win_start
            hi = ev + win_stop
            left = np.searchsorted(u.spike_times, lo, side="left")
            right = np.searchsorted(u.spike_times, hi, side="right")
            rel_spikes = u.spike_times[left:right] - ev
            counts += np.histogram(rel_spikes, bins=edges)[0]

        # Convert counts/bin/event to Hz
        mat[i, :] = counts / (len(events) * bin_width)

    return mat, t


def zscore_psth(mat, t, baseline_stop=0.0):
    """Z-score each unit using baseline bins."""
    baseline = mat[:, t < baseline_stop]

    mu = np.nanmean(baseline, axis=1, keepdims=True)
    sd = np.nanstd(baseline, axis=1, keepdims=True)

    sd[sd == 0] = np.nan
    z = (mat - mu) / sd
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z


def smooth_psth(mat, sigma_bins):
    """Gaussian smooth along time axis."""
    return gaussian_filter1d(mat, sigma=sigma_bins, axis=1, mode="nearest")


def peak_sort(mat, t):
    """Sort rows by time of peak response."""
    peak_idx = np.nanargmax(mat, axis=1)
    return np.argsort(t[peak_idx])


def event_mean_relative(data_path, ref_event_path, other_event_path):
    """
    Compute mean timing of another event relative to the alignment event.
    Uses trial-wise pairing after truncating to the shorter vector.
    """
    rel_all = []

    for path in _nwb_files(data_path):
        try:
            ref = _read_h5_vector(path, ref_event_path).astype(float).ravel()
            other = _read_h5_vector(path, other_event_path).astype(float).ravel()
        except KeyError:
            continue

        n = min(len(ref), len(other))
        ref = ref[:n]
        other = other[:n]

        valid = np.isfinite(ref) & np.isfinite(other) & (ref >= 0) & (other >= 0)
        if np.any(valid):
            rel_all.append(other[valid] - ref[valid])

    if not rel_all:
        return None, 0

    rel_all = np.concatenate(rel_all)
    return float(np.mean(rel_all)), int(len(rel_all))


# =============================================================================
# Plot helpers
# =============================================================================

def region_order_from_depth(units):
    """Return region order from shallow to deep, using mean depth."""
    region_depth = {}
    for r in sorted(set(u.region for u in units)):
        depths = np.asarray([u.depth_um for u in units if u.region == r], dtype=float)
        if np.all(~np.isfinite(depths)):
            region_depth[r] = np.inf
        else:
            region_depth[r] = float(np.nanmean(depths))

    shallow_to_deep = sorted(region_depth, key=lambda r: region_depth[r])
    return shallow_to_deep, region_depth


def add_event_line(ax, x, color="white", linewidth=1.2):
    ax.axvline(x, color=color, linestyle="--", linewidth=linewidth)


def save_figure(fig, stem):
    if not cfg.save_path:
        return

    outdir = Path(cfg.save_path)
    outdir.mkdir(parents=True, exist_ok=True)

    out = outdir / f"{stem}.{cfg.save_format}"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading units and events...")

    units = load_all_units(cfg.data_path)
    units = filter_units(
        units,
        min_firing_rate=cfg.min_firing_rate,
        regions=cfg.regions,
    )

    events_by_file = load_all_events(cfg.data_path, cfg.event_path)

    if len(units) == 0:
        raise RuntimeError("No units left after filtering.")

    n_units = len(units)
    print(f"Total units: {n_units} across {len(_nwb_files(cfg.data_path))} session(s)")

    # Optional event markers, equivalent to your Julia additions
    marker_specs = [
        ("imec_stim_on", "intervals/trials/imec_stim_on", "green"),
        ("imec_stim_off", "intervals/trials/imec_stim_off", "green"),
        ("imec_lick", "intervals/trials/imec_lick", "purple"),
    ]

    marker_lines = []
    for name, path, color in marker_specs:
        mean_rel, n_valid = event_mean_relative(cfg.data_path, cfg.event_path, path)
        if mean_rel is not None:
            print(f"Mean {name} relative to alignment event: {mean_rel:.3f} s "
                  f"({n_valid} valid pairs)")
            marker_lines.append((mean_rel, color, name))

    # Region ordering
    shallow_to_deep, region_depth = region_order_from_depth(units)
    present_regions = list(dict.fromkeys(u.region for u in units))
    shallow_to_deep = [r for r in shallow_to_deep if r in present_regions]

    # Matrix row order must be deep -> shallow so the plot displays shallow at top
    plot_regions = list(reversed(shallow_to_deep))

    region_sort = []
    ylab = np.asarray([u.region for u in units])
    for r in plot_regions:
        region_sort.extend(np.where(ylab == r)[0].tolist())
    region_sort = np.asarray(region_sort, dtype=int)

    ylab_sorted_region = ylab[region_sort]

    region_sizes = [int(np.sum(ylab_sorted_region == r)) for r in plot_regions]
    boundaries = np.cumsum(region_sizes)
    region_mids = []
    start = 0
    for size in region_sizes:
        region_mids.append(start + size / 2)
        start += size

    print("Region order, top to bottom:", ", ".join(shallow_to_deep))
    print("Mean depth by region:")
    for r in shallow_to_deep:
        d = region_depth[r]
        print(f"  {r} => {d:.2f}" if np.isfinite(d) else f"  {r} => unknown")

    # Compute PSTHs
    print("Computing PSTHs...")
    mat, t = population_psth_multi(
        units,
        events_by_file,
        cfg.psth_bin,
        cfg.win_start,
        cfg.win_stop,
    )

    mat_reg = mat[region_sort, :]
    z_reg = zscore_psth(mat_reg, t, baseline_stop=cfg.baseline_stop)

    print(f"Smoothing: sigma = {cfg.smooth_sigma} bins "
          f"= {cfg.smooth_sigma * cfg.psth_bin * 1000:.1f} ms")
    s_mat_reg = smooth_psth(mat_reg, cfg.smooth_sigma)
    s_z_reg = smooth_psth(z_reg, cfg.smooth_sigma)

    # Within-region peak sort
    intra_idx = []
    offset = 0
    for r, block_size in zip(plot_regions, region_sizes):
        block = np.arange(offset, offset + block_size)
        local_order = peak_sort(s_z_reg[block, :], t)
        intra_idx.extend(block[local_order].tolist())
        offset += block_size

    intra_idx = np.asarray(intra_idx, dtype=int)
    s_z_sorted = s_z_reg[intra_idx, :]

    print(f"Done. Matrix: {s_z_sorted.shape} units x bins")

    # Plot z-scored heatmap
    fig, ax = plt.subplots(figsize=(10, 15))

    im = ax.imshow(
        s_z_sorted,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0.5, n_units + 0.5],
        cmap="RdBu_r",
        vmin=-cfg.zlim,
        vmax=cfg.zlim,
        interpolation="nearest",
    )

    ax.set_xlabel("Time from event (s)", fontsize=14)
    ax.set_ylabel("")
    ax.set_title("z-scored", fontsize=16)

    # Region labels
    ax.set_yticks(region_mids)
    ax.set_yticklabels(plot_regions, fontsize=10)

    # Main alignment line at t = 0
    add_event_line(ax, 0.0, color="black")

    # Extra event marker lines
    for x, color, _name in marker_lines:
        add_event_line(ax, x, color=color)

    # Region boundaries
    for b in boundaries[:-1]:
        ax.axhline(b + 0.5, color="black", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z", fontsize=14)

    session_ids = ", ".join(sorted(set(u.session_id for u in units)))
    fig.suptitle(f"Region + peak-sorted · {session_ids}", y=0.995, fontsize=14)

    fig.tight_layout()

    stem = "heatmap_" + cfg.event_path.replace("/", "_")
    save_figure(fig, stem)

    plt.show()


if __name__ == "__main__":
    main()
