"""
Grain-size statistics from laser diffraction volume distributions.

Implements the Folk & Ward (1957) graphical measures, computed in phi units
from cumulative volume distributions. Designed for Malvern Mastersizer 3000
exports but works with any binned volume-percent data.

Convention note
---------------
phi = -log2(d_mm). Phi increases as grain size decreases, so cumulative
curves are built in ascending phi (coarse -> fine). Under this convention
phi_16 sits at the coarse end of the distribution and phi_84 at the fine end,
which is what the Folk & Ward formulae assume.
"""

import numpy as np
import pandas as pd


def um_to_phi(diameter_um):
    """Convert diameter in micrometres to phi units."""
    d = np.asarray(diameter_um, dtype=float)
    return -np.log2(d / 1000.0)


def phi_to_um(phi):
    """Convert phi units back to diameter in micrometres."""
    return 1000.0 * 2.0 ** (-np.asarray(phi, dtype=float))


def cumulative_curve(bins_um, volume_pct):
    """
    Build a cumulative volume curve in ascending phi.

    Returns
    -------
    phi_sorted : ndarray
        Bin centres in phi, ascending (coarse to fine).
    cum_pct : ndarray
        Cumulative volume percent, normalised to 0-100.
    """
    phi = um_to_phi(bins_um)
    vol = np.asarray(volume_pct, dtype=float)

    order = np.argsort(phi)
    phi_sorted = phi[order]
    vol_sorted = vol[order]

    total = vol_sorted.sum()
    if total <= 0:
        raise ValueError("Volume percentages sum to zero; check input.")

    cum_pct = np.cumsum(vol_sorted) / total * 100.0
    return phi_sorted, cum_pct


def percentiles_phi(bins_um, volume_pct, targets=(5, 16, 25, 50, 75, 84, 95)):
    """
    Interpolate phi values at given cumulative percentages.

    Linear interpolation in phi space, which is standard practice for
    graphical grain-size statistics.
    """
    phi_sorted, cum_pct = cumulative_curve(bins_um, volume_pct)

    # np.interp needs a strictly increasing x; cumulative curves can plateau
    # across empty bins, so keep only the first occurrence of each value.
    keep = np.concatenate(([True], np.diff(cum_pct) > 0))
    cum_clean = cum_pct[keep]
    phi_clean = phi_sorted[keep]

    values = np.interp(targets, cum_clean, phi_clean)
    return dict(zip(targets, values))


def folk_ward(bins_um, volume_pct):
    """
    Folk & Ward (1957) graphical grain-size statistics.

    Returns a dict with:
        mean_phi, median_phi, sorting, skewness, kurtosis  (phi-based)
        d10_um, d50_um, d90_um                             (micrometres)
        mean_um                                            (micrometres)

    Sorting is the inclusive graphic standard deviation. Skewness is
    dimensionless and positive for fine-tailed distributions. Kurtosis is
    the graphic kurtosis (KG).
    """
    p = percentiles_phi(bins_um, volume_pct)
    p5, p16, p25, p50, p75, p84, p95 = (
        p[5], p[16], p[25], p[50], p[75], p[84], p[95]
    )

    mean_phi = (p16 + p50 + p84) / 3.0
    sorting = (p84 - p16) / 4.0 + (p95 - p5) / 6.6

    skew_num1 = (p16 + p84 - 2 * p50)
    skew_den1 = 2 * (p84 - p16)
    skew_num2 = (p5 + p95 - 2 * p50)
    skew_den2 = 2 * (p95 - p5)
    skewness = (
        (skew_num1 / skew_den1 if skew_den1 != 0 else 0.0)
        + (skew_num2 / skew_den2 if skew_den2 != 0 else 0.0)
    )

    kurt_den = 2.44 * (p75 - p25)
    kurtosis = (p95 - p5) / kurt_den if kurt_den != 0 else np.nan

    # Percentiles in phi run coarse->fine, so the phi-16 value is the
    # coarse end: D90 in micrometres corresponds to phi_10.
    p_extra = percentiles_phi(bins_um, volume_pct, targets=(10, 50, 90))

    return {
        "mean_phi": mean_phi,
        "median_phi": p50,
        "sorting": sorting,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "d10_um": phi_to_um(p_extra[90]),
        "d50_um": phi_to_um(p_extra[50]),
        "d90_um": phi_to_um(p_extra[10]),
        "mean_um": phi_to_um(mean_phi),
    }


def sorting_class(sorting):
    """Folk & Ward verbal sorting classification."""
    edges = [0.35, 0.50, 0.71, 1.00, 2.00, 4.00]
    labels = [
        "very well sorted",
        "well sorted",
        "moderately well sorted",
        "moderately sorted",
        "poorly sorted",
        "very poorly sorted",
        "extremely poorly sorted",
    ]
    return labels[int(np.searchsorted(edges, sorting))]


def size_fractions(bins_um, volume_pct):
    """
    Percent clay (<2 um), silt (2-63 um) and sand (>63 um).

    Uses the Udden-Wentworth boundaries most common in lacustrine work.
    """
    bins = np.asarray(bins_um, dtype=float)
    vol = np.asarray(volume_pct, dtype=float)
    total = vol.sum()

    clay = vol[bins < 2.0].sum() / total * 100.0
    silt = vol[(bins >= 2.0) & (bins < 63.0)].sum() / total * 100.0
    sand = vol[bins >= 63.0].sum() / total * 100.0
    return {"clay_pct": clay, "silt_pct": silt, "sand_pct": sand}


def analyse_core(bins_um, volume_matrix, depths_cm):
    """
    Run statistics down a whole core.

    Parameters
    ----------
    bins_um : array-like, shape (n_bins,)
    volume_matrix : array-like, shape (n_samples, n_bins)
    depths_cm : array-like, shape (n_samples,)

    Returns
    -------
    pandas.DataFrame indexed by depth.
    """
    rows = []
    for depth, row in zip(depths_cm, np.asarray(volume_matrix)):
        stats = folk_ward(bins_um, row)
        stats.update(size_fractions(bins_um, row))
        stats["sorting_class"] = sorting_class(stats["sorting"])
        stats["depth_cm"] = depth
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("depth_cm")
    return df


def read_mastersizer_csv(path, bin_row=0):
    """
    Load a Malvern Xplorer wide-format export.

    Expects size-bin diameters (um) as column headers and one row per
    measurement. Adjust to match your own export layout -- Xplorer's
    column arrangement is configurable, so this is a starting point rather
    than a universal reader.
    """
    df = pd.read_csv(path)
    bin_cols = [c for c in df.columns if _is_number(c)]
    bins_um = np.array([float(c) for c in bin_cols])
    volumes = df[bin_cols].to_numpy(dtype=float)
    meta = df.drop(columns=bin_cols)
    return bins_um, volumes, meta


def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
