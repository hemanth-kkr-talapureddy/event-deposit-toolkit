"""
Synthetic lacustrine core generator.

Produces a stand-in dataset with the structure of a real karst-lake core:
laser-diffraction grain-size distributions, XRF elemental counts, an
age-depth model, and a catalogue of historical events to test against.

The event beds are built with contrasting signatures so that the analysis
workflow has something real to separate. Those signatures are simplified
versions of criteria discussed in the event-deposit literature:

  Flood / hyperpycnal beds
      - finer overall, better sorted
      - inverse-to-normal grading (coarsening-up base, fining-up top)
      - terrigenous geochemical enrichment: Ti up, Ca diluted

  Seismically triggered turbidites
      - coarser, poorly sorted, often polymodal base
      - simple normal grading
      - dominated by remobilised intrabasinal material, so the carbonate
        signal is retained and Ti/Ca stays near background

Real cores are considerably messier than this, and the criteria above are
actively debated rather than settled. Treat the generator as a testbed for
the code, not as a model of how lakes work.
"""

import numpy as np
import pandas as pd

# Mastersizer 3000-like log-spaced bins, in micrometres
N_BINS = 90
BINS_UM = np.logspace(np.log10(0.05), np.log10(2000.0), N_BINS)

TOP_YEAR = 2024
SED_RATE_CM_PER_YR = 0.35
CORE_LENGTH_CM = 200


# Illustrative catalogue of events known to have affected northern Guatemala.
# Dates are real; the sedimentary response is invented.
EVENT_CATALOG = [
    {"year": 2020, "name": "Hurricanes Eta / Iota", "type": "flood",      "thickness_cm": 4},
    {"year": 2010, "name": "Tropical Storm Agatha", "type": "flood",      "thickness_cm": 2},
    {"year": 2005, "name": "Hurricane Stan",        "type": "flood",      "thickness_cm": 3},
    {"year": 1998, "name": "Hurricane Mitch",       "type": "flood",      "thickness_cm": 5},
    {"year": 1976, "name": "Motagua M7.5",          "type": "earthquake", "thickness_cm": 6},
    {"year": 1974, "name": "Hurricane Fifi",        "type": "flood",      "thickness_cm": 3},
    {"year": 1942, "name": "Guatemala M7.9",        "type": "earthquake", "thickness_cm": 4},
    {"year": 1902, "name": "Quetzaltenango M7.5",   "type": "earthquake", "thickness_cm": 5},
    {"year": 1816, "name": "Guatemala M7.5",        "type": "earthquake", "thickness_cm": 4},
    {"year": 1785, "name": "Guatemala M7.0",        "type": "earthquake", "thickness_cm": 3},
]


def _phi_axis():
    return -np.log2(BINS_UM / 1000.0)


def _lognormal_mode(phi_axis, mean_phi, sd_phi, weight=1.0):
    """A single Gaussian mode in phi space, returned as volume percent."""
    mode = np.exp(-0.5 * ((phi_axis - mean_phi) / sd_phi) ** 2)
    return weight * mode


def _make_distribution(modes, rng, noise=0.03):
    """
    Build a volume-percent distribution from a list of (mean_phi, sd, weight).

    Adds multiplicative noise so no two samples are identical, then
    normalises to sum to 100.
    """
    phi_axis = _phi_axis()
    dist = np.zeros_like(phi_axis)
    for mean_phi, sd_phi, weight in modes:
        dist += _lognormal_mode(phi_axis, mean_phi, sd_phi, weight)

    dist *= 1.0 + rng.normal(0, noise, size=dist.shape)
    dist = np.clip(dist, 0, None)
    return dist / dist.sum() * 100.0


def _background_sample(rng):
    """Hemipelagic carbonate mud: fine, unimodal, moderately poorly sorted."""
    mean_phi = rng.normal(7.6, 0.18)
    sd_phi = rng.normal(1.55, 0.08)
    return _make_distribution([(mean_phi, sd_phi, 1.0)], rng)


def _flood_sample(rng, position):
    """
    Flood bed. `position` runs 0 (base) to 1 (top).

    Inverse grading through the lowest third, then normal grading above --
    the classic hyperpycnal signature of a rising then falling discharge.
    """
    if position < 0.33:
        coarseness = 5.9 - 0.9 * (position / 0.33)
    else:
        coarseness = 5.0 + 2.4 * ((position - 0.33) / 0.67)

    mean_phi = rng.normal(coarseness, 0.12)
    sd_phi = rng.normal(1.05, 0.06)   # better sorted than background
    return _make_distribution([(mean_phi, sd_phi, 1.0)], rng)


def _seismic_sample(rng, position):
    """
    Seismic turbidite. Coarse, poorly sorted, polymodal base with simple
    normal grading upward.
    """
    coarse_mean = rng.normal(3.4 + 3.6 * position, 0.15)
    coarse_weight = max(0.0, 1.0 - 0.75 * position)
    fine_mean = rng.normal(7.5, 0.2)
    fine_weight = 0.35 + 0.65 * position

    modes = [
        (coarse_mean, rng.normal(1.5, 0.08), coarse_weight),
        (fine_mean, rng.normal(1.7, 0.08), fine_weight),
    ]
    return _make_distribution(modes, rng)


def _xrf_sample(rng, facies, position=0.0):
    """
    XRF counts for the major elements measured on a typical core scanner.

    Ti and K track detrital input from the catchment. Ca tracks endogenic
    carbonate produced in the lake. Their ratio is the discriminant.
    """
    if facies == "background":
        ti = rng.normal(1800, 150)
        ca = rng.normal(52000, 3000)
        fe = rng.normal(14000, 900)
        k = rng.normal(6200, 400)
        si = rng.normal(21000, 1500)
        mn = rng.normal(900, 90)

    elif facies == "flood":
        # Strong catchment erosion signal: detritals up, carbonate diluted
        intensity = 1.0 - 0.6 * position
        ti = rng.normal(1800 + 4200 * intensity, 250)
        ca = rng.normal(52000 - 26000 * intensity, 2500)
        fe = rng.normal(14000 + 9000 * intensity, 1000)
        k = rng.normal(6200 + 5000 * intensity, 450)
        si = rng.normal(21000 + 14000 * intensity, 1600)
        mn = rng.normal(900 - 250 * intensity, 90)

    else:  # earthquake
        # Remobilised in-lake sediment: carbonate largely preserved,
        # detrital enrichment modest
        intensity = 1.0 - 0.5 * position
        ti = rng.normal(1800 + 700 * intensity, 200)
        ca = rng.normal(52000 - 3000 * intensity, 3000)
        fe = rng.normal(14000 + 2200 * intensity, 950)
        k = rng.normal(6200 + 900 * intensity, 420)
        si = rng.normal(21000 + 3500 * intensity, 1500)
        mn = rng.normal(900 + 400 * intensity, 110)

    return {
        "Ti": max(ti, 0), "Ca": max(ca, 0), "Fe": max(fe, 0),
        "K": max(k, 0), "Si": max(si, 0), "Mn": max(mn, 0),
    }


def generate_core(seed=42):
    """
    Build the full synthetic core.

    Event beds are instantaneous deposits, so they occupy depth without
    consuming time. The age model is therefore built on an event-free depth
    scale, which is what you would do with a real core before radiometric
    dating.

    Returns
    -------
    grainsize : DataFrame  (index depth_cm, columns = bin diameters in um)
    core      : DataFrame  (depth, age, facies, event name, XRF counts)
    bins_um   : ndarray
    """
    rng = np.random.default_rng(seed)

    events_by_depth = {}
    for ev in EVENT_CATALOG:
        ef_depth = round((TOP_YEAR - ev["year"]) * SED_RATE_CM_PER_YR)
        events_by_depth[ef_depth] = ev

    distributions = []
    records = []
    depth = 0.0

    for ef_depth in range(CORE_LENGTH_CM):
        if ef_depth in events_by_depth:
            ev = events_by_depth[ef_depth]
            thickness = ev["thickness_cm"]

            # Event beds are emitted top-down, so position 1 is the top of
            # the bed and position 0 the base.
            for i in range(thickness):
                position = 1.0 - (i / max(thickness - 1, 1))
                if ev["type"] == "flood":
                    dist = _flood_sample(rng, position)
                else:
                    dist = _seismic_sample(rng, position)

                xrf = _xrf_sample(rng, ev["type"], position)
                distributions.append(dist)
                records.append({
                    "depth_cm": depth,
                    "age_yr_ce": ev["year"],
                    "facies": ev["type"],
                    "event_name": ev["name"],
                    "in_event_bed": True,
                    **xrf,
                })
                depth += 1.0

        dist = _background_sample(rng)
        xrf = _xrf_sample(rng, "background")
        records.append({
            "depth_cm": depth,
            "age_yr_ce": TOP_YEAR - ef_depth / SED_RATE_CM_PER_YR,
            "facies": "background",
            "event_name": None,
            "in_event_bed": False,
            **xrf,
        })
        distributions.append(dist)
        depth += 1.0

    core = pd.DataFrame(records)
    core["Ti_Ca"] = core["Ti"] / core["Ca"]
    core["K_Ca"] = core["K"] / core["Ca"]

    grainsize = pd.DataFrame(
        np.array(distributions),
        columns=[f"{b:.4g}" for b in BINS_UM],
    )
    grainsize.insert(0, "depth_cm", core["depth_cm"].values)

    return grainsize, core, BINS_UM


def write_csvs(outdir="data"):
    """Write the synthetic dataset to disk in a Mastersizer-like layout."""
    import os
    os.makedirs(outdir, exist_ok=True)

    grainsize, core, _ = generate_core()
    grainsize.to_csv(os.path.join(outdir, "grainsize_synthetic.csv"), index=False)
    core.to_csv(os.path.join(outdir, "core_metadata_synthetic.csv"), index=False)
    pd.DataFrame(EVENT_CATALOG).to_csv(
        os.path.join(outdir, "event_catalog.csv"), index=False
    )
    return outdir


if __name__ == "__main__":
    print(f"Synthetic data written to: {write_csvs()}")
