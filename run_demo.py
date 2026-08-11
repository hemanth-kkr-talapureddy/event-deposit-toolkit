"""
End-to-end demonstration of the workflow.

    python run_demo.py

Generates a synthetic core, computes Folk & Ward statistics down its length,
tests whether grain size and geochemistry separate flood beds from seismic
turbidites, and writes three figures to figures/.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import grainsize
import plots
from synthetic_core import generate_core, EVENT_CATALOG


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Generating synthetic core...")
    gs, core, bins_um = generate_core(seed=42)

    bin_cols = [c for c in gs.columns if c != "depth_cm"]
    volumes = gs[bin_cols].to_numpy(dtype=float)

    print(f"  {len(core)} samples, {len(bins_um)} size bins, "
          f"{core['depth_cm'].max():.0f} cm of core")
    print(f"  {core['in_event_bed'].sum()} samples in event beds "
          f"across {len(EVENT_CATALOG)} events")

    print("\nComputing Folk & Ward statistics...")
    stats = grainsize.analyse_core(bins_um, volumes, core["depth_cm"].values)

    merged = core.set_index("depth_cm").join(stats)

    print("\nFacies summary")
    print("-" * 72)
    summary = merged.groupby("facies")[
        ["d50_um", "sorting", "skewness", "sand_pct", "Ti_Ca"]
    ].mean().round(3)
    print(summary.to_string())

    print("\nSeparation check: flood beds vs seismic turbidites")
    print("-" * 72)
    flood = merged[merged["facies"] == "flood"]
    quake = merged[merged["facies"] == "earthquake"]

    for var in ["d50_um", "sorting", "skewness", "sand_pct", "Ti_Ca"]:
        f_mean, q_mean = flood[var].mean(), quake[var].mean()
        pooled_sd = np.sqrt((flood[var].var() + quake[var].var()) / 2)
        cohens_d = abs(f_mean - q_mean) / pooled_sd if pooled_sd > 0 else np.nan
        verdict = (
            "strong" if cohens_d > 1.5
            else "moderate" if cohens_d > 0.8
            else "weak"
        )
        print(f"  {var:<12} flood={f_mean:>9.3f}  quake={q_mean:>9.3f}   "
              f"|d|={cohens_d:>5.2f}  ({verdict})")

    print("\nEvent beds matched to the historical catalogue")
    print("-" * 72)
    beds = (
        merged[merged["in_event_bed"]]
        .groupby(["event_name", "facies"])
        .agg(top_cm=("d50_um", lambda s: s.index.min()),
             thickness_cm=("d50_um", "size"),
             mean_d50=("d50_um", "mean"),
             mean_TiCa=("Ti_Ca", "mean"))
        .sort_values("top_cm")
        .round(3)
    )
    print(beds.to_string())

    print("\nWriting figures...")
    plots.downcore_log(stats, core, max_depth=100,
                       savepath="figures/downcore_log.png")
    plots.distribution_comparison(bins_um, gs, core,
                                  savepath="figures/distributions_by_facies.png")
    plots.discrimination_plot(stats, core,
                              savepath="figures/discrimination.png")

    gs.to_csv("data/grainsize_synthetic.csv", index=False)
    core.to_csv("data/core_metadata_synthetic.csv", index=False)
    pd.DataFrame(EVENT_CATALOG).to_csv("data/event_catalog.csv", index=False)
    stats.to_csv("data/grainsize_statistics.csv")

    print("  figures/downcore_log.png")
    print("  figures/distributions_by_facies.png")
    print("  figures/discrimination.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
