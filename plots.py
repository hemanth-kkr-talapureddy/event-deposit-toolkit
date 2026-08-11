"""
Figures for event-deposit analysis: downcore logs, distribution curves,
and a grain-size / geochemistry discrimination plot.
"""

import matplotlib.pyplot as plt
import numpy as np

FACIES_COLORS = {
    "background": "#9aa5b1",
    "flood": "#2b7bba",
    "earthquake": "#c0392b",
}


def _shade_event_beds(ax, core):
    """Shade event intervals across a downcore panel."""
    in_bed = core["in_event_bed"].values
    depths = core["depth_cm"].values
    facies = core["facies"].values

    start = None
    for i, flag in enumerate(in_bed):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ax.axhspan(
                depths[start] - 0.5, depths[i - 1] + 0.5,
                color=FACIES_COLORS[facies[start]], alpha=0.18, lw=0,
            )
            start = None
    if start is not None:
        ax.axhspan(
            depths[start] - 0.5, depths[-1] + 0.5,
            color=FACIES_COLORS[facies[start]], alpha=0.18, lw=0,
        )


def downcore_log(stats, core, max_depth=100, savepath=None):
    """
    Multi-panel downcore log: median grain size, sorting, sand fraction
    and the Ti/Ca terrigenous proxy, with event beds shaded.
    """
    merged = core.set_index("depth_cm").join(stats)
    merged = merged[merged.index <= max_depth]

    panels = [
        ("d50_um", "D50 (um)", False),
        ("sorting", "Sorting (phi)", False),
        ("sand_pct", "Sand (%)", False),
        ("Ti_Ca", "Ti / Ca", False),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(13, 8), sharey=True)

    for ax, (col, label, logx) in zip(axes, panels):
        _shade_event_beds(ax, merged.reset_index())
        ax.plot(merged[col], merged.index, color="#1f2933", lw=1.1)
        ax.set_xlabel(label, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        if logx:
            ax.set_xscale("log")

    axes[0].set_ylabel("Depth (cm)", fontsize=11)
    axes[0].invert_yaxis()

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FACIES_COLORS["flood"], alpha=0.35),
        plt.Rectangle((0, 0), 1, 1, color=FACIES_COLORS["earthquake"], alpha=0.35),
    ]
    fig.legend(
        handles, ["Flood bed", "Seismic turbidite"],
        loc="lower center", ncol=2, frameon=False, fontsize=10,
    )
    fig.suptitle("Downcore event-deposit log (synthetic)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig


def distribution_comparison(bins_um, grainsize, core, savepath=None):
    """
    Representative volume distributions for each facies, plus the mean
    curve for that facies across the core.
    """
    bin_cols = [c for c in grainsize.columns if c != "depth_cm"]
    volumes = grainsize[bin_cols].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, facies in zip(axes, ["background", "flood", "earthquake"]):
        mask = (core["facies"] == facies).to_numpy()
        subset = volumes[mask]

        for row in subset[:: max(len(subset) // 12, 1)]:
            ax.plot(bins_um, row, color=FACIES_COLORS[facies], alpha=0.25, lw=0.8)

        ax.plot(bins_um, subset.mean(axis=0),
                color=FACIES_COLORS[facies], lw=2.2, label="facies mean")

        ax.set_xscale("log")
        ax.set_xlabel("Grain diameter (um)", fontsize=10)
        ax.set_title(facies.capitalize(), fontsize=11)
        ax.grid(alpha=0.25, lw=0.5)
        ax.axvline(63, color="#52606d", ls="--", lw=0.8)
        ax.axvline(2, color="#52606d", ls=":", lw=0.8)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Volume (%)", fontsize=10)
    fig.suptitle(
        "Grain-size distributions by facies  (dashed = sand/silt, dotted = silt/clay)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig


def discrimination_plot(stats, core, savepath=None):
    """
    The point of the whole exercise: does grain size plus geochemistry
    separate the two event types?

    Sorting on x, Ti/Ca on y. Flood beds should plot well-sorted and
    terrigenous-enriched; seismic turbidites poorly sorted with a
    near-background carbonate signal.
    """
    merged = core.set_index("depth_cm").join(stats)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for facies in ["background", "earthquake", "flood"]:
        sub = merged[merged["facies"] == facies]
        ax.scatter(
            sub["sorting"], sub["Ti_Ca"],
            c=FACIES_COLORS[facies], s=38, alpha=0.75,
            edgecolor="white", linewidth=0.5,
            label=facies.capitalize(),
        )

    ax.set_xlabel("Sorting, inclusive graphic standard deviation (phi)", fontsize=11)
    ax.set_ylabel("Ti / Ca  (detrital / endogenic carbonate)", fontsize=11)
    ax.set_title("Event-deposit discrimination (synthetic data)", fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig
