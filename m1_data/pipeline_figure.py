"""Figure: M1 preprocessing pipeline + key EDA findings (matplotlib).

Every number is read from files produced by build_dataset.py and eda.py, so the
figure is reproducible from the repository:

    python -m m1_data.build_dataset
    python -m m1_data.eda
    python -m m1_data.pipeline_figure     -> m1_data/outputs/eda/preprocessing_pipeline.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "outputs"
EDA = OUT / "eda"

# colour per stage kind: (fill, edge, text)
STYLE = {
    "input": ("#F1EFE8", "#888780", "#2C2C2A"),
    "process": ("#E6F1FB", "#185FA5", "#0C447C"),
    "label": ("#FAEEDA", "#854F0B", "#633806"),
    "output": ("#EAF3DE", "#3B6D11", "#27500A"),
}


def load_numbers() -> dict:
    s = json.loads((EDA / "eda_summary.json").read_text())
    inv = pd.read_csv(EDA / "inventory.csv")
    trips = pd.read_csv(EDA / "trip_summary.csv")
    stats_tr = pd.read_csv(OUT / "dataset_stats_train.csv")
    feats = pd.read_parquet(OUT / "features" / "features_train.parquet",
                            columns=["label", "label_reliable", "heading_diff"])
    rel = feats[feats["label_reliable"] == 1]
    files_present = int(inv[[c for c in inv if c.endswith("_exists")]].to_numpy().sum())
    return {
        "trips_train": s["trips"]["train"], "trips_test": s["trips"]["student_test"],
        "files_present": files_present, "files_total": 4 * len(inv),
        "dropped": s["invalid_or_duplicate_gps_rows"],
        "proj_err": s["projection_rel_error_max_pct"],
        "two_way": s["two_way_share_median"],
        "label_med": float(stats_tr["label_dist_median_m"].median()),
        "unreliable": s["train_unreliable_label_share"],
        "found": 1 - s["train_candidate_miss_rate_reliable_only"],
        "hd_true": float(rel.loc[rel["label"] == 1, "heading_diff"].median()),
        "hd_other": float(rel.loc[rel["label"] == 0, "heading_diff"].median()),
        "rows": len(feats), "pos": s["train_positive_rate_rows"],
        "nearest": s["nearest_arc_top1_reliable"], "dt1": s["share_dt_exactly_1s"],
        "gap_trips": s["trips_with_gaps_over_10s"], "n_trips": len(trips),
        "max_gap_h": s["max_gap_s"] / 3600, "jumps": s["trips_with_step_over_200kmh"],
        "breaks": s["routes_with_breaks"], "full_scc": s["trips_full_route_in_largest_scc"],
    }


def main():
    n = load_numbers()
    steps = [
        ("input", "Raw files",
         f"{n['trips_train']} train + {n['trips_test']} student-test trips  ·  .track .nodes .arcs .route",
         f"{n['files_present']} / {n['files_total']} files"),
        ("process", "Clean the GPS track",
         "drop invalid coordinates and non-increasing timestamps", f"{n['dropped']} rows dropped"),
        ("process", "Degrees → metres, per trip",
         "every trip is a different city → own local projection", f"max error {n['proj_err']:.2f}%"),
        ("process", "Road network + spatial index",
         "length, bearing, exits, reverse twin  ·  KD-tree, points every 10 m",
         f"{n['two_way']:.0%} roads two-way"),
        ("label", "Build the answer key",
         "forward-only alignment of route to GPS dots  ·  > 50 m flagged",
         f"median {n['label_med']:.1f} m  ·  {n['unreliable']:.1%} flagged"),
        ("process", "Candidate roads per dot",
         "all roads within 50 m, closest 12  ·  200 m fallback", f"true road found {n['found']:.1%}"),
        ("process", "16 features per candidate",
         "distance · direction · context ±3 s · road structure · neighbourhood",
         f"heading {n['hd_true']:.1f}° vs {n['hd_other']:.1f}°"),
        ("output", "Feature table → M2 models",
         "one row per (dot, candidate)  ·  label 1 = true road",
         f"{n['rows']:,} rows  ·  {n['pos']:.1%} yes"),
    ]
    findings = [
        (f"{n['nearest']:.1%}", "accuracy if you just pick\nthe nearest road", "two-way twins tie →\nheading features"),
        (f"{n['dt1']:.1%}", "of fixes exactly\n1 s apart", "dense enough for\n±3 s context features"),
        (f"{n['gap_trips']} / {n['n_trips']}", f"trips with gaps > 10 s\n(longest {n['max_gap_h']:.1f} h)",
         "M3 must restart\ndecoding after gaps"),
        (f"{n['jumps']}", "trips with a jump\nfaster than 200 km/h", "context features\nresist GPS jumps"),
        (f"{n['breaks']}", "reference routes\nwith breaks", "route treated as\nreference, not truth"),
        (f"{n['full_scc']} / {n['n_trips']}", "routes fully inside the\nmain road network", "M3 must handle\n“no path” cases"),
    ]

    fig = plt.figure(figsize=(18, 11), dpi=150)
    fig.patch.set_facecolor("#FBFBF8")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 110)
    ax.axis("off")

    ax.text(4, 105, "PathSnap · M1 preprocessing pipeline", fontsize=22, weight="bold", color="#14213D", va="center")
    ax.text(4, 101, "From raw GPS + road-network files to the feature table the models learn from "
                    "(training partition unless stated)", fontsize=12, color="#4A5568", va="center")

    # pipeline column
    x0, w, h, gap, top = 4, 100, 9.2, 2.8, 95
    for i, (kind, title, sub, badge) in enumerate(steps):
        fill, edge, txt = STYLE[kind]
        y = top - i * (h + gap) - h
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fill, ec=edge, lw=1.2))
        ax.add_patch(FancyBboxPatch((x0 + 2.2, y + h / 2 - 2.2), 4.4, 4.4,
                                    boxstyle="round,pad=0,rounding_size=2.2", fc=edge, ec=edge))
        ax.text(x0 + 4.4, y + h / 2, str(i + 1), ha="center", va="center", fontsize=12, weight="bold", color="white")
        ax.text(x0 + 9, y + h * 0.66, title, fontsize=13.5, weight="bold", color=txt, va="center")
        ax.text(x0 + 9, y + h * 0.30, sub, fontsize=10.5, color="#4A5568", va="center")
        ax.text(x0 + w - 2, y + h / 2, badge, fontsize=11.5, weight="bold", color=txt, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=edge, lw=0.8))
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x0 + w / 2, y), (x0 + w / 2, y - gap), arrowstyle="-|>",
                                         mutation_scale=14, color="#888780", lw=1.4))

    # findings column
    ax.text(112, 95.5, "What the EDA found, and what it changed", fontsize=14, weight="bold", color="#14213D")
    tw, th, tg = 30, 24, 3
    for k, (big, cap, so) in enumerate(findings):
        col, row = k % 2, k // 2
        tx, ty = 112 + col * (tw + tg), 92 - (row + 1) * th - row * tg
        ax.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="white", ec="#D3D1C7", lw=1))
        ax.text(tx + 2.5, ty + th - 5.5, big, fontsize=22, weight="bold", color="#14213D", va="center")
        ax.text(tx + 2.5, ty + th - 12, cap, fontsize=10.5, color="#4A5568", va="center", linespacing=1.3)
        ax.text(tx + 2.5, ty + 5, so, fontsize=10.5, color="#185FA5", va="center", linespacing=1.3,
                style="italic")

    # legend
    lx = 112
    for j, (kind, name) in enumerate([("input", "Raw input"), ("process", "Processing"),
                                      ("label", "Labels (answer key)"), ("output", "Output to M2")]):
        fill, edge, _ = STYLE[kind]
        xx = lx + (j % 2) * 33
        yy = 8 - (j // 2) * 4
        ax.add_patch(FancyBboxPatch((xx, yy - 1.1), 2.6, 2.2, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=fill, ec=edge, lw=1))
        ax.text(xx + 3.8, yy, name, fontsize=10.5, color="#4A5568", va="center")

    out = EDA / "preprocessing_pipeline.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
