#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_size_label(n):
    # Robust mapping for your three buckets:
    #   low  ≈ 140, medium ≈ 1320, high ≈ 2800  (from your runners)
    if pd.isna(n):
        return "unknown"
    if n <= 200:
        return "low"
    if n <= 2000:
        return "medium"
    return "high"

def parse_args():
    ap = argparse.ArgumentParser(description="3x3 grid: accuracy vs class separation (transformers)")
    ap.add_argument("--csv", default="combined_results.csv", help="Path to combined_results.csv")
    ap.add_argument("--head-hidden", type=int, default=4, help="Filter: d_model (head_hidden)")
    ap.add_argument("--head-layers", type=int, default=2, help="Filter: num layers (head_layers)")
    ap.add_argument("--regime", default="erm", help="Filter: regime (erm/reweight/downsample/upsample)")
    ap.add_argument("--metric", default="acc_overall", choices=["acc_overall","acc_worst_group"],
                    help="Accuracy metric to plot on y-axis")
    ap.add_argument("--id_sigma_core", type=float, default=0.1, help="Filter: core sigma")
    # ap.add_argument("--d-noise", default="16,64,256", help="Comma list of d_noise values for columns")
    ap.add_argument("--d-noise", default="2,4,6", help="Comma list of d_noise values for columns")
    ap.add_argument("--title", default="Transformer: Accuracy vs Class Separation (mean ± std over seeds)")
    ap.add_argument("--outfile", default="transformer_classsep_grid_ID.png")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Filter to single backbone + regime
    base = df[
        (df["head_hidden"] == args.head_hidden) &
        (df["head_layers"] == args.head_layers) &
        (df["regime"] == args.regime) &
        (df["id_sigma_core"] == args.id_sigma_core) 
    ].copy()

    if base.empty:
        raise ValueError("No rows after filtering by head_hidden/head_layers/regime. "
                         f"Got head_hidden={args.head_hidden}, head_layers={args.head_layers}, regime={args.regime}")

    # Ensure we have the fields we need
    needed_cols = ["model","split","seed","d_noise","n_train_requested","id_core_scale", args.metric]
    for c in needed_cols:
        if c not in base.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # Add data size label
    base["data_size"] = base["n_train_requested"].map(data_size_label)

    # Normalize model names to our two buckets
    model_map = {
        "disc_transformer": "Discriminative",
        "gen_transformer_gauss": "Generative",
    }
    base = base[base["model"].isin(model_map.keys())].copy()
    base["Model"] = base["model"].map(model_map)

    # Keep only ID/OOD splits we care about
    # base = base[base["split"].isin(["ID","OOD"])].copy()
    base = base[base["split"] == "ID"].copy()
    # base = base[base["split"] == "OOD"].copy()
    # Columns (d_noise)
    dnoise_list = [int(x.strip()) for x in args.d_noise.split(",") if x.strip()]

    # Rows (data sizes)
    rows = ["low","medium","high"]

    # Aggregate: mean/std over seeds per (Model, split, d_noise, data_size, id_core_scale)
    # gb_cols = ["Model","split","d_noise","data_size","id_core_scale"]
    # agg = base.groupby(gb_cols)[args.metric].agg(["mean","std"]).reset_index()

    gb = ["Model","d_noise","data_size","id_core_scale"]
    # agg = base.groupby(gb)[["acc_worst_group","acc_best_group"]].agg(["mean","std"]).reset_index()
    agg = base.groupby(gb)[["acc_worst_group","acc_overall"]].agg(["mean","std"]).reset_index()


    # For consistent x-order per subplot (ascending class separation)
    # We'll build lookup dicts per (d_noise, data_size)
    # And plot 4 series (Disc-ID, Disc-OOD, Gen-ID, Gen-OOD)
    fig, axes = plt.subplots(len(rows), len(dnoise_list), figsize=(18, 12), sharey=True)
    if len(rows) == 1 and len(dnoise_list) == 1:
        axes = np.array([[axes]])

    # Styles
    # series_defs = [
    #     ("Discriminative","ID",  "tab:blue",  "Disc (ID)"),
    #     ("Discriminative","OOD", "lightskyblue", "Disc (OOD)"),
    #     ("Generative","ID",      "tab:red",   "Gen (ID)"),
    #     ("Generative","OOD",     "lightcoral","Gen (OOD)"),
    # ]

    # series_defs = [
    # ("Discriminative","acc_worst_group", "tab:blue",      "Disc (Worst)"),
    # ("Discriminative","acc_best_group",  "lightskyblue",  "Disc (Best)"),
    # ("Generative","acc_worst_group",     "tab:red",       "Gen (Worst)"),
    # ("Generative","acc_best_group",      "lightcoral",    "Gen (Best)"),
    # ]

    series_defs = [
    ("Discriminative","acc_worst_group", "tab:blue",      "Disc (Worst)"),
    ("Discriminative","acc_overall",  "lightskyblue",  "Disc (overall)"),
    ("Generative","acc_worst_group",     "tab:red",       "Gen (Worst)"),
    ("Generative","acc_overall",      "lightcoral",    "Gen (overall)"),
    ]


    # Collect 1 legend set from first subplot
    legend_handles = []
    built_legend = False

    for r, dsize in enumerate(rows):
        for c, dn in enumerate(dnoise_list):
            ax = axes[r][c]

            sub = agg[(agg["d_noise"] == dn) & (agg["data_size"] == dsize)]
            if sub.empty:
                ax.set_title(f"d_noise={dn} | {dsize} (no data)")
                ax.set_xlabel("class separation (id_core_scale)")
                ax.set_ylabel(args.metric)
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.2)
                continue

            # Unique sorted x
            x_vals = sorted(sub["id_core_scale"].unique())
            x_index = {v:i for i,v in enumerate(x_vals)}
            xticks = np.arange(len(x_vals))

            # Plot each series
            # for (mdl, spl, color, label) in series_defs:
            for mdl, metric_col, color, label in series_defs:
                # s = sub[(sub["Model"] == mdl) & (sub["split"] == spl)].sort_values("id_core_scale")
                s = sub[sub["Model"] == mdl].sort_values("id_core_scale")
                if s.empty:
                    continue
                # y_mean = s["mean"].to_numpy()
                # y_std  = s["std"].to_numpy()
                y_mean = s[(metric_col, "mean")].to_numpy()
                y_std  = s[(metric_col, "std")].to_numpy()
                xs = [x_index[v] for v in s["id_core_scale"].to_numpy()]
                line = ax.errorbar(xs, y_mean, yerr=y_std, marker="o", capsize=4, color=color, label=label, linewidth=1.6)
                if not built_legend and r == 0 and c == 0:
                    legend_handles.append(line)

            ax.set_title(f"d_noise={dn} | {dsize} data")
            ax.set_xlabel("class separation (id_core_scale)")
            if c == 0:
                ax.set_ylabel("Accuracy")
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(v) for v in x_vals])
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, which="both", ls="-", alpha=0.2)

        built_legend = True

    fig.suptitle(args.title + f"\n(head_hidden={args.head_hidden}, layers={args.head_layers}, regime={args.regime}, metric={args.metric})",
                 fontsize=14, y=0.98)
    # Single legend at top
    labels = [sd[3] for sd in series_defs]
    if legend_handles:
        fig.legend(legend_handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(args.outfile, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {args.outfile}")

if __name__ == "__main__":
    main()
