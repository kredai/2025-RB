
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import ttest_ind
from scipy.stats import norm

cwd = "C:/Nexus/Academic/#OneDrive [IISc]/Coursework/Projects/2025-04 [Ramray Bhat]/CC3D/analysis/cancore2/"
df_scan0 = pd.read_csv(cwd + "cancore2_scan-20/cancore2_scan-20_1200.csv")
df_scan0a = pd.read_csv(cwd + "cancore2_scan-20/cancore2_scan-20_0960.csv")
df_scan0b = pd.read_csv(cwd + "cancore2_scan-20/cancore2_scan-20_0720.csv")
df_scan0c = pd.read_csv(cwd + "cancore2_scan-20/cancore2_scan-20_0480.csv")
df_scan1 = pd.read_csv(cwd + "cancore2_scan-21/cancore2_scan-21_1200.csv")
df_scan2 = pd.read_csv(cwd + "cancore2_scan-22/cancore2_scan-22_1200.csv")
df_scan2a = pd.read_csv(cwd + "cancore2_scan-22/cancore2_scan-22_0960.csv")
df_scan2b = pd.read_csv(cwd + "cancore2_scan-22/cancore2_scan-22_0720.csv")
df_scan2c = pd.read_csv(cwd + "cancore2_scan-22/cancore2_scan-22_0480.csv")
df_scan3 = pd.read_csv(cwd + "cancore2_scan-23/cancore2_scan-23_1200.csv")
df_scan4 = pd.read_csv(cwd + "cancore2_scan-24/cancore2_scan-24_1200_adj.csv")
df_scan5 = pd.read_csv(cwd + "cancore2_scan-25/cancore2_scan-25_1200.csv")
df_scan6 = pd.read_csv(cwd + "cancore2_scan-26/cancore2_scan-26_1200.csv")
df_scan7 = pd.read_csv(cwd + "cancore2_scan-27/cancore2_scan-27_1200_adj.csv")
df_scan8 = pd.read_csv(cwd + "cancore2_scan-28/cancore2_scan-28_1200.csv")

# Read the CSV file
df_scan0["Condition"] = "SSOx = 0.0"
df_scan0a["Condition"] = "0.0 MCS 960"
df_scan0b["Condition"] = "0.0 MCS 720"
df_scan0c["Condition"] = "0.0 MCS 480"
df_scan1["Condition"] = "SSOx = 2.0"
df_scan2["Condition"] = "SSOx = 4.0"
df_scan2a["Condition"] = "4.0 MCS 960"
df_scan2b["Condition"] = "4.0 MCS 720"
df_scan2c["Condition"] = "4.0 MCS 480"
df_scan3["Condition"] = "SSOx = 8.0"
df_scan4["Condition"] = "SSOx = Max"
df_scan5["Condition"] = "0.05 ResHi"
df_scan6["Condition"] = "0.05 ResLo"
df_scan7["Condition"] = "ResLo Max"
df_scan8["Condition"] = "Spatial Position"
df = pd.concat([df_scan0, df_scan2, df_scan5, df_scan6], ignore_index=True)
df = df[(df["MajorAxis"] <= 480) & (df["Density"] >= 0.40) & (df["Convolution"] <= 6.8)] # eliminate outliers
features = ["CellGrowthModifier","CellECMContactEnergy","CellCellContactEnergy","ChemotaxisModifier","MMPSecretionModifier"]

selected_features = ["MajorAxis", "Density", "Convolution"]
cluster_colors = ["tab:green", "tab:blue", "tab:red", "tab:pink", "tab:olive", "tab:cyan"]
num_clusters = None  # Will be set per condition, but use the max for plotting
df["cluster"] = -1  # Initialize with invalid cluster
for condition in df["Condition"].unique():
    cond_mask = df["Condition"] == condition
    # For "Spatial Position", add "SSOx = 0.0" datapoints to clustering
    if condition == "Spatial Position":
        spatial_mask = (df["Condition"] == "Spatial Position") | (df["Condition"] == "SSOx = 0.0")
        X = df.loc[spatial_mask, selected_features].values
        # But only assign clusters to "Spatial Position" rows
    else:
        X = df.loc[cond_mask, selected_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled[:, 0] *= 1.00
    X_scaled[:, 1] *= 0.50 # Deemphasize density
    X_scaled[:, 2] *= 1.00
    max_k = 4; scores = []
    for k in range(2, max_k + 1):
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=0, n_init=20, assign_labels='cluster_qr')
        labels = spectral.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    best_k = scores.index(max(scores)) + 2
    if num_clusters is None or best_k > num_clusters:
        num_clusters = best_k  # Use the largest k for plotting
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=0, n_init=20, assign_labels='cluster_qr')
    labels = spectral.fit_predict(X_scaled)

    # Label clusters: cluster 1 has highest density, cluster 2 has highest convolution, cluster 3 has highest diameter
    temp_df = df.loc[cond_mask].copy()
    temp_df["tmp_cluster"] = labels
    density_means = temp_df.groupby("tmp_cluster")["Density"].mean()
    convolution_means = temp_df.groupby("tmp_cluster")["Convolution"].mean()
    diameter_means = temp_df.groupby("tmp_cluster")["MajorAxis"].mean()    
    cluster1 = density_means.idxmax()
    convolution_means_excl1 = convolution_means.drop(cluster1)
    cluster2 = convolution_means_excl1.idxmax()
    diameter_means_excl1_2 = diameter_means.drop([cluster1, cluster2])
    label_map = {cluster1: 0, cluster2: 1}
    if len(diameter_means_excl1_2) > 0:
        cluster3 = diameter_means_excl1_2.idxmax()
        label_map[cluster3] = 2
    mapped_labels = temp_df["tmp_cluster"].map(label_map)
    df.loc[cond_mask, "cluster"] = mapped_labels.values
    
# Assign clusters to MCS conditions based on identical state-switching threshold as their parent scan
if "StateSwitchingThreshold" in df.columns:
    # Identify SSOx and MCS conditions by string patterns
    ss_conditions = [cond for cond in df["Condition"].unique() if "MCS" not in str(cond)]
    mcs_conditions = [cond for cond in df["Condition"].unique() if "MCS" in str(cond)]
    # Build a fast lookup DataFrame for SSOx rows by (sst, replicate, iteration)
    ss_df = df[df["Condition"].isin(ss_conditions)].copy()
    ss_df["rep"] = ss_df.get("Replicate", None)
    ss_df["iter"] = ss_df.get("Iteration", None)
    ss_lookup = ss_df.set_index(
        ["StateSwitchingThreshold", "rep", "iter"]
    )["cluster"].to_dict()
    # Now assign clusters for MCS rows
    mcs_df = df[df["Condition"].isin(mcs_conditions)].copy()
    mcs_df["rep"] = mcs_df.get("Replicate", None)
    mcs_df["iter"] = mcs_df.get("Iteration", None)
    for idx, row in mcs_df.iterrows():
        key = (row["StateSwitchingThreshold"], row["rep"], row["iter"])
        if key in ss_lookup:
            df.at[idx, "cluster"] = ss_lookup[key]

z_scores_tables = []
cluster_counts = df["cluster"].value_counts().sort_index()
conditions = df["Condition"].unique().tolist()
for idx, condition in enumerate(conditions):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(7, 10))
    cond_df = df[df["Condition"] == condition]
    z_scores_table = []
    for cluster in range(num_clusters):
        cluster_points = cond_df[cond_df["cluster"] == cluster]
        color = cluster_colors[cluster % len(cluster_colors)]
        ax.scatter(cluster_points["Density"], cluster_points["Convolution"], cluster_points["MajorAxis"],
                   label=f"Cluster {cluster+1}", alpha=0.8, color=color)
        count = len(cluster_points)
        z_scores_row = [f"Cluster {cluster+1}", f"{count}"]
        for feat in features:
            cluster_mean = cluster_points[feat].mean()
            overall_mean = cond_df[feat].mean()
            overall_std = cond_df[feat].std()
            if overall_std != 0:
                z_score = (cluster_mean - overall_mean) / overall_std
            else: z_score = 0.0
            z_scores_row.append(f"{z_score:.2f}")
        z_scores_table.append(z_scores_row)
    z_scores_tables.append(z_scores_table)
    ax.set_xlim(0.40, 1.00)
    ax.set_ylim(1.0, 7.0)
    ax.set_zlim(100, 500)
    ax.set_xlabel("Density")
    ax.set_ylabel("Convolution")
    ax.set_zlabel("Major Axis")
    ax.view_init(elev=15, azim=225, roll=-2)
    ax.set_title(condition)
    ax.legend()
    col_labels = ["Cluster", "Count", "Gmod", "Eadh", "Cadh", "Ctax", "Msec"]
    table_data = [[str(i+1)] + row[1:] for i, row in enumerate(z_scores_table)]
    plt.subplots_adjust(bottom=0.40)
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="bottom", cellLoc="center", bbox=[0.0, -0.32, 1, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout(w_pad=-8.0,rect=[0, 0.12, 1, 0.92])
    plt.show()

for cluster in range(num_clusters):
    cluster_points = df[df["cluster"] == cluster]
    color = cluster_colors[cluster % len(cluster_colors)]
    ax.scatter(cluster_points["Density"], cluster_points["Convolution"], cluster_points["MajorAxis"],
                label=f"Cluster {cluster+1}", alpha=0.8, color=color)
    count = cluster_counts.get(cluster, 0)
    # Compute z-scores for each feature
    z_scores_row = [f"Cluster {cluster+1}", f"{count}"]
    for feat in features:
        cluster_mean = cluster_points[feat].mean()
        overall_mean = df[feat].mean()
        overall_std = df[feat].std()
        if overall_std != 0:
            z_score = (cluster_mean - overall_mean) / overall_std
        else: z_score = 0.0
        z_scores_row.append(f"{z_score:.2f}")
    z_scores_table.append(z_scores_row)

# Collect z-scores for all conditions and clusters
all_zscores = []
for condition in conditions:
    # Compute z-scores for each cluster
    for cluster in sorted(df["cluster"].unique()):
        cond_mask = (df["Condition"] == condition)
        df_cond_rep = df[cond_mask]
        cluster_points = df_cond_rep[df_cond_rep["cluster"] == cluster]
        count = len(cluster_points)
        z_scores_row = {
            "Condition": condition,
            "Cluster": cluster + 1,
            "Count": count
        }
        for feat, short in zip(features, ["Gmod", "Eadh", "Cadh", "Ctax", "Msec"]):
            # Compute means and stds within the current condition
            cluster_mean = cluster_points[feat].mean()
            overall_mean = df_cond_rep[feat].mean()
            overall_std = df_cond_rep[feat].std()
            if overall_std != 0:
                z_score = (cluster_mean - overall_mean) / overall_std
            else:
                z_score = 0.0
            z_scores_row[short] = z_score
        all_zscores.append(z_scores_row)
zscore_df = pd.DataFrame(all_zscores)
print(zscore_df)

# Run MANOVA for each cluster type across parameters
for cluster in sorted(df["cluster"].unique()):
    dep_vars = ["CellGrowthModifier", "CellECMContactEnergy", "CellCellContactEnergy", "ChemotaxisModifier", "MMPSecretionModifier"]
    cluster_data = df[df["cluster"] == cluster]
    cluster_data = cluster_data.dropna(subset=dep_vars + ["Condition"])
    # Check if all z-scores for this cluster are identical across conditions
    zscores = []
    for condition in cluster_data["Condition"].unique():
        cond_points = cluster_data[cluster_data["Condition"] == condition]
        z_row = []
        for feat in dep_vars:
            mean = cond_points[feat].mean()
            overall_mean = cluster_data[feat].mean()
            std = cluster_data[feat].std()
            z = 0.0 if std == 0 else (mean - overall_mean) / std
            z_row.append(round(z, 8))
        zscores.append(tuple(z_row))
    if len(set(zscores)) <= 1:
        print(f"\nMANOVA for Cluster {cluster+1}: Skipped (identical z-scores across conditions)")
        continue
    print(f"\nMANOVA for Cluster {cluster+1}:")
    manova = MANOVA.from_formula(" + ".join(dep_vars) + " ~ Condition", data=cluster_data)
    print(manova.mv_test())

# Run MANOVA for each cluster type across measurables
for cluster in sorted(df["cluster"].unique()):
    dep_vars = ["MajorAxis", "Density", "Convolution"]
    cluster_data = df[df["cluster"] == cluster]
    cluster_data = cluster_data.dropna(subset=dep_vars + ["Condition"])
    zscores = []
    for condition in cluster_data["Condition"].unique():
        cond_points = cluster_data[cluster_data["Condition"] == condition]
        z_row = []
        for feat in dep_vars:
            mean = cond_points[feat].mean()
            overall_mean = cluster_data[feat].mean()
            std = cluster_data[feat].std()
            z = 0.0 if std == 0 else (mean - overall_mean) / std
            z_row.append(round(z, 8))
        zscores.append(tuple(z_row))
    if len(set(zscores)) <= 1:
        print(f"\nMANOVA for Cluster {cluster+1}: Skipped (identical z-scores across conditions)")
        continue
    print(f"\nMANOVA for Cluster {cluster+1}:")
    manova = MANOVA.from_formula(" + ".join(dep_vars) + " ~ Condition", data=cluster_data)
    print(manova.mv_test())

# Stacked bar charts of parameter necessity for SSOx = 0.0
param_condition = "0.05 ResHi"
ssox0_df = df[df["Condition"] == param_condition].copy()
param_features = [
    ("CellGrowthModifier", "Cell Growth Modifier"),
    ("CellECMContactEnergy", "ECM Contact Energy"),
    ("CellCellContactEnergy", "Cell-Cell Contact Energy"),
    ("ChemotaxisModifier", "Chemotaxis Modifier"),
    ("MMPSecretionModifier", "MMP Secretion Modifier")
]
clusters = sorted(ssox0_df["cluster"].dropna().unique())
n_params = len(param_features)
fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 5), sharey=False)
if n_params == 1:
    axes = [axes]
for idx, (param, param_label) in enumerate(param_features):
    ax = axes[idx]
    # Prepare data for all clusters
    value_set = sorted(ssox0_df[param].dropna().unique())
    palette = sns.color_palette("RdPu", n_colors=len(value_set))
    # For each cluster, sort parameter values by their fraction (descending), but keep color mapping fixed
    bar_data = []
    bar_order = []
    for cluster in clusters:
        cluster_df = ssox0_df[ssox0_df["cluster"] == cluster]
        values, counts = np.unique(cluster_df[param], return_counts=True)
        count_dict = dict(zip(values, counts))
        total = counts.sum()
        fracs = [count_dict.get(v, 0) / total if total > 0 else 0 for v in value_set]
        # Sort values and fractions for this cluster by fraction descending
        sorted_pairs = sorted(zip(value_set, fracs), key=lambda x: x[1], reverse=True)
        sorted_values, sorted_fracs = zip(*sorted_pairs)
        bar_data.append(sorted_fracs)
        bar_order.append(sorted_values)
    bar_data = np.array(bar_data)  # shape: (n_clusters, n_values)
    # For each cluster, get the order of parameter values (by fraction descending)
    # For stacking, we need to plot the highest fraction at the bottom, but color by value_set index
    bottoms = np.zeros(len(clusters))
    for i in range(len(value_set)):
        # For each cluster, find which value is at this stack position
        heights = []
        colors = []
        labels = []
        for cidx, cluster in enumerate(clusters):
            val = bar_order[cidx][i]
            frac = bar_data[cidx, i]
            heights.append(frac)
            # Color by value_set index (consistent across clusters)
            color_idx = value_set.index(val)
            colors.append(palette[color_idx])
            labels.append(val)
        bars = ax.bar(
            [f"Cluster {int(c)+1}" for c in clusters],
            heights,
            bottom=bottoms,
            color=colors,
            edgecolor='white',
        )
        # Add value labels centered in each bar segment
        for bar, frac, val in zip(bars, heights, labels):
            if frac > 0.05:
                height = bar.get_height()
                y = bar.get_y() + height / 2
                if param in ["CellECMContactEnergy", "CellCellContactEnergy", "ChemotaxisModifier"]:
                    label_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                else:
                    label_str = f"{val:.2f}" if isinstance(val, float) else str(val)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    label_str,
                    ha='center', va='center', fontsize=8, color='black', rotation=0
                )
        bottoms += heights
    ax.set_title(param_label, fontsize=10)
    if idx == 0:
        ax.set_ylabel("Relative Frequency")
    else: ax.set_ylabel("")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
plt.suptitle(f"Parameter Necessity Across Clusters ({param_condition})", y=0.90)
plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], w_pad=2.0)
plt.show()

# Create heatmap of parameter sufficiency for SSOx = 0.0, for each cluster separately (one heatmap per cluster)
clusters = sorted(ssox0_df["cluster"].dropna().unique())
row_params = [
    ("CellGrowthModifier", "Rval"),
    ("CellECMContactEnergy", "Eadh"),
]
col_params = [
    ("CellCellContactEnergy", "Cadh"),
    ("ChemotaxisModifier", "Ctax"),
    ("MMPSecretionModifier", "Msec"),
]
row1_vals = sorted(ssox0_df[row_params[0][0]].unique())
row2_vals = sorted(ssox0_df[row_params[1][0]].unique())
col1_vals = sorted(ssox0_df[col_params[0][0]].unique())
col2_vals = sorted(ssox0_df[col_params[1][0]].unique())
col3_vals = sorted(ssox0_df[col_params[2][0]].unique())
row_combos = [(r1, r2) for r1 in row1_vals for r2 in row2_vals]  # 9
col_combos = [(c1, c2, c3) for c1 in col1_vals for c2 in col2_vals for c3 in col3_vals]  # 27
def stack_label(label, sep=","):
    return "\n".join(str(x) for x in label.split(sep))
for cluster in clusters:
    heatmap_data = np.full((len(row_combos), len(col_combos)), np.nan)
    for i, (r1, r2) in enumerate(row_combos):
        for j, (c1, c2, c3) in enumerate(col_combos):
            mask = (
                (ssox0_df[row_params[0][0]] == r1) &
                (ssox0_df[row_params[1][0]] == r2) &
                (ssox0_df[col_params[0][0]] == c1) &
                (ssox0_df[col_params[1][0]] == c2) &
                (ssox0_df[col_params[2][0]] == c3)
            )
            subset = ssox0_df[mask]
            if len(subset) > 0:
                # Sufficiency: fraction of all cells with this parameter combo that are in this cluster
                heatmap_data[i, j] = np.mean(subset["cluster"] == cluster)
    row_labels = [stack_label(f"{r1:.2f},{r2}") for (r1, r2) in row_combos]
    col_labels = [stack_label(f"{c1},{c2},{c3:.2f}") for (c1, c2, c3) in col_combos]
    fig, ax = plt.subplots(figsize=(20, 8))
    heatmap = sns.heatmap(
        heatmap_data,
        cmap="RdPu",
        vmin=0,
        vmax=1,
        annot=False,
        cbar=True,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        linewidths=0.2,
        square=False,
        cbar_kws={"pad": 0.02}
    )
    ax.set_xlabel("Cell-Cell Contact Energy, Chemotaxis Modifier, MMP Secretion Modifier")
    ax.set_ylabel("Cell Growth Modifier and Cell-ECM Contact Energy")
    ax.set_title(f"Parameter Sufficiency for Cluster {int(cluster)+1} ({param_condition})", y=1.02)
    plt.xticks(rotation=0, fontsize=8, ha='center')
    plt.yticks(rotation=90, fontsize=8, ha='center')
    ax.tick_params(axis='y', pad=10)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.subplots_adjust(right=1.04, left=0.08, top=0.88, bottom=0.16)
    plt.show()

# Rose plots (polar histograms) of rotation for each cluster within each condition
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    # Create a figure with rose plots and one normal subplot for eccentricity
    fig = plt.figure(figsize=(4*(num_clusters+1), 5))
    axes = []
    # Rose plots for each cluster (polar)
    for cluster in range(num_clusters):
        ax = fig.add_subplot(1, num_clusters + 1, cluster + 1, projection='polar')
        axes.append(ax)
        cluster_df = cond_df[cond_df["cluster"] == cluster]
        if "Rotation" not in cluster_df.columns or cluster_df["Rotation"].dropna().empty:
            ax.set_title(f"Cluster {cluster+1}\n(No Rotation Data)")
            continue
        align_deg = cluster_df["Rotation"].dropna() % 360
        extended_deg = pd.concat([align_deg, (align_deg + 180) % 360])
        angles_rad = np.deg2rad(extended_deg)
        n_bins = 24
        counts, bins = np.histogram(angles_rad, bins=n_bins, range=(0, 2 * np.pi))
        total = counts.sum()
        if total > 0:
            fractions = counts / total
        else: fractions = counts
        widths = np.diff(bins)
        # Add light grey fill from 22.5° to 67.5° and 202.5° to 247.5°
        for centre in [45, 225]:
            ax.bar(
            [np.deg2rad(centre)], [1.0], width=[np.pi/4], bottom=0.0,
            color="tab:olive", alpha=0.5, edgecolor=None, linewidth=0, zorder=0
            )
        ax.bar(bins[:-1], fractions, width=widths, bottom=0.0, color=cluster_colors[cluster % len(cluster_colors)], alpha=0.8, edgecolor='k')
        # Set max histogram axis height per cluster
        if cluster == 0: ax.set_ylim(0, 0.07)
        if cluster == 1: ax.set_ylim(0, 0.14)
        if cluster == 2: ax.set_ylim(0, 0.10)
        for angle in [np.pi/4, 5*np.pi/4]:
            ax.plot([angle, angle], [0.0, 1.0], color='k', linewidth=4, solid_capstyle='round', zorder=10)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(f"Cluster {cluster+1}", fontsize=10)
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels([f"{int(np.rad2deg(tick))}°" for tick in np.linspace(0, 2*np.pi, 8, endpoint=False)])
        ax.set_rlabel_position(285)

    # Eccentricity boxplot in the last column (cartesian, not polar)
    ax2 = fig.add_subplot(1, num_clusters + 1, num_clusters + 1)
    axes.append(ax2)
    ecc_data = []
    for cluster in range(num_clusters):
        cluster_df = cond_df[cond_df["cluster"] == cluster]
        if "Eccentricity" in cluster_df.columns:
            for val in cluster_df["Eccentricity"].dropna():
                ecc_data.append({"Cluster": f"Cluster {cluster+1}", "Eccentricity": val})
    ecc_df = pd.DataFrame(ecc_data)
    if not ecc_df.empty:
        sns.boxplot(
            x="Cluster", y="Eccentricity", hue="Cluster", data=ecc_df, ax=ax2,
            order=[f"Cluster {i+1}" for i in range(num_clusters)],
            boxprops=dict(alpha=0.8), showfliers=False, legend=False,
            palette=[cluster_colors[i % len(cluster_colors)] for i in range(num_clusters)]
        )
        means = ecc_df.groupby("Cluster")["Eccentricity"].mean()
        for i in range(num_clusters):
            cname = f"Cluster {i+1}"
            if cname in means:
                ax2.hlines(means[cname], i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax2.set_title(f"Eccentricity", fontsize=10)
        ax2.set_xlabel("")
        ax2.set_ylabel("    ", fontsize=18)
        ax2.set_ylim(0, 1.0)
        # Pairwise t-tests for significance annotation
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 0.88; y_offset = 0.12
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax2.set_ylim(bottom=0, top=max_y + y_offset/4)
        for k, (idx1, idx2) in enumerate(pairs):
            cname1 = f"Cluster {idx1+1}"
            cname2 = f"Cluster {idx2+1}"
            group1 = ecc_df[ecc_df["Cluster"] == cname1]["Eccentricity"]
            group2 = ecc_df[ecc_df["Cluster"] == cname2]["Eccentricity"]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.01:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.05:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.10:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Rose Plots for Alignment ({condition})", y=0.92)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=-0.5)
    plt.show()
    
# Plot diameter comparisons for SSOx = 2.0, 4.0, 8.0 vs 0.0 and Max
# Add df_scan4 (SSOx = Max) to the main dataframe for this plot
df_plot = pd.concat([df, df_scan4], ignore_index=True)
ss_conditions = [(cond, cond) for cond in conditions if cond not in ["SSOx = 0.0", "SSOx = Max"]]
fig, axes = plt.subplots(1, len(ss_conditions), figsize=(4*len(ss_conditions), 5), sharey=True)
if len(ss_conditions) == 1:
    axes = [axes]
for i, (mid_cond, mid_label) in enumerate(ss_conditions):
    ax = axes[i]
    plot_conds = ["SSOx = 0.0", mid_cond, "SSOx = Max"]
    plot_labels = ["Epithelial", "Mixed", "Mesenchymal"]
    plot_df = df_plot[df_plot["Condition"].isin(plot_conds)].copy()
    plot_df["Condition"] = plot_df["Condition"].map({
        "SSOx = 0.0": "Epithelial",
        mid_cond: "Mixed",
        "SSOx = Max": "Mesenchymal"
    })
    sns.boxplot(
        x="Condition", y="MajorAxis", data=plot_df, ax=ax,
        order=plot_labels,
        boxprops=dict(alpha=0.8), showfliers=False,
    )
    means = plot_df.groupby("Condition")["MajorAxis"].mean()
    for j, label in enumerate(plot_labels):
        if label in means:
            ax.hlines(means[label], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
    ax.set_title(f"{mid_label}", fontsize=10)
    ax.set_xlabel("Phenotype")
    if i == 0:
        ax.set_ylabel("Major Axis")
    else: ax.set_ylabel("")
    # Pairwise t-tests for significance annotation
    pairs = [(0,1), (1,2), (0,2)]
    y_max = 450; y_offset = 80
    max_y = y_max + y_offset * (len(pairs) + 1.25)
    ax.set_ylim(bottom=0, top=max_y + y_offset/4)
    # Map plot_conds to plot_labels for correct annotation positions
    cond_to_label = {
        "SSOx = 0.0": "Epithelial",
        mid_cond: "Mixed",
        "SSOx = Max": "Mesenchymal"
    }
    label_to_pos = {label: idx for idx, label in enumerate(plot_labels)}
    for k, (idx1, idx2) in enumerate(pairs):
        val1, val2 = plot_conds[idx1], plot_conds[idx2]
        label1, label2 = cond_to_label[val1], cond_to_label[val2]
        group1 = plot_df[plot_df["Condition"] == label1]["MajorAxis"]
        group2 = plot_df[plot_df["Condition"] == label2]["MajorAxis"]
        if group1.empty or group2.empty:
            continue
        t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
        x1, x2 = label_to_pos[label1], label_to_pos[label2]
        y = y_max + y_offset * (k + 1)
        if pair_pval < 0.001:
            ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
            ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
        elif pair_pval < 0.01:
            ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
            ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
        elif pair_pval < 0.05:
            ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
            ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
        elif pair_pval < 0.10:
            ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
            ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
plt.suptitle("Major Axis Across Homotypic and Heterotypic Populations", y=0.90)
plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
plt.show()

# Generate plots comparing diameter, density, and convolution across conditions for each cluster
measurables = [
    ("MajorAxis", "Major Axis"),
    ("Density", "Density"),
    ("Convolution", "Convolution")
]
if any("MCS" in str(cond) for cond in conditions):
    title = "Monte Carlo Steps"
else: title = "State-Switching Thresholds"
cluster_values = sorted(df["cluster"].dropna().unique())
n_cols = len(cluster_values)
for i, (meas, meas_label) in enumerate(measurables):
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6), sharey=False)
    for col_idx, cluster in enumerate(cluster_values):
        cluster_df = df[df["cluster"] == cluster]
        ax = axes[col_idx]
        cluster_df = cluster_df.copy()
        cluster_df["ConditionShort"] = cluster_df["Condition"].astype(str).str[-3:]
        short_cond = [str(cond)[-3:] for cond in conditions]
        sns.boxplot(
            x="ConditionShort", y=meas, data=cluster_df, ax=ax, 
            order=short_cond,
            boxprops=dict(alpha=0.8), showfliers=False,
        )
        means = cluster_df.groupby("Condition")[meas].mean()
        for i, condition in enumerate(conditions):
            if condition in means:
                ax.hlines(means[condition], i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Cluster {int(cluster)+1}", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(meas_label)
        else: ax.set_ylabel("")
        ax.set_xlabel("")
        pairs = [(1,2), (2,3), (1,3), (0,1), (0,2), (0,3)]
        # t-test between conditions for this cluster and measurable
        if meas == "MajorAxis":
            y_max = 500; y_offset = 60
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=100, top=max_y + y_offset/4)
        if meas == "Density":
            y_max = 0.90; y_offset = 0.08
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=0.4, top=max_y + y_offset/4)
        if meas == "Convolution":
            y_max = 5.4; y_offset = 0.75
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=1.0, top=max_y + y_offset/4)
        # Annotate significant differences
        for i, (idx1, idx2) in enumerate(pairs):
            e1, e2 = conditions[idx1], conditions[idx2]
            group1 = cluster_df[cluster_df["Condition"] == e1][meas]
            group2 = cluster_df[cluster_df["Condition"] == e2][meas]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            y = y_max + y_offset * (i + 1)
            if pair_pval < 0.001:
                ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((idx1 + idx2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.01:
                ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((idx1 + idx2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.05:
                ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((idx1 + idx2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.10:
                ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((idx1 + idx2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"{meas_label} Across {title}", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Plot density profile versus mean diameter for all cluster types side by side for each condition
if "DensityProfile" in df.columns:
    shell_width = 20  # Each shell is 20 microns wide
    shell_centers = np.arange(shell_width/2, shell_width*25, shell_width)
    # Set color for each condition in order
    condition_palette = ["tab:blue", "tab:purple", "tab:pink", "tab:red"]
    condition_colors = {cond: condition_palette[i % len(condition_palette)] for i, cond in enumerate(conditions)}
    line_styles = {}
    # Pre-parse all valid DensityProfiles for speed
    parsed_profiles = {}
    for condition in conditions:
        cond_df = df[df["Condition"] == condition]
        for cluster in range(num_clusters):
            cluster_df = cond_df[cond_df["cluster"] == cluster]
            # Drop rows with missing or malformed DensityProfile
            cluster_df = cluster_df[cluster_df["DensityProfile"].notnull()]
            profiles = []
            for prof in cluster_df["DensityProfile"]:
                if isinstance(prof, str):
                    try:
                        arr = np.array([float(x) for x in prof.split(":")])
                        if arr.size == 25:
                            profiles.append(arr)
                    except Exception:
                        continue
                elif isinstance(prof, (list, np.ndarray)) and len(prof) == 25:
                    profiles.append(np.array(prof))
            if profiles:
                parsed_profiles[(condition, cluster)] = np.array(profiles)
    fig, axes = plt.subplots(1, num_clusters, figsize=(4* num_clusters, 5), sharey=False)
    for cluster in range(num_clusters):
        ax = axes[cluster] if num_clusters > 1 else axes
        for i, condition in enumerate(conditions):
            profiles = parsed_profiles.get((condition, cluster), None)
            if profiles is None or len(profiles) == 0:
                continue
            mean_profile = profiles.mean(axis=0)
            std_profile = profiles.std(axis=0)
            style = line_styles.get(condition, {"linestyle": "-", "linewidth": 2, "alpha": 0.8})
            color = condition_colors.get(condition, None)
            ax.plot(
                shell_centers, mean_profile,
                label=condition,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                color=color
            )
            ax.fill_between(
                shell_centers, mean_profile-std_profile, mean_profile+std_profile,
                alpha=0.10, color=color
            )
        ax.set_xlabel("Mean Diameter")
        if cluster == 0:
            ax.set_ylabel("Density")
        else: ax.set_ylabel("")
        ax.set_title(f"Cluster {cluster+1}", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 500)
        ax.legend(fontsize=8, loc='upper right')
    plt.suptitle(f"Density Across Mean Diameter", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92])
    plt.show()

# Plot density by section for each cluster type across conditions
if any("MCS" in str(cond) for cond in conditions):
    title = "Monte Carlo Steps"
else: title = "State-Switching Thresholds"
section_names = ["CentreDensity", "MarginDensity"]
section_labels = ["Centre", "Margin"]
clusters = sorted(df["cluster"].dropna().unique())
n_clusters = len(clusters)
for row_idx, (sec, label) in enumerate(zip(section_names, section_labels)):
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 6), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    for col_idx, cluster in enumerate(clusters):
        cluster_df = df[df["cluster"] == cluster]
        ax = axes[col_idx]
        cluster_df = cluster_df.copy()
        cluster_df["ConditionShort"] = cluster_df["Condition"].astype(str).str[-3:]
        short_cond = [str(cond)[-3:] for cond in df["Condition"].unique()]
        sns.boxplot(
            x="ConditionShort", y=sec, data=cluster_df, ax=ax, 
            order=short_cond,
            boxprops=dict(alpha=0.8), showfliers=False,
        )
        means = cluster_df.groupby("Condition")[sec].mean()
        for j, cond in enumerate(df["Condition"].unique()):
            if cond in means:
                ax.hlines(means[cond], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Cluster {int(cluster)+1}", fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Density")
        else: ax.set_ylabel("")
        ax.set_ylim(0, 1.0)
        # Annotate significance between all pairs
        pairs = [(1,2), (2,3), (1,3), (0,1), (0,2), (0,3)]
        y_max = 1.0; y_offset = 0.16
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=0, top=max_y + y_offset/4)
        for k, (idx1, idx2) in enumerate(pairs):
            conds = list(df["Condition"].unique())
            if idx1 >= len(conds) or idx2 >= len(conds):
                continue
            cond1, cond2 = conds[idx1], conds[idx2]
            group1 = cluster_df[cluster_df["Condition"] == cond1][sec].dropna()
            group2 = cluster_df[cluster_df["Condition"] == cond2][sec].dropna()
            if len(group1) > 1 and len(group2) > 1:
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                x1, x2 = idx1, idx2
                y = y_max + y_offset * (k + 1)
                if pair_pval < 0.001:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"{label} Density Across {title}", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Plot centre and margin density for clusters in two plots side-by-side, sequentially for conditions
section_names = ["CentreDensity", "MarginDensity"]
section_labels = ["Centre Density", "Margin Density"]
clusters = sorted(df["cluster"].dropna().unique())
n_clusters = len(clusters)
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    fig, axes = plt.subplots(1, 2, figsize=(4*len(section_names), 5), sharey=False)
    for i, (sec, label) in enumerate(zip(section_names, section_labels)):
        ax = axes[i]
        plot_data = []
        for cluster in clusters:
            cluster_df = cond_df[cond_df["cluster"] == cluster]
            plot_data.extend([{"Cluster": f"Cluster {int(cluster)+1}", label: v} for v in cluster_df[sec].dropna()])
        plot_df = pd.DataFrame(plot_data)
        if not plot_df.empty:
            sns.boxplot(
                x="Cluster", y=label, hue="Cluster", data=plot_df, ax=ax,
                order=[f"Cluster {int(c)+1}" for c in clusters],
                boxprops=dict(alpha=0.8), showfliers=False, legend=False,
                palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters]
            )
            means = plot_df.groupby("Cluster")[label].mean()
            for j, cname in enumerate([f"Cluster {int(c)+1}" for c in clusters]):
                if cname in means:
                    ax.hlines(means[cname], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
            # Annotate significance between clusters
            pairs = [(0, 1), (1, 2), (0, 2)]
            y_max = 1.0; y_offset = 0.125
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=0.2, top=max_y + y_offset/4)
            for k, (idx1, idx2) in enumerate(pairs):
                cname1 = f"Cluster {idx1+1}"
                cname2 = f"Cluster {idx2+1}"
                group1 = plot_df[plot_df["Cluster"] == cname1][label]
                group2 = plot_df[plot_df["Cluster"] == cname2][label]
                if group1.empty or group2.empty:
                    continue
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                x1, x2 = idx1, idx2
                y = y_max + y_offset * (k + 1)
                if pair_pval < 0.001:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=14)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=14)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=14)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=14)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Density")
        else: ax.set_ylabel("")
    plt.suptitle(f"Centre and Margin Density by Cluster ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], w_pad=2.0)
    plt.show()

# Plot centre and margin lineage distributions across mean diameter for spatial position
if "HypoxicDensityProfile" in df.columns and "DensityProfile" in df.columns:
    spatial_condition = "Spatial Position"
    cond_df = df[df["Condition"] == spatial_condition]
    if not cond_df.empty:
        shell_width = 20
        shell_centers = np.arange(shell_width/2, shell_width*25, shell_width)
        clusters = sorted(cond_df["cluster"].dropna().unique())
        num_clusters = len(clusters)
        fig, axes = plt.subplots(1, num_clusters, figsize=(4*num_clusters, 5), sharey=True)
        if num_clusters == 1:
            axes = [axes]
        for i, cluster in enumerate(clusters):
            ax = axes[i]
            cluster_df = cond_df[cond_df["cluster"] == cluster]
            margin_profiles = []
            centre_profiles = []
            for idx, row in cluster_df.iterrows():
                hyp_prof = row.get("HypoxicDensityProfile", None)
                dens_prof = row.get("DensityProfile", None)
                hyp_arr, dens_arr = None, None
                if isinstance(hyp_prof, str):
                    try:
                        hyp_arr = np.array([float(x) for x in hyp_prof.split(":")])
                    except Exception:
                        continue
                elif isinstance(hyp_prof, (list, np.ndarray)) and len(hyp_prof) == 25:
                    hyp_arr = np.array(hyp_prof)
                if isinstance(dens_prof, str):
                    try:
                        dens_arr = np.array([float(x) for x in dens_prof.split(":")])
                    except Exception:
                        continue
                elif isinstance(dens_prof, (list, np.ndarray)) and len(dens_prof) == 25:
                    dens_arr = np.array(dens_prof)
                if hyp_arr is not None and dens_arr is not None and len(hyp_arr) == 25 and len(dens_arr) == 25:
                    margin_arr = np.maximum(0, dens_arr - hyp_arr)
                    centre_profiles.append(hyp_arr)
                    margin_profiles.append(margin_arr)
            # Plot Margin Lineage (DensityProfile - HypoxicDensityProfile)
            if margin_profiles:
                margin_profiles = np.array(margin_profiles)
                mean_margin = margin_profiles.mean(axis=0)
                std_margin = margin_profiles.std(axis=0)
                ax.plot(shell_centers, mean_margin, label="Margin Lineage", color="tab:blue", linewidth=2, alpha=0.8)
                ax.fill_between(shell_centers, mean_margin - std_margin, mean_margin + std_margin, alpha=0.15, color="tab:blue")
            # Plot Centre Lineage (HypoxicDensityProfile)
            if centre_profiles:
                centre_profiles = np.array(centre_profiles)
                mean_centre = centre_profiles.mean(axis=0)
                std_centre = centre_profiles.std(axis=0)
                ax.plot(shell_centers, mean_centre, label="Centre Lineage", color="tab:purple", linewidth=2, alpha=0.8)
                ax.fill_between(shell_centers, mean_centre - std_centre, mean_centre + std_centre, alpha=0.15, color="tab:purple")
            ax.set_xlabel("Mean Diameter")
            if i == 0:
                ax.set_ylabel("Lineage Density")
            else: ax.set_ylabel("")
            ax.set_title(f"Cluster {int(cluster)+1}", fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 500)
            ax.legend(fontsize=9, loc='upper right')
        plt.suptitle("Margin and Centre Lineage Distribution Across Mean Diameter", y=0.90)
        plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92])
        plt.show()

# Plot hypoxic fraction across diameter for clusters 1+3 and cluster 2
for condition in conditions:
    # Define cluster filters and titles
    cluster_filters = [
        (None, "All Clusters"),
        ([0, 2], "Clusters 1+3"),
        ([1], "Cluster 2")
    ]
    # Skip conditions without hypoxia data
    if condition in ["SSOx = 0.0", "SSOx = Max"]:
        continue
    cond_df = df[df["Condition"] == condition]
    fig, axes = plt.subplots(1, len(cluster_filters) + 1, figsize=(4*(len(cluster_filters)+1), 5), sharey=False)
    # Pre-parse NormoxicDensityProfile for speed
    normoxic_widths = []
    for idx, row in cond_df.iterrows():
        dens_prof = row.get("DensityProfile", None)
        hyp_prof = row.get("HypoxicDensityProfile", None)
        norm_arr = None
        if isinstance(dens_prof, str) and isinstance(hyp_prof, str):
            try:
                dens_arr = np.array([float(x) for x in dens_prof.split(":")])
                hyp_arr = np.array([float(x) for x in hyp_prof.split(":")])
                if dens_arr.size == 25 and hyp_arr.size == 25:
                    norm_arr = np.maximum(0, dens_arr - hyp_arr)
            except Exception:
                norm_arr = None
        elif isinstance(dens_prof, (list, np.ndarray)) and isinstance(hyp_prof, (list, np.ndarray)):
            if len(dens_prof) == 25 and len(hyp_prof) == 25:
                norm_arr = np.maximum(0, np.array(dens_prof) - np.array(hyp_prof))
        if norm_arr is not None:
            # Find contiguous bins where density > 0.05
            mask = norm_arr > 0.05
            # Find the longest contiguous True segment
            max_width = 0
            current = 0
            for val in mask:
                if val:
                    current += 1
                    if current > max_width:
                        max_width = current
                else: current = 0
            width = 20 * max_width
            normoxic_widths.append({"index": idx, "cluster": row.get("cluster", np.nan), "NormoxicWidth": width})
        else:
            normoxic_widths.append({"index": idx, "cluster": row.get("cluster", np.nan), "NormoxicWidth": np.nan})
    normoxic_widths_df = pd.DataFrame(normoxic_widths).set_index("index")
    cond_df = cond_df.copy()
    df.loc[normoxic_widths_df.index, "NormoxicWidth"] = normoxic_widths_df["NormoxicWidth"]
    cond_df["NormoxicWidth"] = normoxic_widths_df["NormoxicWidth"]  
    for row_idx, (cluster_sel, title) in enumerate(cluster_filters):
        ax = axes[row_idx] if len(cluster_filters) > 1 else axes
        # Filter data for selected clusters
        if cluster_sel is None:
            df_sub = cond_df.dropna(subset=["MajorAxis", "HypoxicFraction"])
        else:
            df_sub = cond_df[cond_df["cluster"].isin(cluster_sel)].dropna(subset=["MajorAxis", "HypoxicFraction"])
        if df_sub.empty:
            ax.set_title(f"{title}\n(No Data)")
            continue
        df_sub = df_sub.copy()
        clusters_to_plot = cluster_sel if cluster_sel is not None else sorted(cond_df["cluster"].dropna().unique())
        for cluster in clusters_to_plot:
            cdf = df_sub[df_sub["cluster"] == cluster]
            if not cdf.empty:
                ax.scatter(
                    cdf["MajorAxis"], cdf["HypoxicFraction"],
                    color=cluster_colors[cluster % len(cluster_colors)],
                    alpha=0.8, label=f"Cluster {cluster+1}"
                )
        # Fit a polynomial (degree 3)  using 95th percentile min and max for x_fit
        if len(df_sub) > 1:
            x, y = df_sub["MajorAxis"].values, df_sub["HypoxicFraction"].values
            coeffs = np.polyfit(x, y, 3)
            poly = np.poly1d(coeffs)
            x_min = np.percentile(x, 2.5)
            x_max = np.percentile(x, 97.5)
            x_fit = np.linspace(x_min, x_max, 100)
            ax.plot(x_fit, poly(x_fit), color="k", linestyle="--", linewidth=2, label="Poly Trendline")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Major Axis")
        ax.set_xlim(100, 500)
        if row_idx == 0:
            ax.set_ylabel("Hypoxic Fraction")
        else: ax.set_ylabel("")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc='upper right')
    # Fourth plot: Normoxic width boxplot across clusters
    ax = axes[-1]
    plot_df = cond_df.dropna(subset=["NormoxicWidth", "cluster"])
    clusters = sorted(plot_df["cluster"].dropna().unique())
    if not plot_df.empty and len(clusters) > 0:
        sns.boxplot(
            x="cluster", y="NormoxicWidth", hue="cluster", data=plot_df, ax=ax,
            order=clusters,
            boxprops=dict(alpha=0.8), showfliers=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters],
            legend=False
        )
        means = plot_df.groupby("cluster")["NormoxicWidth"].mean()
        for i, cluster in enumerate(clusters):
            if cluster in means:
                ax.hlines(means[cluster], i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title("Normoxic Width Across Clusters", fontsize=10)
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"Cluster {int(c)+1}" for c in clusters])
        ax.set_ylabel("    ", fontsize=18)
        ax.set_xlabel("")
        # Annotate significance between clusters
        pairs = [(0, 1), (1, 2), (0, 2)]
        y_max = 240; y_offset = 40
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=0, top=max_y + y_offset/4)
        for k, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(clusters) or idx2 >= len(clusters):
                continue
            cname1 = clusters[idx1]
            cname2 = clusters[idx2]
            group1 = plot_df[plot_df["cluster"] == cname1]["NormoxicWidth"]
            group2 = plot_df[plot_df["cluster"] == cname2]["NormoxicWidth"]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.01:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.05:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.10:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Hypoxic Fraction Across Achieved Major Axis ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=-1.0)
    plt.show()

# Compare hypoxic fraction of clusters across diameter bins for each condition
bins = [160, 240, 280, 320, 400]
bin_labels = ["160-240", "240-280", "280-320", "320-400"]
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    if "StateSwitchingThreshold" in cond_df.columns:
        if (cond_df["StateSwitchingThreshold"] == 0.0).all():
            continue
    # Only keep rows with valid MajorAxis and HypoxicFraction
    cond_df = cond_df.dropna(subset=["MajorAxis", "HypoxicFraction", "cluster"])
    cond_df = cond_df[(cond_df["MajorAxis"] >= 100) & (cond_df["MajorAxis"] <= 500)]
    cond_df = cond_df.copy()
    cond_df["MajorAxisBin"] = pd.cut(cond_df["MajorAxis"], bins=bins, labels=bin_labels, include_lowest=True, right=False)
    n_cols = len(bin_labels)
    clusters = sorted(cond_df["cluster"].dropna().unique())
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5), sharey=False)
    for col_idx, bin_label in enumerate(bin_labels):
        ax = axes[col_idx] if n_cols > 1 else axes
        bin_df = cond_df[cond_df["MajorAxisBin"] == bin_label]
        plot_data = []
        for cluster in clusters:
            cluster_df = bin_df[bin_df["cluster"] == cluster]
            if len(cluster_df) < 5:
                plot_data.append({"cluster": f"Cluster {int(cluster)+1}", "HypoxicFraction": -1})
            else:
                for _, row in cluster_df.iterrows():
                    plot_data.append({"cluster": f"Cluster {int(cluster)+1}", "HypoxicFraction": row["HypoxicFraction"]})
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(
            x="cluster", y="HypoxicFraction", hue="cluster", data=plot_df, ax=ax,
            order=[f"Cluster {int(c)+1}" for c in clusters],
            boxprops=dict(alpha=0.8), showfliers=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters], legend=False
        )
        for i, cname in enumerate([f"Cluster {int(c)+1}" for c in clusters]):
            group_mean = plot_df[plot_df["cluster"] == cname]["HypoxicFraction"].mean()
            ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Major Axis {bin_label}", fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Hypoxic Fraction")
        else: ax.set_ylabel("")
        # t-test between clusters
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 0.80; y_offset = 0.12
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=0.00, top=max_y + y_offset/4)
        # Annotate significant differences
        for k, (idx1, idx2) in enumerate(pairs):
            cname1 = f"Cluster {int(clusters[idx1])+1}"
            cname2 = f"Cluster {int(clusters[idx2])+1}"
            group1 = plot_df[plot_df["cluster"] == cname1]["HypoxicFraction"]
            group2 = plot_df[plot_df["cluster"] == cname2]["HypoxicFraction"]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.01:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.05:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.10:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Hypoxic Fraction Across Clusters ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Box plots of Normoxic Width across clusters for each condition
bins = [160, 240, 280, 320, 400]
bin_labels = ["160-240", "240-280", "280-320", "320-400"]
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    if "StateSwitchingThreshold" in cond_df.columns:
        if (cond_df["StateSwitchingThreshold"] == 0.0).all():
            continue
    # Only keep rows with valid MajorAxis, NormoxicWidth, and cluster
    cond_df = cond_df.dropna(subset=["MajorAxis", "NormoxicWidth", "cluster"])
    cond_df = cond_df[(cond_df["MajorAxis"] >= 100) & (cond_df["MajorAxis"] <= 500)]
    cond_df = cond_df.copy()
    cond_df["MajorAxisBin"] = pd.cut(cond_df["MajorAxis"], bins=bins, labels=bin_labels, include_lowest=True, right=False)
    n_cols = len(bin_labels)
    clusters = sorted(cond_df["cluster"].dropna().unique())
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5), sharey=False)
    for col_idx, bin_label in enumerate(bin_labels):
        ax = axes[col_idx] if n_cols > 1 else axes
        bin_df = cond_df[cond_df["MajorAxisBin"] == bin_label]
        plot_data = []
        for cluster in clusters:
            cluster_df = bin_df[bin_df["cluster"] == cluster]
            if len(cluster_df) < 5:
                plot_data.append({"cluster": f"Cluster {int(cluster)+1}", "NormoxicWidth": -1})
            else:
                for _, row in cluster_df.iterrows():
                    plot_data.append({"cluster": f"Cluster {int(cluster)+1}", "NormoxicWidth": row["NormoxicWidth"]})
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(
            x="cluster", y="NormoxicWidth", hue="cluster", data=plot_df, ax=ax,
            order=[f"Cluster {int(c)+1}" for c in clusters],
            boxprops=dict(alpha=0.8), showfliers=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters], legend=False
        )
        for i, cname in enumerate([f"Cluster {int(c)+1}" for c in clusters]):
            group_mean = plot_df[plot_df["cluster"] == cname]["NormoxicWidth"].mean()
            ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Major Axis {bin_label}", fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Normoxic Width")
        else: ax.set_ylabel("")
        # t-test between clusters
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 240; y_offset = 40
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=0.00, top=max_y + y_offset/4)
        # Annotate significant differences
        for k, (idx1, idx2) in enumerate(pairs):
            cname1 = f"Cluster {int(clusters[idx1])+1}"
            cname2 = f"Cluster {int(clusters[idx2])+1}"
            group1 = plot_df[plot_df["cluster"] == cname1]["NormoxicWidth"].dropna()
            group2 = plot_df[plot_df["cluster"] == cname2]["NormoxicWidth"].dropna()
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.01:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.05:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
            elif pair_pval < 0.10:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Normoxic Width Across Clusters ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()
    
# Boxplots of Mixing Width across conditions for each cluster
# MixingWidth = 20 * number of contiguous bins where both DensityProfile and HypoxicDensityProfile > 0.20
if "DensityProfile" in df.columns and "HypoxicDensityProfile" in df.columns:
    # Only use conditions where StateSwitchingThreshold is not zero (if present)
    valid_conditions = []
    for condition in conditions:
        cond_df = df[df["Condition"] == condition]
        if "StateSwitchingThreshold" in cond_df.columns:
            if (cond_df["StateSwitchingThreshold"] == 0.0).all():
                continue
        valid_conditions.append(condition)
    # Precompute MixingWidth for all rows in valid conditions
    mixing_widths = []
    for idx, row in df[df["Condition"].isin(valid_conditions)].iterrows():
        dens_prof = row.get("DensityProfile", None)
        hyp_prof = row.get("HypoxicDensityProfile", None)
        dens_arr, hyp_arr = None, None
        if isinstance(dens_prof, str):
            try:
                dens_arr = np.array([float(x) for x in dens_prof.split(":")])
            except Exception:
                continue
        elif isinstance(dens_prof, (list, np.ndarray)) and len(dens_prof) == 25:
            dens_arr = np.array(dens_prof)
        if isinstance(hyp_prof, str):
            try:
                hyp_arr = np.array([float(x) for x in hyp_prof.split(":")])
            except Exception:
                continue
        elif isinstance(hyp_prof, (list, np.ndarray)) and len(hyp_prof) == 25:
            hyp_arr = np.array(hyp_prof)
        if dens_arr is not None and hyp_arr is not None and len(dens_arr) == 25 and len(hyp_arr) == 25:
            mask = (((dens_arr - hyp_arr) > 0.05) & (hyp_arr/dens_arr > 0.05) & (hyp_arr/dens_arr < 0.50))
            # Find the longest contiguous True segment
            max_width = 0
            current = 0
            for val in mask:
                if val:
                    current += 1
                    if current > max_width:
                        max_width = current
                else:
                    current = 0
            mixing_width = 20 * max_width
            mixing_widths.append({"index": idx, "cluster": row.get("cluster", np.nan), "Condition": row.get("Condition", None), "MixingWidth": mixing_width})
    mixing_widths_df = pd.DataFrame(mixing_widths).set_index("index")
    df.loc[mixing_widths_df.index, "MixingWidth"] = mixing_widths_df["MixingWidth"]
    # Prepare data for plotting: for each cluster, MixingWidth across conditions
    clusters = sorted(df["cluster"].dropna().unique())
    plot_data = []
    for cluster in clusters:
        for condition in valid_conditions:
            sub_df = df[(df["cluster"] == cluster) & (df["Condition"] == condition)]
            for val in sub_df["MixingWidth"].dropna():
                plot_data.append({"Cluster": f"Cluster {int(cluster)+1}", "Condition": condition, "MixingWidth": val})
    plot_df = pd.DataFrame(plot_data)
    if not plot_df.empty:
        fig, axes = plt.subplots(1, len(clusters), figsize=(4*len(clusters), 5), sharey=False)
        if len(clusters) == 1:
            axes = [axes]
        for i, cluster in enumerate(clusters):
            ax = axes[i]
            cstr = f"Cluster {int(cluster)+1}"
            sub_df = plot_df[plot_df["Cluster"] == cstr]
            if sub_df.empty:
                ax.set_title(cstr + "\n(No Data)")
                continue
            sns.boxplot(
                x="Condition", y="MixingWidth", data=sub_df, ax=ax,
                order=valid_conditions,
                boxprops=dict(alpha=0.8), showfliers=False,
            )
            means = sub_df.groupby("Condition")["MixingWidth"].mean()
            for j, cond in enumerate(valid_conditions):
                if cond in means:
                    ax.hlines(means[cond], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
            # t-test between conditions for this cluster
            pairs = [(i, j) for i in range(len(valid_conditions)) for j in range(i+1, len(valid_conditions))]
            y_max = 240; y_offset = 40
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=0, top=max_y + y_offset/4)
            for k, (idx1, idx2) in enumerate(pairs):
                cond1, cond2 = valid_conditions[idx1], valid_conditions[idx2]
                group1 = sub_df[sub_df["Condition"] == cond1]["MixingWidth"].dropna()
                group2 = sub_df[sub_df["Condition"] == cond2]["MixingWidth"].dropna()
                if group1.empty or group2.empty:
                    continue
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                x1, x2 = idx1, idx2
                y = y_max + y_offset * (k + 1)
                if pair_pval < 0.001:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
            ax.set_title(f"{cstr}", fontsize=10)
            ax.set_xlabel("")
            if i == 0:
                ax.set_ylabel("Mixing Width")
            else: ax.set_ylabel("")
        plt.suptitle("Mixing Width Across Conditions", y=0.90)
        plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
        plt.show()

# Plot hypoxic and normoxic density profiles versus mean diameter for all cluster types side by side for each condition
if "HypoxicDensityProfile" in df.columns:
    shell_width = 20  # Each shell is 20 microns wide
    shell_centers = np.arange(shell_width/2, shell_width*25, shell_width)
    condition_palette = ["tab:blue", "tab:purple", "tab:pink", "tab:red"]
    condition_colors = {cond: condition_palette[i % len(condition_palette)] for i, cond in enumerate(conditions)}
    line_styles = {}
    # Pre-parse all valid HypoxicDensityProfiles and NormoxicDensityProfiles for speed
    parsed_hypoxic_profiles = {}
    parsed_normoxic_profiles = {}
    for condition in conditions:
        cond_df = df[df["Condition"] == condition]
        for cluster in range(num_clusters):
            cluster_df = cond_df[cond_df["cluster"] == cluster]
            # Drop rows with missing or malformed HypoxicDensityProfile
            cluster_df = cluster_df[cluster_df["HypoxicDensityProfile"].notnull()]
            hypoxic_profiles = []
            normoxic_profiles = []
            for idx, row in cluster_df.iterrows():
                # Parse HypoxicDensityProfile
                hyp_prof = row["HypoxicDensityProfile"]
                if isinstance(hyp_prof, str):
                    try:
                        hyp_arr = np.array([float(x) for x in hyp_prof.split(":")])
                        if hyp_arr.size == 25:
                            hypoxic_profiles.append(hyp_arr)
                    except Exception:
                        continue
                elif isinstance(hyp_prof, (list, np.ndarray)) and len(hyp_prof) == 25:
                    hypoxic_profiles.append(np.array(hyp_prof))
                # Parse NormoxicDensityProfile
                dens_prof = row.get("DensityProfile", None)
                if dens_prof is not None:
                    if isinstance(dens_prof, str):
                        try:
                            dens_arr = np.array([float(x) for x in dens_prof.split(":")])
                        except Exception:
                            dens_arr = None
                    elif isinstance(dens_prof, (list, np.ndarray)) and len(dens_prof) == 25:
                        dens_arr = np.array(dens_prof)
                    else:
                        dens_arr = None
                    if dens_arr is not None and len(hypoxic_profiles) > 0:
                        norm_arr = np.maximum(0, dens_arr - hypoxic_profiles[-1])
                        if len(norm_arr) == 25:
                            normoxic_profiles.append(norm_arr)
            if hypoxic_profiles:
                parsed_hypoxic_profiles[(condition, cluster)] = np.array(hypoxic_profiles)
            if normoxic_profiles:
                parsed_normoxic_profiles[(condition, cluster)] = np.array(normoxic_profiles)
    # Plot HypoxicDensityProfile and NormoxicDensityProfile for each condition
    for density_type, parsed_profiles, ylabel, suptitle in [
        ("Hypoxic", parsed_hypoxic_profiles, "Hypoxic Density", "Hypoxic Density Across Mean Diameter"),
        ("Normoxic", parsed_normoxic_profiles, "Normoxic Density", "Normoxic Density Across Mean Diameter")
    ]:
        fig, axes = plt.subplots(1, num_clusters, figsize=(4 * num_clusters, 5), sharey=False)
        for cluster in range(num_clusters):
            ax = axes[cluster] if num_clusters > 1 else axes
            for i, condition in enumerate(conditions):
                profiles = parsed_profiles.get((condition, cluster), None)
                if profiles is None or len(profiles) == 0:
                    continue
                mean_profile = profiles.mean(axis=0)
                std_profile = profiles.std(axis=0)
                style = line_styles.get(condition, {"linestyle": "-", "linewidth": 2, "alpha": 0.8})
                color = condition_colors.get(condition, None)
                ax.plot(
                    shell_centers, mean_profile,
                    label=condition,
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style["alpha"],
                    color=color
                )
                ax.fill_between(
                    shell_centers, mean_profile - std_profile, mean_profile + std_profile,
                    alpha=0.10, color=color
                )
            ax.set_xlabel("Mean Diameter")
            if cluster == 0:
                ax.set_ylabel(ylabel)
            else: ax.set_ylabel("")
            ax.set_title(f"Cluster {cluster+1}", fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 500)
            ax.legend(fontsize=8, loc='upper right')
        plt.suptitle(suptitle, y=0.90)
        plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92])
        plt.show()

# Box plots of diameter across different parameter values for each condition, with each condition on a separate row
param_features = [
    ("CellGrowthModifier", "Cell Growth Modifier"),
    ("CellECMContactEnergy", "ECM Contact Energy"),
    ("CellCellContactEnergy", "Cell-Cell Contact Energy"),
    ("ChemotaxisModifier", "Chemotaxis Modifier"),
    ("MMPSecretionModifier", "MMP Secretion Modifier")
]
for condition in conditions:
    # Define cluster filters and row labels
    cluster_filters = [
        (None, "All Clusters"),
        ([0,2], "Clusters 1+3"),
        ([1], "Cluster 2")
    ]
    n_rows = len(cluster_filters)
    n_cols = len(param_features)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 6*n_rows), sharey=True)
    for row_idx, (cluster_sel, row_title) in enumerate(cluster_filters):
        if cluster_sel is None:
            plot_df = df[df["Condition"] == condition].copy()
        else:
            plot_df = df[(df["Condition"] == condition) & (df["cluster"].isin(cluster_sel))].copy()
        for col_idx, (param, param_label) in enumerate(param_features):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            param_values = sorted(df[param].unique())
            sns.boxplot(
                x=param, y="MajorAxis", data=plot_df, ax=ax, order=param_values,
                boxprops=dict(alpha=0.8), showfliers=False
            )
            means = plot_df.groupby(param)["MajorAxis"].mean()
            for i, val in enumerate(param_values):
                if val in means:
                    ax.hlines(means[val], i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
            if row_idx == 0:
                ax.set_title(param_label, fontsize=10)
            else: ax.set_title("")
            if col_idx == 0:
                ax.set_ylabel(row_title + "\nMajor Axis")
            else: ax.set_ylabel("")
            ax.set_xlabel("")
            # Pairwise t-tests for significance annotation
            pair_indices = [(0,1), (1,2), (0,2)]
            y_max = 450; y_offset = 100
            max_y = y_max + y_offset * (len(pair_indices) + 1.25)
            ax.set_ylim(bottom=100, top=max_y + y_offset/4)
            # Annotate significant differences
            for i, (idx1, idx2) in enumerate(pair_indices):
                val1 = param_values[idx1]
                val2 = param_values[idx2]
                group1 = plot_df[plot_df[param] == val1]["MajorAxis"]
                group2 = plot_df[plot_df[param] == val2]["MajorAxis"]
                if group1.empty or group2.empty:
                    continue
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                y = y_max + y_offset * (i + 1)
                if pair_pval < 0.001:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.01:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.05:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.10:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Major Axis Across Parameter Values ({condition})", y=0.95)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Repeat the previous plots, but compare (cluster 1 + cluster 3) vs cluster 2 for each parameter value
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    n_cols = len(param_features)
    n_rows = len(df[param_features[0][0]].unique())  # assumes all params have same number of unique values
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 6*n_rows), sharey=True)
    for col_idx, (param, param_label) in enumerate(param_features):
        param_values = sorted(df[param].unique())
        for row_idx, param_value in enumerate(param_values):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            subset = cond_df[cond_df[param] == param_value]
            # Group clusters: group A = cluster 1 + cluster 3, group B = cluster 2
            groupA = subset[subset["cluster"].isin([0, 2])]
            groupB = subset[subset["cluster"] == 1]
            plot_data = []
            for _, row in groupA.iterrows():
                plot_data.append({"group": "Clusters 1+3", "MajorAxis": row["MajorAxis"]})
            for _, row in groupB.iterrows():
                plot_data.append({"group": "Cluster 2", "MajorAxis": row["MajorAxis"]})
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(
                x="group", y="MajorAxis", hue="group", data=plot_df, ax=ax,
                order=["Clusters 1+3", "Cluster 2"],
                boxprops=dict(alpha=0.8), showfliers=False,
                palette=["tab:olive", "tab:blue"], legend=False
            )
            for i, group_name in enumerate(["Clusters 1+3", "Cluster 2"]):
                group_mean = plot_df[plot_df["group"] == group_name]["MajorAxis"].mean()
                ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
            ax.set_title(f"{param_label} = {param_value}", fontsize=10)
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Major Axis")
            else: ax.set_ylabel("")
            # t-test between groupA and groupB
            pairs = [(0, 1)]
            y_max = 450; y_offset = 80
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=100, top=max_y + y_offset/4)
            # Annotate significant differences
            for i, (idx1, idx2) in enumerate(pairs):
                e1, e2 = param_values[idx1], param_values[idx2]
                group1 = cond_df[cond_df[param] == e1]["MajorAxis"]
                group2 = cond_df[cond_df[param] == e2]["MajorAxis"]
                if group1.empty or group2.empty:
                    continue
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                y = y_max + y_offset * (i + 1)
                if pair_pval < 0.001:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "***", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.01:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "**", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.05:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "*", ha='center', va='bottom', color='k', fontsize=16)
                elif pair_pval < 0.10:
                    ax.plot([idx1, idx1, idx2, idx2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((idx1 + idx2) / 2, y + y_offset/8, "⚬", ha='center', va='bottom', color='k', fontsize=16)
    plt.suptitle(f"Major Axis Across Clusters ({condition})", y=0.95)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=7.0)
    plt.show()

