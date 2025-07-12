
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

# Read the CSV file
df_scan0["Condition"] = "Model 1.0, MCS 1200"
df_scan0a["Condition"] = "Model 1.0, MCS 0960"
df_scan0b["Condition"] = "Model 1.0, MCS 0720"
df_scan0c["Condition"] = "Model 1.0, MCS 0480"
df = pd.concat([df_scan0, df_scan0a, df_scan0b, df_scan0c], ignore_index=True)
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
    # Assign custom names: 1=Slow Bulk, 2=Dendritic, 3=Fast Bulk
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
    # Assign custom cluster names
    cluster_name_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    df.loc[cond_mask, "cluster_name"] = df.loc[cond_mask, "cluster"].map(cluster_name_map)
    
# Assign clusters to MCS conditions based on identical state-switching threshold as their parent scan
if "StateSwitchingThreshold" in df.columns:
    # Identify SSOx and MCS conditions by string patterns
    ss_conditions = [cond for cond in df["Condition"].unique() if "1200" in str(cond) or "MCS" not in str(cond)]
    mcs_conditions = [cond for cond in df["Condition"].unique() if "MCS" in str(cond) and "1200" not in str(cond)]
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

cluster_counts = df["cluster"].value_counts().sort_index()
total_points = len(df)
conditions = df["Condition"].unique().tolist()
for idx, condition in enumerate(conditions):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(6, 6))
    cond_df = df[df["Condition"] == condition]
    for cluster in range(num_clusters):
        cluster_points = cond_df[cond_df["cluster"] == cluster]
        color = cluster_colors[cluster % len(cluster_colors)]
        count = len(cluster_points)
        percent = 100.0 * count / len(cond_df) if len(cond_df) > 0 else 0
        # Use cluster name if available, else fallback to Cluster N
        if "cluster_name" in cluster_points.columns and not cluster_points["cluster_name"].isnull().all():
            cname = cluster_points["cluster_name"].iloc[0]
            label = f"{cname} ({percent:.1f}%)"
        else:
            label = f"Cluster {cluster+1} ({percent:.1f}%)"
        ax.scatter(
            cluster_points["Density"], cluster_points["Convolution"], cluster_points["MajorAxis"],
            label=label, alpha=0.8, color=color
        )
    ax.set_xlim(0.40, 1.00)
    ax.set_ylim(1.0, 7.0)
    ax.set_zlim(100, 600)
    ax.set_xlabel("Density")
    ax.set_ylabel("Convolution")
    ax.set_zlabel("Major Diameter (μm)")
    ax.view_init(elev=10, azim=45, roll=-1)
    ax.set_title(condition)
    # Move legend to bottom, arrange labels horizontally (side by side)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=1, frameon=False)
    col_labels = ["Cluster", "Count", "Gmod", "Eadh", "Cadh", "Ctax", "Msec"]
    plt.tight_layout(w_pad=-8.0, rect=[0.04, 0.08, 0.96, 0.92])
    plt.show()

# Stacked bar charts of parameter necessity for SSOx = 0.0
param_condition = "Model 1.0, MCS 1200"
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
fig, axes = plt.subplots(1, n_params, figsize=(12, 5), sharey=True)
if n_params == 1:
    axes = [axes]

# Use cluster_name mapping from earlier if available, but abbreviate
abbrev_map = {"Slow Bulk": "S.B.", "Dendritic": "Den.", "Fast Bulk": "F.B."}
if "cluster_name" in ssox0_df.columns:
    cluster_label_map = ssox0_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    cluster_labels = [abbrev_map.get(cluster_label_map.get(c, ""), f"Cluster {int(c)+1}") for c in clusters]
else:
    cluster_labels = ["S.B.", "Den.", "F.B."]

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
            cluster_labels,
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
plt.suptitle(f"Parameter Necessity Across Modes ({param_condition})", y=0.90)
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

# Use cluster_name mapping from earlier if available
if "cluster_name" in ssox0_df.columns:
    cluster_label_map = ssox0_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
else:
    cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}

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
    fig, ax = plt.subplots(figsize=(12, 6))
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
    cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
    ax.set_xlabel("Cell-Cell Contact Energy, Chemotaxis Modifier, MMP Secretion Modifier")
    ax.set_ylabel("Cell Growth Modifier and Cell-ECM Contact Energy")
    ax.set_title(f"Parameter Sufficiency For {cluster_label} Mode ({param_condition})", y=1.04)
    plt.xticks(rotation=0, fontsize=6, ha='center')
    plt.yticks(rotation=90, fontsize=6, ha='center')
    ax.tick_params(axis='y', pad=10)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.subplots_adjust(right=1.04, left=0.08, top=0.88, bottom=0.16)
    plt.show()

# Rose plots (polar histograms) of rotation for each cluster within each condition
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    # Use cluster_name mapping from current condition's data
    if "cluster_name" in cond_df.columns:
        cluster_label_map = cond_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    fig = plt.figure(figsize=(12, 4))
    axes = []
    # Rose plots for each cluster (polar)
    for cluster in range(num_clusters):
        ax = fig.add_subplot(1, num_clusters + 1, cluster + 1, projection='polar')
        axes.append(ax)
        cluster_df = cond_df[cond_df["cluster"] == cluster]
        cluster_label = cluster_label_map.get(cluster, f"Cluster {cluster+1}")
        if "Rotation" not in cluster_df.columns or cluster_df["Rotation"].dropna().empty:
            ax.set_title(f"{cluster_label}\n(No Rotation Data)")
            continue
        align_deg = cluster_df["Rotation"].dropna() % 360
        extended_deg = pd.concat([align_deg, (align_deg + 180) % 360])
        angles_rad = np.deg2rad(extended_deg)
        n_bins = 24
        counts, bins = np.histogram(angles_rad, bins=n_bins, range=(0, 2 * np.pi))
        total = counts.sum()
        if total > 0:
            fractions = counts / total
        else:
            fractions = counts
        widths = np.diff(bins)
        # Add light grey fill from 22.5° to 67.5° and 202.5° to 247.5°
        for centre in [45, 225]:
            ax.bar(
                [np.deg2rad(centre)], [1.0], width=[np.pi/4], bottom=0.0,
                color="tab:olive", alpha=0.5, edgecolor=None, linewidth=0, zorder=0
            )
        ax.bar(bins[:-1], fractions, width=widths, bottom=0.0, color=cluster_colors[cluster % len(cluster_colors)], alpha=0.8, edgecolor='k')
        # Set max histogram axis height per cluster
        if cluster == 0:
            ax.set_ylim(0, 0.07)
        if cluster == 1:
            ax.set_ylim(0, 0.14)
        if cluster == 2:
            ax.set_ylim(0, 0.10)
        for angle in [np.pi/4, 5*np.pi/4]:
            ax.plot([angle, angle], [0.0, 1.0], color='k', linewidth=4, solid_capstyle='round', zorder=10)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_title(cluster_label, fontsize=10)
        ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
        ax.set_xticklabels([f"{int(np.rad2deg(tick))}°" for tick in np.linspace(0, 2*np.pi, 8, endpoint=False)])
        ax.set_rlabel_position(285)

     # Eccentricity boxplot in the last column (cartesian, not polar)
    ax2 = fig.add_subplot(1, num_clusters + 1, num_clusters + 1)
    axes.append(ax2)
    ecc_data = []
    # Use abbreviations for cluster labels
    abbrev_map = {0: "S.B.", 1: "Den.", 2: "F.B."}
    for cluster in range(num_clusters):
        cluster_df = cond_df[cond_df["cluster"] == cluster]
        cluster_label = abbrev_map.get(cluster, f"Cluster {cluster+1}")
        if "Eccentricity" in cluster_df.columns:
            for val in cluster_df["Eccentricity"].dropna():
                ecc_data.append({"Cluster": cluster_label, "Eccentricity": val})
    ecc_df = pd.DataFrame(ecc_data)
    cluster_order = [abbrev_map.get(i, f"Cluster {i+1}") for i in range(num_clusters)]
    if not ecc_df.empty:
        sns.boxplot(
            x="Cluster", y="Eccentricity", hue="Cluster", data=ecc_df, ax=ax2,
            order=cluster_order,
            boxprops=dict(alpha=0.8), showfliers=False, legend=False,
            palette=[cluster_colors[i % len(cluster_colors)] for i in range(num_clusters)]
        )
        means = ecc_df.groupby("Cluster")["Eccentricity"].mean()
        for i, cname in enumerate(cluster_order):
            if cname in means:
                ax2.hlines(means[cname], i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax2.set_title(f"Eccentricity", fontsize=10)
        ax2.set_xlabel("")
        ax2.set_ylabel("    ", fontsize=18)
        ax2.set_ylim(0, 1.0)
        # Pairwise t-tests for significance annotation
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 0.92; y_offset = 0.12
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax2.set_ylim(bottom=0, top=max_y + y_offset/4)
        for k, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(cluster_order) or idx2 >= len(cluster_order):
                continue
            cname1 = cluster_order[idx1]
            cname2 = cluster_order[idx2]
            group1 = ecc_df[ecc_df["Cluster"] == cname1]["Eccentricity"]
            group2 = ecc_df[ecc_df["Cluster"] == cname2]["Eccentricity"]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/12, "***", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.01:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/12, "**", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.05:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/12, "*", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.10:
                ax2.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax2.text((x1 + x2) / 2, y + y_offset/12, "⚬", ha='center', va='bottom', color='k', fontsize=12)
    plt.suptitle(f"Rose Plots For Alignment ({condition})", y=0.92)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=-1.0)
    plt.show()

# Plot density profile versus mean diameter for all cluster types side by side for each condition
if "DensityProfile" in df.columns:
    shell_width = 20  # Each shell is 20 microns wide
    shell_centers = np.arange(shell_width/2, shell_width*25, shell_width)
    # Set color for each condition in order
    condition_palette = ["tab:blue", "tab:purple", "tab:pink", "tab:red"]
    condition_colors = {cond: condition_palette[i % len(condition_palette)] for i, cond in enumerate(conditions)}
    line_styles = {}
    # Use cluster_name mapping (cluster number -> name)
    if "cluster_name" in df.columns:
        cluster_label_map = df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
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
    fig, axes = plt.subplots(1, num_clusters, figsize=(12, 5), sharey=False)
    for cluster in range(num_clusters):
        ax = axes[cluster] if num_clusters > 1 else axes
        cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
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
        ax.set_xlabel("Mean Diameter (μm)")
        if cluster == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
        ax.set_title(cluster_label, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0, 500)
        ax.legend(fontsize=8, loc='upper right')
    plt.suptitle(f"Density Across Mean Diameter", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], w_pad=1.0)
    plt.show()
    
# Plot density by section for each cluster type across conditions, and add per-cluster plot for first condition
if any("MCS" in str(cond) for cond in conditions):
    title = "Monte Carlo Steps"
else:
    title = "State-Switching Thresholds"
section_names = ["CentreDensity", "MarginDensity"]
section_labels = ["Centre", "Margin"]
clusters = sorted(df["cluster"].dropna().unique())
n_clusters = len(clusters)
first_condition = conditions[0]

# Use cluster_name mapping (cluster number -> name)
if "cluster_name" in df.columns:
    cluster_label_map = df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
else:
    cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}

for row_idx, (sec, label) in enumerate(zip(section_names, section_labels)):
    # 3 plots for clusters across conditions + 1 plot for all clusters in first condition
    fig, axes = plt.subplots(1, n_clusters + 1, figsize=(12, 5), sharey=True)
    if n_clusters == 1:
        axes = list(axes)
    # Per-cluster plots across conditions
    for col_idx, cluster in enumerate(clusters):
        cluster_df = df[df["cluster"] == cluster]
        ax = axes[col_idx]
        cluster_df = cluster_df.copy()
        cluster_df["ConditionShort"] = cluster_df["Condition"].astype(str).str[-4:]
        short_cond = [str(cond)[-4:] for cond in df["Condition"].unique()]
        sns.boxplot(
            x="ConditionShort", y=sec, data=cluster_df, ax=ax,
            order=short_cond,
            boxprops=dict(alpha=0.8), showfliers=False,
        )
        means = cluster_df.groupby("Condition")[sec].mean()
        for j, cond in enumerate(df["Condition"].unique()):
            if cond in means:
                ax.hlines(means[cond], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
        cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
        ax.set_title(cluster_label, fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")
        ax.set_ylim(0, 1.0)
        # Annotate significance between all pairs
        pairs = [(1,2), (2,3), (1,3), (0,1), (0,2), (0,3)]
        y_max = 1.0; y_offset = 0.125
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
                    ax.text((x1 + x2) / 2, y + y_offset/12, "***", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "**", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "*", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "⚬", ha='center', va='bottom', color='k', fontsize=12)
    # Additional plot: all clusters for the first condition
    ax = axes[-1]
    cond_df = df[df["Condition"] == first_condition]
    plot_data = []
    for cluster in clusters:
        cluster_df = cond_df[cond_df["cluster"] == cluster]
        cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
        plot_data.extend([{"Cluster": cluster_label, "Density": v} for v in cluster_df[sec].dropna()])
    plot_df = pd.DataFrame(plot_data)
    cluster_order = [cluster_label_map.get(c, f"Cluster {int(c)+1}") for c in clusters]
    if not plot_df.empty:
        sns.boxplot(
            x="Cluster", y="Density", hue="Cluster", data=plot_df, ax=ax,
            order=cluster_order,
            boxprops=dict(alpha=0.8), showfliers=False, legend=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters]
        )
        means = plot_df.groupby("Cluster")["Density"].mean()
        for j, cname in enumerate(cluster_order):
            if cname in means:
                ax.hlines(means[cname], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
        # Annotate significance between clusters
        pairs = [(0, 1), (1, 2), (0, 2)]
        y_max = 1.0; y_offset = 0.125
        max_y = y_max + y_offset * (2*len(pairs) + 1.25)
        ax.set_ylim(bottom=0.0, top=max_y + y_offset/4)
        for k, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(cluster_order) or idx2 >= len(cluster_order):
                continue
            cname1 = cluster_order[idx1]
            cname2 = cluster_order[idx2]
            group1 = plot_df[plot_df["Cluster"] == cname1]["Density"]
            group2 = plot_df[plot_df["Cluster"] == cname2]["Density"]
            if group1.empty or group2.empty:
                continue
            t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
            x1, x2 = idx1, idx2
            y = y_max + y_offset * (k + 1)
            if pair_pval < 0.001:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/12, "***", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.01:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/12, "**", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.05:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/12, "*", ha='center', va='bottom', color='k', fontsize=12)
            elif pair_pval < 0.10:
                ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                ax.text((x1 + x2) / 2, y + y_offset/12, "⚬", ha='center', va='bottom', color='k', fontsize=12)
        ax.set_title(f"{label} Density Across Modes", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
    else:
        ax.set_title(f"{label} Density Across Modes", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
    plt.suptitle(f"{label} Density Across {title} ({first_condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Box plots of diameter across different parameter values for each condition, with each condition on a separate row
param_features = [
    ("CellGrowthModifier", "Cell Growth Modifier"),
    ("CellECMContactEnergy", "Cell-ECM Contact Energy"),
    ("CellCellContactEnergy", "Cell-Cell Contact Energy"),
    ("ChemotaxisModifier", "Chemotaxis Modifier"),
    ("MMPSecretionModifier", "MMP Secretion Modifier")
]
for condition in conditions:
    # Define cluster filters and row labels
    cluster_filters = [
        (None, "All Modes"),
        ([0,2], "Bulk Modes"),
        ([1], "Dendritic Mode")
    ]
    n_rows = len(cluster_filters)
    n_cols = len(param_features)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 16), sharey=True)
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
                ax.set_ylabel(row_title + "\nMajor Diameter (μm)")
            else: ax.set_ylabel("")
            ax.set_xlabel("")
            # Pairwise t-tests for significance annotation
            pair_indices = [(0,1), (1,2), (0,2)]
            y_max = 450; y_offset = 75
            max_y = y_max + y_offset * (len(pair_indices) + 1.25)
            ax.set_ylim(bottom=100, top=800)
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
                x1, x2 = idx1, idx2
                if pair_pval < 0.001:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "***", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "**", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "*", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "⚬", ha='center', va='bottom', color='k', fontsize=12)
    plt.suptitle(f"Major Diameter Across Parameter Values ({condition})", y=0.95)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Repeat the previous plots, but compare (cluster 1 + cluster 3) vs cluster 2 for each parameter value
param_features = [
    ("CellGrowthModifier", "Cell Growth Mod."),
    ("CellECMContactEnergy", "Cell-ECM Energy"),
    ("CellCellContactEnergy", "Cell-Cell Energy"),
    ("ChemotaxisModifier", "Chemotaxis Mod."),
    ("MMPSecretionModifier", "MMP Secretion Mod.")
]
for condition in conditions:
    cond_df = df[df["Condition"] == condition]
    n_cols = len(param_features)
    n_rows = len(df[param_features[0][0]].unique())  # assumes all params have same number of unique values
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 16), sharey=True)
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
                plot_data.append({"group": "Bulk", "MajorAxis": row["MajorAxis"]})
            for _, row in groupB.iterrows():
                plot_data.append({"group": "Dendritic", "MajorAxis": row["MajorAxis"]})
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(
                x="group", y="MajorAxis", hue="group", data=plot_df, ax=ax,
                order=["Bulk", "Dendritic"],
                boxprops=dict(alpha=0.8), showfliers=False,
                palette=["tab:olive", "tab:blue"], legend=False
            )
            for i, group_name in enumerate(["Bulk", "Dendritic"]):
                group_mean = plot_df[plot_df["group"] == group_name]["MajorAxis"].mean()
                ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
            ax.set_title(f"{param_label} = {param_value}", fontsize=10)
            ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("Major Diameter (μm)")
            else: ax.set_ylabel("")
            # t-test between groupA and groupB
            pairs = [(0, 1)]
            y_max = 450; y_offset = 60
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=100, top=600)
            # Annotate significant differences
            for i, (idx1, idx2) in enumerate(pairs):
                e1, e2 = param_values[idx1], param_values[idx2]
                group1 = cond_df[cond_df[param] == e1]["MajorAxis"]
                group2 = cond_df[cond_df[param] == e2]["MajorAxis"]
                if group1.empty or group2.empty:
                    continue
                t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
                y = y_max + y_offset * (i + 1)
                x1, x2 = idx1, idx2
                if pair_pval < 0.001:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "***", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.01:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "**", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.05:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "*", ha='center', va='bottom', color='k', fontsize=12)
                elif pair_pval < 0.10:
                    ax.plot([x1, x1, x2, x2], [y, y + y_offset/4, y + y_offset/4, y], lw=1.2, c='k')
                    ax.text((x1 + x2) / 2, y + y_offset/12, "⚬", ha='center', va='bottom', color='k', fontsize=12)
    plt.suptitle(f"Major Diameter Across Modes ({condition})", y=0.95)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=7.0)
    plt.show()

