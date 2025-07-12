
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
df_scan4 = pd.read_csv(cwd + "cancore2_scan-24/cancore2_scan-24_1200.csv")
df_scan4a = pd.read_csv(cwd + "cancore2_scan-24/cancore2_scan-24_1200_adj.csv")
df_scan8 = pd.read_csv(cwd + "cancore2_scan-28/cancore2_scan-28_1200.csv")
# Read the CSV file
df_scan0["Condition"] = "Model 1.0, Threshold = 0.0"
df_scan0a["Condition"] = "Model 1.0, MCS 0960"
df_scan0b["Condition"] = "Model 1.0, MCS 0720"
df_scan0c["Condition"] = "Model 1.0, MCS 0480"
df_scan1["Condition"] = "Model 1.1, Threshold = 2.0"
df_scan2["Condition"] = "Model 1.1, Threshold = 4.0"
df_scan2a["Condition"] = "Model 1.1, MCS 0960"
df_scan2b["Condition"] = "Model 1.1, MCS 0720"
df_scan2c["Condition"] = "Model 1.1, MCS 0480"
df_scan3["Condition"] = "Model 1.1, Threshold = 8.0"
df_scan4["Condition"] = "Model 1.0, Mesenchymal"
df_scan4a["Condition"] = "Model 1.0, Mesenchymal (Adjusted)"
df_scan8["Condition"] = "Spatial Position"
df = pd.concat([df_scan0, df_scan1, df_scan2, df_scan3], ignore_index=True)
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

# Plot centre and margin lineage distributions across mean diameter for spatial position
if "HypoxicDensityProfile" in df.columns and "DensityProfile" in df.columns:
    spatial_condition = "Spatial Position"
    cond_df = df[df["Condition"] == spatial_condition]
    if not cond_df.empty:
        shell_width = 20
        shell_centers = np.arange(shell_width/2, shell_width*25, shell_width)
        clusters = sorted(cond_df["cluster"].dropna().unique())
        num_clusters = len(clusters)
        # Use cluster_name mapping (cluster number -> name)
        if "cluster_name" in cond_df.columns:
            cluster_label_map = cond_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
        else:
            cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
        fig, axes = plt.subplots(1, num_clusters, figsize=(12, 5), sharey=False)
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
            ax.set_xlabel("Mean Diameter (μm)")
            if i == 0:
                ax.set_ylabel("Lineage Density")
            else:
                ax.set_ylabel("")
            cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
            ax.set_title(cluster_label, fontsize=10)
            ax.set_ylim(0, 1.0)
            ax.set_xlim(0, 500)
            ax.legend(fontsize=9, loc='upper right')
        plt.suptitle("Centre and Margin Lineage Distribution", y=0.90)
        plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], w_pad=1.0)
        plt.show()

# Plot diameter comparisons for SSOx = 2.0, 4.0, 8.0 vs 0.0 and Max
# Add df_scan4 (SSOx = Max) to the main dataframe for this plot
df_plot = pd.concat([df, df_scan4a], ignore_index=True)
ss_conditions = [(cond, cond) for cond in conditions if cond not in ["Model 1.0, Threshold = 0.0", "Model 1.0, Mesenchymal (Adjusted)"]]
fig, axes = plt.subplots(1, len(ss_conditions), figsize=(12, 5), sharey=False)
if len(ss_conditions) == 1:
    axes = [axes]
for i, (mid_cond, mid_label) in enumerate(ss_conditions):
    ax = axes[i]
    plot_conds = ["Model 1.0, Threshold = 0.0", mid_cond, "Model 1.0, Mesenchymal (Adjusted)"]
    plot_labels = ["Epithelial", "Mixed", "Mesenchymal"]
    plot_df = df_plot[df_plot["Condition"].isin(plot_conds)].copy()
    plot_df["Condition"] = plot_df["Condition"].map({
        "Model 1.0, Threshold = 0.0": "Epithelial",
        mid_cond: "Mixed",
        "Model 1.0, Mesenchymal (Adjusted)": "Mesenchymal"
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
        ax.set_ylabel("Major Diameter (μm)")
    else: ax.set_ylabel("")
    # Pairwise t-tests for significance annotation
    pairs = [(0,1), (1,2), (0,2)]
    y_max = 450; y_offset = 60
    max_y = y_max + y_offset * (len(pairs) + 1.25)
    ax.set_ylim(bottom=0, top=max_y + y_offset/4)
    # Map plot_conds to plot_labels for correct annotation positions
    cond_to_label = {
        "Model 1.0, Threshold = 0.0": "Epithelial",
        mid_cond: "Mixed",
        "Model 1.0, Mesenchymal (Adjusted)": "Mesenchymal"
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
plt.suptitle("Major Diameter Across Homotypic and Heterotypic Populations", y=0.90)
plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=1.0)
plt.show()

# Generate plots comparing diameter, density, and convolution across conditions for each cluster
measurables = [
    ("MajorAxis", "Major Diameter"),
    ("Density", "Density"),
    ("Convolution", "Convolution")
]
if any("MCS" in str(cond) for cond in conditions):
    title = "Monte Carlo Steps"
else:
    title = "State-Switching Thresholds"
cluster_values = sorted(df["cluster"].dropna().unique())
n_cols = len(cluster_values)

# Use cluster_name mapping (cluster number -> name)
if "cluster_name" in df.columns:
    cluster_label_map = df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
else:
    cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}

for i, (meas, meas_label) in enumerate(measurables):
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 5), sharey=False)
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
        cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
        ax.set_title(cluster_label, fontsize=10)
        if col_idx == 0:
            if meas == "MajorAxis":
                ax.set_ylabel(f"{meas_label} (μm)")
            else:
                ax.set_ylabel(meas_label)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")
        pairs = [(1,2), (2,3), (1,3), (0,1), (0,2), (0,3)]
        # t-test between conditions for this cluster and measurable
        if meas == "MajorAxis":
            y_max = 450; y_offset = 50
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=100, top=820)
        if meas == "Density":
            y_max = 0.85; y_offset = 0.06
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=0.4, top=1.3)
        if meas == "Convolution":
            y_max = 5.4; y_offset = 0.60
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=1.0, top=10)
        # Annotate significant differences
        for i, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(conditions) or idx2 >= len(conditions):
                continue
            e1, e2 = conditions[idx1], conditions[idx2]
            group1 = cluster_df[cluster_df["Condition"] == e1][meas]
            group2 = cluster_df[cluster_df["Condition"] == e2][meas]
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
    plt.suptitle(f"{meas_label} Across {title}", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=1.0)
    plt.show()

# Plot density profile versus Mean Diameter (μm) for all cluster types side by side for each condition
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
    cond_df = df[df["Condition"] == "Model 1.1, Threshold = 4.0"]
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
    plt.suptitle(f"{label} Density Across {title} (Model 1.1, Threshold = 4.0)", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Plot hypoxic fraction across diameter for clusters 1+3 and cluster 2 using cluster labels
for condition in conditions:
    # Define cluster filters and titles
    cluster_filters = [
        (None, "All Modes"),
        ([0, 2], "Bulk Modes"),
        ([1], "Dendritic Mode")
    ]
    # Skip conditions without hypoxia data
    if condition in ["SSOx = 0.0", "SSOx = Max"]:
        continue
    cond_df = df[df["Condition"] == condition]
    # Use cluster_name mapping (cluster number -> name)
    if "cluster_name" in df.columns:
        cluster_label_map = df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    fig, axes = plt.subplots(1, len(cluster_filters), figsize=(12, 5), sharey=False)
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
                else:
                    current = 0
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
                cname = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
                ax.scatter(
                    cdf["MajorAxis"], cdf["HypoxicFraction"],
                    color=cluster_colors[int(cluster) % len(cluster_colors)],
                    alpha=0.8, label=cname
                )
        # Fit a polynomial (degree 3) using 95th percentile min and max for x_fit
        if len(df_sub) > 1:
            x, y = df_sub["MajorAxis"].values, df_sub["HypoxicFraction"].values
            coeffs = np.polyfit(x, y, 3)
            poly = np.poly1d(coeffs)
            x_min = np.percentile(x, 2.5)
            x_max = np.percentile(x, 97.5)
            x_fit = np.linspace(x_min, x_max, 100)
            ax.plot(x_fit, poly(x_fit), color="k", linestyle="--", linewidth=2, label="Poly Trendline")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Major Diameter (μm)")
        ax.set_xlim(100, 500)
        if row_idx == 0:
            ax.set_ylabel("Mesenchymal Fraction")
        else:
            ax.set_ylabel("")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8, loc='upper right')
    plt.suptitle(f"Mesenchymal Fraction Across Achieved Major Diameter ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=1.0)
    plt.show()

# Compare hypoxic fraction of clusters across diameter bins for each condition using cluster labels
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
    # Use cluster_name mapping (cluster number -> name)
    if "cluster_name" in cond_df.columns:
        cluster_label_map = cond_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    n_cols = len(bin_labels)
    clusters = sorted(cond_df["cluster"].dropna().unique())
    cluster_names = [cluster_label_map.get(c, f"Cluster {int(c)+1}") for c in clusters]
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 4), sharey=False)
    for col_idx, bin_label in enumerate(bin_labels):
        ax = axes[col_idx] if n_cols > 1 else axes
        bin_df = cond_df[cond_df["MajorAxisBin"] == bin_label]
        plot_data = []
        for cluster, cname in zip(clusters, cluster_names):
            cluster_df = bin_df[bin_df["cluster"] == cluster]
            if len(cluster_df) < 5:
                plot_data.append({"cluster": cname, "HypoxicFraction": -1})
            else:
                for _, row in cluster_df.iterrows():
                    plot_data.append({"cluster": cname, "HypoxicFraction": row["HypoxicFraction"]})
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(
            x="cluster", y="HypoxicFraction", hue="cluster", data=plot_df, ax=ax,
            order=cluster_names,
            boxprops=dict(alpha=0.8), showfliers=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters], legend=False
        )
        for i, cname in enumerate(cluster_names):
            group_mean = plot_df[plot_df["cluster"] == cname]["HypoxicFraction"].mean()
            ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Major Diameter {bin_label} (μm)", fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Mesenchymal Fraction")
        else:
            ax.set_ylabel("")
        # t-test between clusters
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 0.80; y_offset = 0.10
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=0.00, top=max_y + y_offset/4)
        # Annotate significant differences
        for k, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(cluster_names) or idx2 >= len(cluster_names):
                continue
            cname1 = cluster_names[idx1]
            cname2 = cluster_names[idx2]
            group1 = plot_df[plot_df["cluster"] == cname1]["HypoxicFraction"]
            group2 = plot_df[plot_df["cluster"] == cname2]["HypoxicFraction"]
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
    plt.suptitle(f"Mesenchymal Fraction Across Modes ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()

# Box plots of Normoxic Width across clusters for each condition using cluster labels
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
    # Use cluster_name mapping (cluster number -> full name)
    if "cluster_name" in cond_df.columns:
        cluster_label_map = cond_df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    n_cols = len(bin_labels)
    clusters = sorted(cond_df["cluster"].dropna().unique())
    cluster_names = [cluster_label_map.get(c, f"Cluster {int(c)+1}") for c in clusters]
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 4), sharey=False)
    for col_idx, bin_label in enumerate(bin_labels):
        ax = axes[col_idx] if n_cols > 1 else axes
        bin_df = cond_df[cond_df["MajorAxisBin"] == bin_label]
        plot_data = []
        for cluster, cname in zip(clusters, cluster_names):
            cluster_df = bin_df[bin_df["cluster"] == cluster]
            if len(cluster_df) < 5:
                plot_data.append({"cluster": cname, "NormoxicWidth": -1})
            else:
                for _, row in cluster_df.iterrows():
                    plot_data.append({"cluster": cname, "NormoxicWidth": row["NormoxicWidth"]})
        plot_df = pd.DataFrame(plot_data)
        sns.boxplot(
            x="cluster", y="NormoxicWidth", hue="cluster", data=plot_df, ax=ax,
            order=cluster_names,
            boxprops=dict(alpha=0.8), showfliers=False,
            palette=[cluster_colors[int(c) % len(cluster_colors)] for c in clusters], legend=False
        )
        for i, cname in enumerate(cluster_names):
            group_mean = plot_df[plot_df["cluster"] == cname]["NormoxicWidth"].mean()
            ax.hlines(group_mean, i-0.4, i+0.4, color="tab:red", linestyle="-", linewidth=2)
        ax.set_title(f"Major Diameter {bin_label} (μm)", fontsize=10)
        ax.set_xlabel("")
        if col_idx == 0:
            ax.set_ylabel("Epithelial Layer Width (μm)")
        else:
            ax.set_ylabel("")
        # t-test between clusters
        pairs = [(0,1), (1,2), (0,2)]
        y_max = 240; y_offset = 30
        max_y = y_max + y_offset * (len(pairs) + 1.25)
        ax.set_ylim(bottom=50.0, top=max_y + y_offset/4)
        # Annotate significant differences
        for k, (idx1, idx2) in enumerate(pairs):
            if idx1 >= len(cluster_names) or idx2 >= len(cluster_names):
                continue
            cname1 = cluster_names[idx1]
            cname2 = cluster_names[idx2]
            group1 = plot_df[plot_df["cluster"] == cname1]["NormoxicWidth"].dropna()
            group2 = plot_df[plot_df["cluster"] == cname2]["NormoxicWidth"].dropna()
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
    plt.suptitle(f"Epithelial Layer Width Across Modes ({condition})", y=0.90)
    plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0)
    plt.show()


