
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
df_scan2 = pd.read_csv(cwd + "cancore2_scan-22/cancore2_scan-22_1200.csv")
df_scan4a = pd.read_csv(cwd + "cancore2_scan-24/cancore2_scan-24_1200_adj.csv")
df_scan5 = pd.read_csv(cwd + "cancore2_scan-25/cancore2_scan-25_1200.csv")
df_scan6 = pd.read_csv(cwd + "cancore2_scan-26/cancore2_scan-26_1200.csv")
df_scan7 = pd.read_csv(cwd + "cancore2_scan-27/cancore2_scan-27_1200.csv")
df_scan7a = pd.read_csv(cwd + "cancore2_scan-27/cancore2_scan-27_1200_adj.csv")

# Read the CSV file
df_scan0["Condition"] = "Model 1.0, Threshold = 0.0"
df_scan2["Condition"] = "Model 1.1, Threshold = 4.0"
df_scan4a["Condition"] = "Model 1.0, Mesenchymal (Adjusted)"
df_scan5["Condition"] = "Model 1.2, 5% Resistor"
df_scan6["Condition"] = "Model 1.2, 5% Weak Resistor"
df_scan7["Condition"] = "Model 1.0, Weak Resistor"
df_scan7a["Condition"] = "Model 1.0, Weak Resistor (Adjusted)"
df = pd.concat([df_scan0, df_scan2, df_scan4a, df_scan5, df_scan6], ignore_index=True)
df = df[(df["MajorAxis"] <= 520) & (df["Density"] >= 0.40) & (df["Convolution"] <= 6.8)] # eliminate outliers
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

param_condition = "Model 1.0, MCS 1200"
ssox0_df = df[df["Condition"] == param_condition].copy()
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

# Plot diameter comparisons for two sets of conditions, side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
# First plot: scan0, scan4a, scan7a, scan6
plot_conds1 = [
    ("Model 1.0, Threshold = 0.0", "Epi"),
    ("Model 1.0, Mesenchymal (Adjusted)", "Mes"),
    ("Model 1.0, Weak Resistor (Adjusted)", "ResLo"),
    ("Model 1.2, 5% Weak Resistor", "Epi+Mes+ResLo"),
]
df_plot1 = pd.concat([df_scan0, df_scan4a, df_scan7a, df_scan6], ignore_index=True)
plot_labels1 = [label for _, label in plot_conds1]
cond_map1 = {cond: label for cond, label in plot_conds1}
df_plot1 = df_plot1[df_plot1["Condition"].isin(cond_map1.keys())].copy()
df_plot1["Condition"] = df_plot1["Condition"].map(cond_map1)

ax = axes[0]
sns.boxplot(
    x="Condition", y="MajorAxis", data=df_plot1, ax=ax,
    order=plot_labels1,
    boxprops=dict(alpha=0.8), showfliers=False,
)
means = df_plot1.groupby("Condition")["MajorAxis"].mean()
for j, label in enumerate(plot_labels1):
    if label in means:
        ax.hlines(means[label], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
ax.set_xlabel("Phenotype")
ax.set_ylabel("Major Diameter (μm)")
ax.set_title("")
# Pairwise t-tests for significance annotation
pairs = [(1,2), (2,3), (1,3), (0,1), (0,2), (0,3)]
y_max = 450; y_offset = 60
max_y = y_max + y_offset * (len(pairs) + 1.25)
ax.set_ylim(bottom=0, top=max_y + y_offset/4)
label_to_pos1 = {label: idx for idx, label in enumerate(plot_labels1)}
for k, (idx1, idx2) in enumerate(pairs):
    if idx1 >= len(plot_labels1) or idx2 >= len(plot_labels1):
        continue
    label1, label2 = plot_labels1[idx1], plot_labels1[idx2]
    group1 = df_plot1[df_plot1["Condition"] == label1]["MajorAxis"]
    group2 = df_plot1[df_plot1["Condition"] == label2]["MajorAxis"]
    if group1.empty or group2.empty:
        continue
    t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
    x1, x2 = label_to_pos1[label1], label_to_pos1[label2]
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

# Second plot: scan0, scan2, scan5, scan6
plot_conds2 = [
    ("Model 1.0, Threshold = 0.0", "Epi"),
    ("Model 1.1, Threshold = 4.0", "Epi+Mes"),
    ("Model 1.2, 5% Resistor", "Epi+Mes+Res"),
    ("Model 1.2, 5% Weak Resistor", "Epi+Mes+ResLo"),
]
df_plot2 = pd.concat([df_scan0, df_scan2, df_scan5, df_scan6], ignore_index=True)
plot_labels2 = [label for _, label in plot_conds2]
cond_map2 = {cond: label for cond, label in plot_conds2}
df_plot2 = df_plot2[df_plot2["Condition"].isin(cond_map2.keys())].copy()
df_plot2["Condition"] = df_plot2["Condition"].map(cond_map2)

ax = axes[1]
sns.boxplot(
    x="Condition", y="MajorAxis", data=df_plot2, ax=ax,
    order=plot_labels2,
    boxprops=dict(alpha=0.8), showfliers=False,
)
means = df_plot2.groupby("Condition")["MajorAxis"].mean()
for j, label in enumerate(plot_labels2):
    if label in means:
        ax.hlines(means[label], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
ax.set_xlabel("Phenotype")
ax.set_ylabel("Major Diameter (μm)")
ax.set_title("")
# Pairwise t-tests for significance annotation
ax.set_ylim(bottom=0, top=max_y + y_offset/4)
label_to_pos2 = {label: idx for idx, label in enumerate(plot_labels2)}
for k, (idx1, idx2) in enumerate(pairs):
    if idx1 >= len(plot_labels2) or idx2 >= len(plot_labels2):
        continue
    label1, label2 = plot_labels2[idx1], plot_labels2[idx2]
    group1 = df_plot2[df_plot2["Condition"] == label1]["MajorAxis"]
    group2 = df_plot2[df_plot2["Condition"] == label2]["MajorAxis"]
    if group1.empty or group2.empty:
        continue
    t_stat, pair_pval = ttest_ind(group1, group2, equal_var=False)
    x1, x2 = label_to_pos2[label1], label_to_pos2[label2]
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
plt.suptitle("Major Diameter Across Population Phenotypes", y=0.90)
plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=2.0)
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
    title = "Population Phenotypes"
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
        # Define short labels for conditions in order
        condition_short_labels = ["Epi", "Epi+Mes", "Res", "ResLo"]
        short_cond = condition_short_labels[:len(conditions)]
        # Map each condition to its short label
        cond_to_short = {cond: condition_short_labels[i] if i < len(condition_short_labels) else str(cond)
                         for i, cond in enumerate(conditions)}
        cluster_df["ConditionShort"] = cluster_df["Condition"].map(cond_to_short)
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
        ax.set_xlabel("Mean Diameter")
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
        # Define short labels for conditions in order
        condition_short_labels = ["Epi", "Epi+Mes", "Res", "ResLo"]
        short_cond = condition_short_labels[:len(df["Condition"].unique())]
        cluster_df["ConditionShort"] = pd.Categorical(
            [condition_short_labels[i] if i < len(condition_short_labels) else str(cond)
             for i, cond in enumerate(df["Condition"].unique())],
            categories=condition_short_labels, ordered=True
        )[cluster_df["Condition"].map({cond: i for i, cond in enumerate(df["Condition"].unique())}).values]
        sns.boxplot(
            x="ConditionShort", y=sec, data=cluster_df, ax=ax,
            order=condition_short_labels,
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
    first_condition = conditions[2]
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

# Plot hypoxic and normoxic density profiles versus Mean Diameter (μm) for all cluster types side by side for each condition
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
        ("Hypoxic", parsed_hypoxic_profiles, "Mesenchymal Density", "Mesenchymal Density Across Mean Diameter"),
        ("Normoxic", parsed_normoxic_profiles, "Epithelial Density", "Epithelial Density Across Mean Diameter")
    ]:
        fig, axes = plt.subplots(1, num_clusters, figsize=(12, 5), sharey=False)
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
            ax.set_xlabel("Mean Diameter (μm)")
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
    # Define short labels for valid conditions
    condition_short_labels = ["Epi+Mes", "Res", "ResLo"]
    short_cond = condition_short_labels[:len(valid_conditions)]
    cond_to_short = {cond: condition_short_labels[i] if i < len(condition_short_labels) else str(cond)
                        for i, cond in enumerate(valid_conditions)}
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
            mixing_widths.append({
                "index": idx,
                "cluster": row.get("cluster", np.nan),
                "cluster_name": row.get("cluster_name", None),
                "Condition": row.get("Condition", None),
                "MixingWidth": mixing_width
            })
    mixing_widths_df = pd.DataFrame(mixing_widths).set_index("index")
    df.loc[mixing_widths_df.index, "MixingWidth"] = mixing_widths_df["MixingWidth"]
    # Prepare data for plotting: for each cluster, MixingWidth across conditions
    clusters = sorted(df["cluster"].dropna().unique())
    # Use cluster_name mapping if available
    if "cluster_name" in df.columns:
        cluster_label_map = df.drop_duplicates("cluster")[["cluster", "cluster_name"]].set_index("cluster")["cluster_name"].to_dict()
    else:
        cluster_label_map = {0: "Slow Bulk", 1: "Dendritic", 2: "Fast Bulk"}
    plot_data = []
    for cluster in clusters:
        cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
        for condition in valid_conditions:
            sub_df = df[(df["cluster"] == cluster) & (df["Condition"] == condition)]
            for val in sub_df["MixingWidth"].dropna():
                plot_data.append({
                    "Cluster": cluster_label,
                    "Condition": condition,
                    "ConditionShort": cond_to_short.get(condition, str(condition)),
                    "MixingWidth": val
                })
    plot_df = pd.DataFrame(plot_data)
    if not plot_df.empty:
        fig, axes = plt.subplots(1, len(clusters), figsize=(12, 5), sharey=False)
        if len(clusters) == 1:
            axes = [axes]
        for i, cluster in enumerate(clusters):
            cluster_label = cluster_label_map.get(cluster, f"Cluster {int(cluster)+1}")
            ax = axes[i]
            sub_df = plot_df[plot_df["Cluster"] == cluster_label]
            if sub_df.empty:
                ax.set_title(cluster_label + "\n(No Data)")
                continue
            sns.boxplot(
                x="ConditionShort", y="MixingWidth", data=sub_df, ax=ax,
                order=short_cond,
                boxprops=dict(alpha=0.8), showfliers=False,
            )
            means = sub_df.groupby("ConditionShort")["MixingWidth"].mean()
            for j, cond_short in enumerate(short_cond):
                if cond_short in means:
                    ax.hlines(means[cond_short], j-0.4, j+0.4, color="tab:red", linestyle="-", linewidth=2)
            # t-test between conditions for this cluster
            pairs = [(i, j) for i in range(len(short_cond)) for j in range(i+1, len(short_cond))]
            y_max = 240; y_offset = 30
            max_y = y_max + y_offset * (len(pairs) + 1.25)
            ax.set_ylim(bottom=0, top=max_y + y_offset/4)
            for k, (idx1, idx2) in enumerate(pairs):
                cond1, cond2 = short_cond[idx1], short_cond[idx2]
                group1 = sub_df[sub_df["ConditionShort"] == cond1]["MixingWidth"].dropna()
                group2 = sub_df[sub_df["ConditionShort"] == cond2]["MixingWidth"].dropna()
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
            ax.set_title(f"{cluster_label}", fontsize=10)
            ax.set_xlabel("")
            if i == 0:
                ax.set_ylabel("Mixing Layer Width (μm)")
            else:
                ax.set_ylabel("")
        plt.suptitle("Mixing Layer Width Across Population Phenotypes", y=0.90)
        plt.tight_layout(rect=[0.04, 0.08, 0.96, 0.92], h_pad=5.0, w_pad=1.0)
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

