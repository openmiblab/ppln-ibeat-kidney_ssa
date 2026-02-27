# --- Standard Library ---
import os
import json
import logging
import colorsys
from collections import Counter, defaultdict
from itertools import product

# --- Third-Party Libraries ---
import zarr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from tqdm import tqdm
import miblab_ssa as ssa
from miblab_plot import pvplot, mp4

# --- Science & Stats ---
from scipy.cluster.hierarchy import linkage, fcluster, set_link_color_palette, dendrogram
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# --- Configuration ---
# Set backend for non-interactive environments
matplotlib.use('Agg')

from miblab import pipe

PIPELINE = 'kidney_ssa'

STAGE = {
    'spectral': 12,
    'chebyshev': 13,
}
VRANGE = {
    'dice': [0.5, 0.9],
    'hausdorff': [0, 50],
}


def run(build):
    
    logging.info("Stage 14 --- Clustering kidneys ---")
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    # Inputs
    masks_path = os.path.join(build, PIPELINE, 'stage_9_stack_normalized', 'normalized_kidney_masks.zarr')
    matrix_csv = {
        'dice': os.path.join(build, PIPELINE, 'stage_10_dice_matrix', 'normalized_kidney_dice.csv'),
        'hausdorff': os.path.join(build, PIPELINE, 'stage_11_hausdorff_matrix', 'normalized_kidney_hausdorff.csv'),
    }

    N_COMP = 100 # Needs a better support for the choice
    
    models = [ 'chebyshev', 'spectral']
    measures = ['dice', 'hausdorff']
    n_clusters_list = [4]

    # Shape model free methods based on mask similarity alone
    for measure in measures:
        for n_clusters in n_clusters_list:

            # Output dir
            dir_msr_output = os.path.join(dir_output, f'{measure}_{n_clusters}')
            os.makedirs(dir_msr_output, exist_ok=True)
        
            # Outputs
            cluster_plot_png = os.path.join(dir_msr_output, f'kidney_cluster_plot.png')
            cluster_maps_json = os.path.join(dir_msr_output, f'kidney_cluster_labels.json')
            cluster_props_png = os.path.join(dir_msr_output, f'kidney_cluster_proportions.png')
            subject_cluster_maps_json = os.path.join(dir_msr_output, f'subject_cluster_labels.json')
            subject_cluster_heatmap_png = os.path.join(dir_msr_output, f'subject_cluster_heatmap.png')
            kidney_cluster_path = os.path.join(dir_msr_output, f'kidney_clusters')

            # Perform similarity clustering
            compute_similarity_clusters(matrix_csv[measure], cluster_maps_json, n_clusters=n_clusters)
            derive_patient_cluster_map(cluster_maps_json, subject_cluster_maps_json)

            # Display clusters
            plot_similarity_clustermap(matrix_csv[measure], cluster_plot_png, n_clusters=n_clusters, metric=measure, vmin=VRANGE[measure][0], vmax=VRANGE[measure][1])
            plot_cluster_proportions(cluster_maps_json, cluster_props_png, source=measure)
            plot_patient_symmetry_heatmap(subject_cluster_maps_json, subject_cluster_heatmap_png, source=measure)
            # plot_cluster_masks(cluster_maps_json, masks_path, kidney_cluster_path)

    for model in models:

        # Model level inputs
        dir_input = os.path.join(build, PIPELINE, f'stage_{STAGE[model]}_{model}_pca')
        scores_path = os.path.join(dir_input, f"{model}_scores.zarr")
         
        for n_clusters in n_clusters_list:
            
            # Output dir
            dir_clstr_output = os.path.join(dir_output, f'{model}_{n_clusters}')
            os.makedirs(dir_clstr_output, exist_ok=True)

            # Outputs
            pca_cluster_plot = os.path.join(dir_clstr_output, 'pca_cluster_plot.png')
            pca_json_path = os.path.join(dir_clstr_output, 'pca_cluster_labels.json')
            pca_subject_cluster_labels = os.path.join(dir_clstr_output, 'pca_subject_cluster_labels.json')
            pca_subject_cluster_heatmap = os.path.join(dir_clstr_output, 'pca_subject_cluster_heatmap.png')
            pca_cluster_scatter = os.path.join(dir_clstr_output, 'pca_cluster_scatter.png')
            pca_cluster_mosaic = os.path.join(dir_clstr_output, 'pca_cluster_mosaic.png')
            pca_cluster_msd_mosaic = os.path.join(dir_clstr_output, 'pca_cluster_similarity_mosaic.png')
            pca_cluster_props = os.path.join(dir_clstr_output, 'pca_cluster_proportions.png')
            pca_cluster_kidneys = os.path.join(dir_clstr_output, 'pca_clusters_kidneys')

            # Compute clusters
            compute_pca_clusters(scores_path, pca_json_path, n_components=N_COMP, n_clusters=n_clusters)
            derive_patient_cluster_map(pca_json_path, pca_subject_cluster_labels)

            # Display clusters
            plot_patient_symmetry_heatmap(pca_subject_cluster_labels, pca_subject_cluster_heatmap, source='PCA')
            plot_clusters(scores_path, pca_cluster_plot, n_components=N_COMP, n_clusters=n_clusters)
            plot_pca_scatter(scores_path, pca_cluster_scatter, n_components=N_COMP, n_clusters=n_clusters)
            plot_pca_mosaic(scores_path, pca_cluster_mosaic, n_components=N_COMP, n_clusters=n_clusters)
            plot_cluster_proportions(pca_json_path, pca_cluster_props, source='PCA')
            plot_pca_similarity_mosaic(scores_path, matrix_csv['dice'], pca_cluster_msd_mosaic, n_components=N_COMP, n_clusters=n_clusters, similarity='DICE')
            # plot_cluster_masks(pca_json_path, masks_path, pca_cluster_kidneys)

    for n_clusters in n_clusters_list:
        cluster_comparison = os.path.join(dir_output, f'comparison_{n_clusters}_clusters.png')
        json_files = {
            'CHEBY': os.path.join(dir_output, f'chebyshev_{n_clusters}', 'pca_cluster_labels.json'), 
            'SPECTRAL': os.path.join(dir_output, f'spectral_{n_clusters}', 'pca_cluster_labels.json'), 
            'DICE': os.path.join(dir_output, f'dice_{n_clusters}', f'kidney_cluster_labels.json'), 
            'HAUS': os.path.join(dir_output, f'hausdorff_{n_clusters}', f'kidney_cluster_labels.json'),
        }
        plot_cluster_comparison_mosaic(json_files, cluster_comparison)
        
    logging.info(f"Stage 14 --- Finished clustering kidneys ---")


def similarity_linkage(dice_csv_path):
    # 1. Load Data
    dice_df = pd.read_csv(dice_csv_path, index_col=0)
    
    # 2. Clustering
    row_linkage = linkage(dice_df, method='ward')
    
    return row_linkage


def compute_similarity_clusters(dice_csv_path, output_json_path=None, n_clusters=4):

    # 1. Load Data
    dice_df = pd.read_csv(dice_csv_path, index_col=0)
    labels = dice_df.index.astype(str).tolist()
    
    # 2. Clustering
    row_linkage = linkage(dice_df, method='ward')
    
    # 3. Extract Initial Cluster IDs
    initial_cluster_ids = fcluster(row_linkage, t=n_clusters, criterion='maxclust')

    # --- NEW STEP: Remap IDs based on Frequency ---
    # Count frequencies of the initial IDs
    counts = Counter(initial_cluster_ids)
    # Sort IDs by frequency (descending)
    # common_ids is a list of (id, count) tuples
    common_ids = counts.most_common() 
    
    # Create a mapping: {old_id: new_rank_id} where rank starts at 1
    rank_mapping = {old_id: i + 1 for i, (old_id, count) in enumerate(common_ids)}
    
    # Apply remapping
    cluster_ids = np.array([rank_mapping[cid] for cid in initial_cluster_ids])
    # ----------------------------------------------

    # 5. JSON Export (Sorted by the new IDs)
    cluster_map = {int(c): [] for c in range(1, n_clusters + 1)}
    for label, cid in zip(labels, cluster_ids):
        cluster_map[int(cid)].append(label)

    if output_json_path is not None:
        with open(output_json_path, 'w') as f:
            json.dump(cluster_map, f, indent=4)

    return cluster_ids, cluster_map



def compute_pca_clusters(zarr_path, output_json_path=None, n_components=200, n_clusters=4):
    """
    Computes clusters and saves them to JSON, remapped by frequency (1 = largest).
    """
    # 1. Open and Load
    root = zarr.open(zarr_path, mode='r')
    scores = root['scores'][:]
    labels = root['labels'][:]
    
    # Standardize labels (handle bytes if necessary)
    labels_str = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
    
    # 2. Clustering Calculation
    df_subset = pd.DataFrame(scores[:, :n_components])
    row_linkage = linkage(df_subset, method='ward')
    initial_ids = fcluster(row_linkage, t=n_clusters, criterion='maxclust')

    # --- NEW: Remap IDs based on Frequency (1 = Largest) ---
    counts = Counter(initial_ids)
    # most_common() returns [(id, count), (id, count), ...] sorted by count
    common_ids = counts.most_common() 
    
    # Create rank mapping {old_id: 1, next_old_id: 2, ...}
    rank_mapping = {old_id: i + 1 for i, (old_id, count) in enumerate(common_ids)}
    
    # Apply the mapping
    cluster_ids = [rank_mapping[cid] for cid in initial_ids]

    # 3. Build the Cluster Map
    # Initialize dictionary with keys in order 1..N to keep JSON tidy
    cluster_map = {int(c): [] for c in range(1, n_clusters + 1)}
    
    for label, cid in zip(labels_str, cluster_ids):
        cluster_map[cid].append(label)

    # 4. Export JSON
    if output_json_path is not None:
        with open(output_json_path, 'w') as f:
            json.dump(cluster_map, f, indent=4)
        
    return cluster_ids, cluster_map, row_linkage



def derive_patient_cluster_map(kidney_json_path, subject_clusters):
    """
    Groups kidneys by patient and assigns patients to (LeftCluster, RightCluster) tuples.
    
    Args:
        kidney_json_path (str): Path to the JSON containing {ClusterID: [KidneyLabels]}
        
    Returns:
        dict: A map where keys are (i, j) tuples and values are lists of PatientIDs.
    """
    # 1. Load the original kidney cluster map
    with open(kidney_json_path, 'r') as f:
        kidney_map = json.load(f)

    # 2. Invert the map to: {KidneyLabel: ClusterID}
    # We ensure ClusterID is an integer for the output tuples
    kidney_to_cluster = {}
    for cluster_id, labels in kidney_map.items():
        for label in labels:
            kidney_to_cluster[label] = int(cluster_id)

    # 3. Group kidneys by PatientID
    # patient_data structure: {PatientID: {'L': cluster_id, 'R': cluster_id}}
    patient_data = defaultdict(dict)
    
    for label, cluster_id in kidney_to_cluster.items():
        if '-' not in label:
            continue
            
        patient_id, side = label.rsplit('-', 1)
        if side.upper() == 'L':
            patient_data[patient_id]['L'] = cluster_id
        elif side.upper() == 'R':
            patient_data[patient_id]['R'] = cluster_id

    # 4. Map patients to (i, j) clusters
    # i = Left kidney cluster, j = Right kidney cluster
    patient_cluster_map = defaultdict(list)
    
    for patient_id, sides in patient_data.items():
        left_cluster = sides.get('L', -1)
        right_cluster = sides.get('R', -1)
        
        cluster_coords = f"{(left_cluster, right_cluster)}"
        patient_cluster_map[cluster_coords].append(patient_id)

    # Convert defaultdict back to standard dict for return/JSON export
    patient_cluster_map = dict(patient_cluster_map)
    with open(subject_clusters, 'w') as f:
        json.dump(patient_cluster_map, f, indent=4)



def plot_cluster_masks(cluster_json_path, masks_path, cluster_movie_path):

    # Read the labels
    with open(cluster_json_path, 'r') as f:
        kidney_map = json.load(f)

    # Loop over the clusters
    for cluster, kidney_labels in tqdm(kidney_map.items(), desc='Building cluster display'):

        # Get a list of masks for the kidneys in the cluster
        masks = ssa.masks_from_zarr(masks_path, kidney_labels)

        # Create the movies
        dir_png = os.path.join(cluster_movie_path, f"images_cluster_{cluster}")
        os.makedirs(dir_png, exist_ok=True)
        ncols, nrows = 16, 8
        pvplot.rotating_mosaics_da(dir_png, masks, kidney_labels, chunksize=ncols * nrows, nviews=25, columns=ncols, rows=nrows)
        mp4.images_to_video(dir_png, os.path.join(cluster_movie_path, f"movie_cluster_{cluster}.mp4"), fps=16)


def plot_patient_symmetry_heatmap(subject_clusters_json, output_image_path, source='PCS'):
    """
    Loads patient clusters from JSON and plots a symmetry heatmap.
    Handles stringified tuples like "(1, 2)" and "-1" for missing kidneys.
    """
    # 1. Load the patient-level cluster map
    with open(subject_clusters_json, 'r') as f:
        patient_cluster_map = json.load(f)

    # 2. Parse keys and determine the range of clusters
    parsed_data = {}
    all_cluster_ids = set()
    
    for coord_str, patients in patient_cluster_map.items():
        # Convert string "(i, j)" or "(i, -1)" back to a tuple of ints
        # We strip parentheses and split by comma
        coords = tuple(map(int, coord_str.replace('(', '').replace(')', '').split(',')))
        parsed_data[coords] = len(patients)
        
        # Track unique IDs (excluding -1) to find n_clusters
        for c in coords:
            if c != -1:
                all_cluster_ids.add(c)
    
    n_clusters = max(all_cluster_ids) if all_cluster_ids else 0
    
    # 3. Define axis labels (-1 followed by 1...N)
    cluster_range = [-1] + sorted(list(all_cluster_ids))
    n_dim = len(cluster_range)
    
    # 4. Initialize and fill the matrix
    matrix = np.zeros((n_dim, n_dim))
    for (l_cid, r_cid), count in parsed_data.items():
        row_idx = cluster_range.index(l_cid)
        col_idx = cluster_range.index(r_cid)
        matrix[row_idx, col_idx] = count

    # 5. Plotting
    plt.figure(figsize=(10, 8))
    tick_labels = [f"C{c}" if c != -1 else "None" for c in cluster_range]
    
    sns.heatmap(
        matrix, 
        annot=True, 
        fmt=".0f", 
        cmap="YlOrRd", 
        xticklabels=tick_labels, 
        yticklabels=tick_labels,
        square=True,
        cbar_kws={'label': 'Number of Patients'}
    )
    
    # 6. Aesthetics
    plt.title(f"Patient-Level Symmetry: Left vs Right Kidney {source} Clusters", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Right Kidney Cluster", fontsize=12)
    plt.ylabel("Left Kidney Cluster", fontsize=12)
    
    # Add a visual guide for the symmetry diagonal (starting from Cluster 1)
    if n_clusters > 0:
        plt.plot([1, n_dim], [1, n_dim], color='blue', linestyle='--', alpha=0.5)

    # 7. Save and Show
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.show()
    plt.close()


def plot_cluster_comparison_mosaic(paths, output_image_path):
    """
    Creates a N-panel figure comparing PCA, DICE, and HAUSDORFF clusters.
    Calculates ARI/NMI for each pair.
    """
    # 1. Load all JSONs
    maps = {}
    for name, path in paths.items():
        with open(path, 'r') as f:
            maps[name] = json.load(f)

    # 2. Helper to get aligned labels
    def get_aligned_data(map_a, map_b):
        rev_a = {label: int(cid) for cid, labels in map_a.items() for label in labels}
        rev_b = {label: int(cid) for cid, labels in map_b.items() for label in labels}
        common = sorted(list(set(rev_a.keys()) & set(rev_b.keys())))
        y_a = [rev_a[sub] for sub in common]
        y_b = [rev_b[sub] for sub in common]
        return y_a, y_b, len(common)

    # 3. Define the comparisons
    methods = list(paths.keys())
    n = len(methods)
    comparisons = [(methods[i], methods[(i + 1) % n]) for i in range(n)]

    # 4. Setup Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    
    for ax, (name_a, name_b) in zip(axes, comparisons):
        y_a, y_b, n_total = get_aligned_data(maps[name_a], maps[name_b])
        
        if n_total == 0:
            ax.set_title(f"No overlap: {name_a} vs {name_b}")
            continue

        # Metrics
        ari = adjusted_rand_score(y_a, y_b)
        nmi = normalized_mutual_info_score(y_a, y_b)

        # Build Matrix
        ids_a = sorted([int(k) for k in maps[name_a].keys()])
        ids_b = sorted([int(k) for k in maps[name_b].keys()])
        
        matrix = np.zeros((len(ids_a), len(ids_b)))
        for val_a, val_b in zip(y_a, y_b):
            matrix[ids_a.index(val_a), ids_b.index(val_b)] += 1
        
        matrix_pct = (matrix / n_total) * 100

        # Plot Heatmap
        sns.heatmap(
            matrix_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax,
            xticklabels=[f"C{i}" for i in ids_b], 
            yticklabels=[f"C{i}" for i in ids_a],
            vmin=0, vmax=max(50, matrix_pct.max()),
            cbar_kws={'label': '% of Population'}
        )

        ax.set_title(f"{name_a} vs {name_b}\nARI: {ari:.3f} | NMI: {nmi:.3f}", 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel(f"{name_b} Clusters", fontsize=12)
        ax.set_ylabel(f"{name_a} Clusters", fontsize=12)

    # 5. Final Touches
    plt.suptitle("Inter-Metric Cluster Agreement Analysis", fontsize=22, y=1.08, fontweight='bold')
    plt.tight_layout()
    
    # 6. Save and Show
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cluster_proportions(cluster_json_path, output_image_path, source='DICE'):
    """
    Visualizes cluster distribution with labels above every bar.
    Maintains cluster colors with subtle dark/light hues for L/R.
    """
    # 1. Load Cluster Data
    with open(cluster_json_path, 'r') as f:
        cluster_map = json.load(f)
    
    # 2. Global Totals
    total_subjects = sum(len(labels) for labels in cluster_map.values())
    all_labels = [label for labels in cluster_map.values() for label in labels]
    global_left_total = sum(1 for l in all_labels if l.endswith('L'))
    global_right_total = sum(1 for l in all_labels if l.endswith('R'))
    
    # 3. Color Logic
    def adjust_lightness(color, amount=1.0):
        rgb = mcolors.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        return colorsys.hls_to_rgb(h, max(0, min(1, l * amount)), s)

    n_clusters = len(cluster_map)
    base_palette = sns.color_palette("Set2", n_clusters)
    
    data = []
    sorted_keys = sorted(cluster_map.keys(), key=int)
    
    for i, cluster_id in enumerate(sorted_keys):
        labels = cluster_map[cluster_id]
        count = len(labels)
        left_count = sum(1 for l in labels if l.endswith('L'))
        right_count = sum(1 for l in labels if l.endswith('R'))
        
        data.append({
            'Cluster': f"C{cluster_id}",
            'Pct Total': (count / total_subjects * 100),
            'Pct Left In Cluster': (left_count / count * 100) if count > 0 else 0,
            'Pct Right In Cluster': (right_count / count * 100) if count > 0 else 0,
            'Pct Global Left': (left_count / global_left_total * 100) if global_left_total > 0 else 0,
            'Pct Global Right': (right_count / global_right_total * 100) if global_right_total > 0 else 0,
            'BaseColor': base_palette[i],
            'DarkColor': adjust_lightness(base_palette[i], 0.85),
            'LightColor': adjust_lightness(base_palette[i], 1.15)
        })

    df = pd.DataFrame(data)

    # 4. Create Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    x = np.arange(len(df['Cluster']))
    width = 0.38 

    # Helper function for adding labels
    def add_labels(ax, bars, fmt='%.1f%%'):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    fmt % height, ha='center', va='bottom', fontsize=9)

    # --- Panel 1: Population Distribution ---
    bars1 = ax1.bar(df['Cluster'], df['Pct Total'], color=df['BaseColor'], edgecolor='black', alpha=0.9)
    ax1.set_title("Population Distribution", fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel("% of Total Population")
    ax1.set_ylim(0, max(df['Pct Total']) + 15)
    add_labels(ax1, bars1)

    # --- Panel 2: Global Distribution ---
    for i in range(len(df)):
        bL = ax2.bar(x[i] - width/2, df.loc[i, 'Pct Global Left'], width, 
                     color=df.loc[i, 'DarkColor'], edgecolor='black', alpha=0.9,
                     label='Left (L)' if i == 0 else "")
        bR = ax2.bar(x[i] + width/2, df.loc[i, 'Pct Global Right'], width, 
                     color=df.loc[i, 'LightColor'], edgecolor='black', alpha=0.9,
                     label='Right (R)' if i == 0 else "")
        add_labels(ax2, bL)
        add_labels(ax2, bR)
        
    ax2.set_title("Global Distribution\n(% of Total L/R kidneys)", fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, max(max(df['Pct Global Left']), max(df['Pct Global Right'])) + 15)
    ax2.legend(title="Kidney Side", loc='upper left')

    # --- Panel 3: Internal Cluster Bias ---
    for i in range(len(df)):
        bL = ax3.bar(x[i] - width/2, df.loc[i, 'Pct Left In Cluster'], width, 
                     color=df.loc[i, 'DarkColor'], edgecolor='black', alpha=0.9)
        bR = ax3.bar(x[i] + width/2, df.loc[i, 'Pct Right In Cluster'], width, 
                     color=df.loc[i, 'LightColor'], edgecolor='black', alpha=0.9)
        add_labels(ax3, bL)
        add_labels(ax3, bR)

    ax3.set_title("Internal Cluster Bias\n(% L/R within Cluster)", fontsize=14, fontweight='bold', pad=20)
    ax3.set_ylim(0, 120)
    ax3.axhline(50, color='black', linestyle='--', alpha=0.3)

    # 5. Formatting
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(df['Cluster'])
        ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    plt.suptitle(f"{source} Cluster Analysis Dashboard (N={total_subjects})", 
                   fontsize=20, y=1.02, fontweight='bold')
    plt.tight_layout()
    
    # 6. Save and Show
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_similarity_clustermap(dice_csv_path, output_image_path, 
                         n_clusters=4, cmap='viridis', metric='DICE', vmin=None, vmax=None):
    """
    Clusters a Dice similarity matrix.
    Remaps cluster IDs from 1 (most common) to N (least common).
    """
    # 1. Load Data
    row_linkage = similarity_linkage(dice_csv_path)

    # 4. Colors
    palette = sns.color_palette("Set2", n_clusters).as_hex()
    set_link_color_palette(palette)

    dice_df = pd.read_csv(dice_csv_path, index_col=0)
    # 6. Generate Clustermap
    g = sns.clustermap(
        dice_df,
        row_linkage=row_linkage,
        col_linkage=row_linkage,
        row_cluster=True,
        col_cluster=True,
        cmap=cmap, 
        figsize=(12, 12),
        xticklabels=False,
        yticklabels=False,
        vmin=vmin, 
        vmax=vmax,  
        dendrogram_ratio=(0.0, 0.15),
        cbar=False 
    )

    if g.ax_cbar is not None:
        g.ax_cbar.remove()

    # 7. Manually Redraw Column Dendrogram
    # Note: Thresholding for dendrogram coloring might need to be specific to the linkage,
    # but the rank-mapping ensures the JSON and visual labels align.
    threshold = row_linkage[-(n_clusters - 1), 2] if n_clusters > 1 else 0
    g.ax_col_dendrogram.clear()
    dendrogram(row_linkage, ax=g.ax_col_dendrogram, color_threshold=threshold, 
               no_labels=True, above_threshold_color='grey')
    g.ax_col_dendrogram.set_axis_off()
    g.ax_row_dendrogram.set_visible(False)

    # 8. Create a Manual Colorbar Axis
    cbar_ax = g.fig.add_axes([0.04, 0.3, 0.02, 0.2]) 
    mappable = g.ax_heatmap.collections[0]
    cb = plt.colorbar(mappable, cax=cbar_ax)
    cb.set_label(f"{metric} Score", fontsize=10, labelpad=10)

    # 9. Force Margins
    g.fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.05)
    g.fig.suptitle(f"{metric} Similarity Clustermap\n{n_clusters} Clusters (Sorted by Size)", 
                   fontsize=18, fontweight='bold', y=0.96)
    
    plt.savefig(output_image_path, dpi=300, bbox_inches=None)
    plt.show()
    plt.close(g.fig)

    set_link_color_palette(None)


def plot_pca_similarity_mosaic(pca_zarr_path, dice_csv_path, output_image_path, 
                    k=4, n_components=200, n_clusters=4, similarity='DICE'):
    """
    Computes clusters from PCA scores, remaps them by frequency (1=largest),
    and visualizes the MDS projection of the Dice matrix.
    """
    # 1. Load PCA data
    root = zarr.open(pca_zarr_path, mode='r')
    labels = root['labels'][:]
    labels_str = [l.decode() if isinstance(l, bytes) else str(l) for l in labels]
    
    # 2. Compute clusters
    cluster_ids, _, _ = compute_pca_clusters(pca_zarr_path, n_components=n_components, n_clusters=n_clusters)

    # 3. Load and Align Dice Matrix
    dice_df = pd.read_csv(dice_csv_path, index_col=0)
    try:
        dice_df = dice_df.loc[labels_str, labels_str]
    except KeyError as e:
        print(f"Warning: Alignment mismatch: {e}")
        
    dice_matrix = dice_df.values
    dist_matrix = 1.0 - dice_matrix
    
    # 4. Perform MDS
    mds = MDS(n_components=k, dissimilarity='precomputed', random_state=42, n_init=4)
    mds_coords = mds.fit_transform(dist_matrix) 
    
    # 5. Setup Figure
    fig, axes = plt.subplots(k, k, figsize=(k*3.5, k*3.5), squeeze=False)
    palette = sns.color_palette("Set2", n_clusters)

    # 6. Fill the Grid
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            
            if i == j:
                sns.histplot(
                    x=mds_coords[:, i], hue=cluster_ids, palette=palette, 
                    element="step", ax=ax, legend=False, stat="percent",
                    common_norm=False, hue_order=range(1, n_clusters + 1)
                )
                ax.set_title(f"MDS Dim {i+1}", fontsize=10, fontweight='bold')
            else:
                sns.scatterplot(
                    x=mds_coords[:, j], y=mds_coords[:, i], 
                    hue=cluster_ids, palette=palette,
                    hue_order=range(1, n_clusters + 1),
                    s=30, alpha=0.7, edgecolor=None, legend=False, ax=ax
                )
            
            if i == k - 1: ax.set_xlabel(f"MDS Dimension {j+1}")
            if j == 0 and i != 0: ax.set_ylabel(f"MDS Dimension {i+1}")

    # 7. Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {c}',
                              markerfacecolor=palette[c-1], markersize=10) 
                       for c in range(1, n_clusters + 1)]
    
    fig.legend(handles=legend_elements, title="Cluster Rank (Size)", 
               loc='center', bbox_to_anchor=(0.7, 0.7), frameon=True)
    
    fig.suptitle(f"PCA clusters - Projection of {similarity} Matrix", fontsize=18, y=0.98)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_pca_mosaic(pca_zarr_path, output_image_path, k=4, n_components=200, n_clusters=4):
    """
    Creates a lower-triangle PCA mosaic with clusters remapped by size.
    Cluster 1 = Most common, Cluster N = Least common.
    """
    # 1. Load data
    root = zarr.open(pca_zarr_path, mode='r')
    scores = root['normalized_scores'][:]
    var_ratio = root['variance_ratio'][:] * 100

    cluster_ids, _, _ = compute_pca_clusters(pca_zarr_path, n_components=n_components, n_clusters=n_clusters)

    # 3. Setup Figure
    fig, axes = plt.subplots(k, k, figsize=(k*3.5, k*3.5), squeeze=False)
    palette = sns.color_palette("Set2", n_clusters)
    cluster_order = list(range(1, n_clusters + 1))

    # 4. Fill the Grid
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            
            # Remove plots on the upper triangle
            if j > i:
                ax.axis('off')
                continue
            
            if i == j:
                # Diagonal: Histogram
                sns.histplot(
                    x=scores[:, i], hue=cluster_ids, palette=palette, 
                    hue_order=cluster_order,
                    element="step", ax=ax, legend=False, stat="percent",
                    common_norm=False 
                )
                ax.set_ylabel("Percent (%)", fontsize=9)
                ax.set_title(f"PC{i+1} Distribution", fontsize=10, fontweight='bold')
                ax.yaxis.set_tick_params(labelleft=True) 
            
            else:
                # Off-diagonal: Scatter
                sns.scatterplot(
                    x=scores[:, j], y=scores[:, i], 
                    hue=cluster_ids, palette=palette,
                    hue_order=cluster_order,
                    s=20, alpha=0.5, edgecolor=None, legend=False, ax=ax
                )
            
            # Labels
            if i == k - 1:
                ax.set_xlabel(f"PC{j+1}\n({var_ratio[j]:.1f}%)", fontsize=10)
            if j == 0 and i != 0:
                ax.set_ylabel(f"PC{i+1}\n({var_ratio[i]:.1f}%)", fontsize=10)

    # 5. Position Legend in the Upper Triangle
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {c}',
                              markerfacecolor=palette[c-1], markersize=10) 
                       for c in cluster_order]
    
    fig.legend(handles=legend_elements, title="Cluster Rank (Size)", 
               loc='center', 
               bbox_to_anchor=(0.7, 0.7), 
               fontsize=12, title_fontsize=14, frameon=True)
    
    fig.suptitle(f"PCA Normalized Mosaic (Top {k} PCs)\nSorted by Cluster Population", 
                 fontsize=18, y=0.98, fontweight='bold')

    # 6. Save and Show
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_pca_scatter(pca_zarr_path, output_image_path, n_components=200, n_clusters=4):
    """
    Plots PC1 vs PC2 with clusters remapped by size (1 = largest).
    Labels include variance explained for each component.
    """
    # 1. Load data
    root = zarr.open(pca_zarr_path, mode='r')
    scores = root['scores'][:]
    var_ratio = root['variance_ratio'][:] * 100
    
    # 2. Recompute Clusters
    cluster_ids, _, _ = compute_pca_clusters(pca_zarr_path, n_components=n_components, n_clusters=n_clusters)

    # 3. Plotting
    plt.figure(figsize=(11, 7))
    palette = sns.color_palette("Set2", n_clusters)
    cluster_order = list(range(1, n_clusters + 1))
    
    scatter = sns.scatterplot(
        x=scores[:, 0], 
        y=scores[:, 1], 
        hue=cluster_ids, 
        palette=palette,
        hue_order=cluster_order, # Forces legend and colors to follow rank 1..N
        legend='full',
        alpha=0.8,
        edgecolor='w',
        s=100
    )

    # 4. Labels and Titles
    plt.xlabel(f"PC1 ({var_ratio[0]:.2f}% Variance)", fontsize=12, fontweight='bold')
    plt.ylabel(f"PC2 ({var_ratio[1]:.2f}% Variance)", fontsize=12, fontweight='bold')
    
    plt.title(f"PCA Projection: PC1 vs PC2\nClusters Ranked by Size (1 = Largest)", 
              fontsize=15, pad=15)
    
    # Position legend outside the plot
    plt.legend(title="Cluster Rank", bbox_to_anchor=(1.02, 1), loc='upper left', 
               borderaxespad=0, title_fontsize=12, fontsize=11)
    
    # 5. Save and Show
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return cluster_ids


def plot_clusters(zarr_path, output_image_path, 
                  n_components=200, n_display=100, cmap='vlag', 
                  n_clusters=4):
    """
    Visualizes PCA-based subject clustering using the shared compute_pca_clusters logic.
    Ensures dendrogram branch colors match frequency-ranked cluster IDs.
    """
    # 1. Reuse your clustering logic to get ranked IDs
    # Note: We don't save a JSON here, just use the returned IDs
    cluster_ids, _, row_linkage = compute_pca_clusters(
        zarr_path, n_components=n_components, n_clusters=n_clusters
    )
    
    # 2. Open and Load data for visualization
    root = zarr.open(zarr_path, mode='r')
    labels = root['labels'][:]
    normalized_scores = root['normalized_scores'][:]
    
    # 3. Align the Palette
    # Cluster Dashboard uses Set2[0] for C1 (Largest). 
    # We need to tell the dendrogram which internal branches correspond to C1.
    base_palette = sns.color_palette("Set2", n_clusters).as_hex()
    
    # Calculate which internal (linkage) ID maps to which Frequency Rank
    # We use fcluster here purely to identify the linkage's internal ID numbers
    from scipy.cluster.hierarchy import fcluster
    internal_ids = fcluster(row_linkage, t=n_clusters, criterion='maxclust')
    
    # Find the mapping from Internal ID -> Frequency Rank
    # We look at the first subject belonging to each rank to find its internal ID
    internal_to_rank = {}
    for i, rank in enumerate(cluster_ids):
        iid = internal_ids[i]
        if iid not in internal_to_rank:
            internal_to_rank[iid] = rank # Rank is 1-based from your function
            
    # Reorder the palette: index 0 of the link palette corresponds to internal ID 1
    # We need to ensure the 'i-th' color in set_link_color_palette is the color for internal ID 'i+1'
    reordered_palette = [None] * n_clusters
    for iid, rank in internal_to_rank.items():
        reordered_palette[iid - 1] = base_palette[rank - 1]
    
    set_link_color_palette(reordered_palette)

    # 4. Generate Clustermap
    df_display = pd.DataFrame(normalized_scores[:, :n_display], index=labels)
    
    g = sns.clustermap(
        df_display.T,
        row_linkage=None,
        col_linkage=row_linkage,
        row_cluster=False,
        col_cluster=True,
        cmap=cmap, 
        figsize=(12, 8),
        xticklabels=False,
        yticklabels=False,
        vmin=-2, 
        vmax=2,  
        center=0,
        dendrogram_ratio=(0.1, 0.2), 
        cbar_pos=(0.025, 0.4, 0.02, 0.2)
    )

    # 5. Manually Re-draw Dendrogram with Correct Threshold
    threshold = row_linkage[-(n_clusters - 1), 2] if n_clusters > 1 else 0
    g.ax_col_dendrogram.clear() 
    dendrogram(row_linkage, 
               ax=g.ax_col_dendrogram, 
               color_threshold=threshold, 
               no_labels=True, 
               above_threshold_color='grey')
    
    g.ax_col_dendrogram.set_axis_off()

    # 6. Aesthetics
    cbar = g.ax_cbar
    cbar.set_title("Z-Score", fontsize=10)
    cbar.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    g.fig.suptitle(f"Subject Clustering (N={n_clusters})\nRanked by Population Size (C1=Largest)", 
                   y=1.02, fontsize=16, fontweight='bold')
    
    g.ax_heatmap.set_ylabel(f"Top {n_display} Components", fontsize=12)
    g.ax_heatmap.set_xlabel(f"Subjects (n={len(labels)})", fontsize=12)

    # 7. Save and Clean up
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    set_link_color_palette(None) # Always reset global state

# def plot_clusters(zarr_path, output_image_path, 
#                   n_components=200, n_display=100, cmap='vlag', 
#                   n_clusters=4):
#     """
#     Visualizes PCA-based subject clustering with aligned color coding.
#     """
#     # 1. Open and Load
#     root = zarr.open(zarr_path, mode='r')
#     labels = root['labels'][:]
#     normalized_scores = root['normalized_scores'][:]
    
#     df_display_subset = pd.DataFrame(normalized_scores[:, :n_display], index=labels)

#     # 2. Clustering Calculation
#     row_linkage = pca_linkage(zarr_path, n_components)
    
#     # 3. Align Palette with Frequency Ranking
#     # Get internal IDs (1 to N)
#     internal_ids = fcluster(row_linkage, t=n_clusters, criterion='maxclust')
    
#     # Map internal IDs to their frequency rank
#     counts = Counter(internal_ids)
#     # common_ids is [(internal_id, count), ...] sorted by count descending
#     common_ids = counts.most_common()
    
#     # rank_mapping: {internal_id: rank_index_0_to_N}
#     rank_mapping = {old_id: i for i, (old_id, count) in enumerate(common_ids)}
    
#     # Create the ranked palette: Set2[0] = Most common, Set2[1] = 2nd most, etc.
#     base_palette = sns.color_palette("Set2", n_clusters).as_hex()
    
#     # Reorder the palette so dendrogram's 'C1' matches Rank 1, etc.
#     # Scipy uses colors in the order they appear in the linkage tree.
#     # We assign the base_palette colors to the internal cluster labels.
#     reordered_palette = [None] * n_clusters
#     for internal_id, rank in rank_mapping.items():
#         reordered_palette[internal_id - 1] = base_palette[rank]
        
#     set_link_color_palette(reordered_palette)

#     # 4. Generate Clustermap
#     g = sns.clustermap(
#         df_display_subset.T,
#         row_linkage=None,
#         col_linkage=row_linkage,
#         row_cluster=False,
#         col_cluster=True,
#         cmap=cmap, 
#         figsize=(12, 8),
#         xticklabels=False,
#         yticklabels=False,
#         vmin=-2, 
#         vmax=2,  
#         center=0,
#         dendrogram_ratio=(0.1, 0.2), 
#         cbar_pos=(0.025, 0.4, 0.02, 0.2)
#     )

#     # 5. Manually Re-draw Dendrogram
#     # The 'color_threshold' ensures only the top n_clusters branches are colored
#     threshold = row_linkage[-(n_clusters - 1), 2] if n_clusters > 1 else 0
#     g.ax_col_dendrogram.clear() 
#     dendrogram(row_linkage, 
#                ax=g.ax_col_dendrogram, 
#                color_threshold=threshold, 
#                no_labels=True, 
#                above_threshold_color='grey')
    
#     g.ax_col_dendrogram.set_axis_off()

#     # 6. Aesthetics
#     cbar = g.ax_cbar
#     cbar.set_title("Z-Score", fontsize=10)
#     cbar.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

#     g.fig.suptitle(f"Subject Clustering: {n_clusters} Identified Groups\n(Ranked by Population Size)", 
#                    y=1.05, fontsize=16, fontweight='bold')
    
#     g.ax_heatmap.yaxis.set_label_position("right")
#     g.ax_heatmap.set_ylabel(f"Top {n_display} Principal Components", fontsize=12)
#     g.ax_heatmap.set_xlabel(f"Subjects (n={len(labels)})", fontsize=12)

#     # 7. Save and Show
#     plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
#     plt.show()

#     # Reset palette to default for other plots
#     set_link_color_palette(None)




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_dask_script(run, BUILD, PIPELINE)