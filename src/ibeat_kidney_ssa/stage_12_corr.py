import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG – update paths to your actual files
# ─────────────────────────────────────────────────────────────────────────────
FILES = {
    "Legendre NonLinear": "table_scores_non_linear_legendre.csv",
    "Legendre":    "table_scores_legendre.csv",
    "Spectral NonLinear": "table_scores_non_linear_spectral.csv",
    "Spectral":    "table_scores_spectral.csv",
}
PC_COLS   = [f"PC_{i}" for i in range(1, 21)]
LABEL_COL = "label"

PANELS = [
    ("Legendre NonLinear", "Spectral NonLinear"),
    ("Legendre NonLinear", "Legendre"),
    ("Legendre",    "Spectral"),
    ("Spectral NonLinear", "Spectral"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & align
# ─────────────────────────────────────────────────────────────────────────────
dfs = {}
for name, path in FILES.items():
    df = pd.read_csv(path).set_index(LABEL_COL)   # ← changed from read_excel
    dfs[name] = df[PC_COLS]

common = dfs[list(FILES.keys())[0]].index
for name in list(FILES.keys())[1:]:
    common = common.intersection(dfs[name].index)
print(f"Shared samples: {len(common)}")
for name in FILES:
    dfs[name] = dfs[name].loc[common]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build 20×20 absolute correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
def abs_corr_matrix(df_a, df_b):
    n = len(PC_COLS)
    mat = np.zeros((n, n))
    for i, pc_i in enumerate(PC_COLS):
        for j, pc_j in enumerate(PC_COLS):
            r, _ = pearsonr(df_a[pc_i].values, df_b[pc_j].values)
            mat[i, j] = abs(r)
    return mat

# ─────────────────────────────────────────────────────────────────────────────
# 3. Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.subplots_adjust(hspace=0.35, wspace=0.35)

pc_labels = [f"PC_{i}" for i in range(1, 21)]
cmap = "RdBu_r"

for ax, (m_y, m_x) in zip(axes.flat, PANELS):
    mat = abs_corr_matrix(dfs[m_y], dfs[m_x])

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Circle on best-matching PC per row
    best_cols = np.argmax(mat, axis=1)
    for row_idx, col_idx in enumerate(best_cols):
        ax.add_patch(plt.Circle(
            (col_idx, row_idx), radius=0.42,
            fill=False, edgecolor="black", linewidth=1.5, zorder=3
        ))

    ax.set_xticks(range(20))
    ax.set_xticklabels(pc_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(20))
    ax.set_yticklabels(pc_labels, fontsize=7)
    ax.set_xlabel(m_x, fontsize=9, labelpad=6)
    ax.set_ylabel(m_y, fontsize=9, labelpad=6)
    ax.set_title(f"{m_y} vs {m_x}", fontsize=11, pad=8)

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Absolute correlation", fontsize=10)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.savefig("pc_cross_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pc_cross_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Export best-match table
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for m_y, m_x in PANELS:
    mat = abs_corr_matrix(dfs[m_y], dfs[m_x])
    best_cols = np.argmax(mat, axis=1)
    best_vals = mat[np.arange(20), best_cols]
    for i, (col_idx, val) in enumerate(zip(best_cols, best_vals)):
        rows.append({
            "Comparison":     f"{m_y} vs {m_x}",
            "PC_row":         PC_COLS[i],
            "Best_match_col": PC_COLS[col_idx],
            "Abs_r":          round(val, 4),
        })

pd.DataFrame(rows).to_csv("best_pc_matches.csv", index=False)   # ← also CSV
print("Saved: best_pc_matches.csv")