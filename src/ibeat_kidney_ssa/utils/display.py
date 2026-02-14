import numpy as np
import pandas as pd
import zarr

from miblab_plot import pvplot, mp4

def get_outlier_labels(data_path: str, n: int = 5, column_idx: int = 0):
    """
    Identifies the N samples with the highest MSE values in a specific column.
    
    Args:
        data_path: Path to the MSE CSV.
        n: Number of top outliers to return.
        column_idx: The index of the column to sort by. 
                    Default is 0 (usually the 'Average' or first reconstruction step).
    
    Returns:
        List of labels (patient IDs) for the top N outliers.
    """
    # Load with index_col=0 so labels are the index
    df = pd.read_csv(data_path, index_col=0)
    
    # Select the target column and sort descending
    # .nlargest(n) is the most efficient way to get the top N in pandas
    top_n_series = df.iloc[:, column_idx].nlargest(n)
    
    outliers = top_n_series.index.tolist()
    
    print(f"Top {n} outliers in column '{df.columns[column_idx]}':")
    for label, value in top_n_series.items():
        print(f"  - {label}: {value:.6f}")
        
    return outliers


def recon(mask_path, dir_png, movie_file):
    recon = zarr.open(mask_path, mode='r')
    masks = recon['masks']
    labels = recon['labels'][:]
    ncols, nrows = 16, 8
    pvplot.rotating_mosaics_da(dir_png, masks, labels, chunksize=ncols * nrows, nviews=25, columns=ncols, rows=nrows)
    mp4.images_to_video(dir_png, movie_file, fps=16)


def recon_err(mask_path, dir_png, movie_file):
    recon = zarr.open(mask_path, mode='r')
    masks = recon['masks'][:].transpose(1, 0, 2, 3, 4)
    n_rows = masks.shape[1]
    cols = recon.attrs['saved_steps'][:]
    labels = np.array([[f"K{y}: {x}" for y in range(n_rows)] for x in cols])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    mp4.images_to_video(dir_png, movie_file, fps=16)


def modes(modes_path, dir_png, movie_file):
    modes = zarr.open(modes_path, mode='r')
    masks = modes['masks'][:]
    n_rows = masks.shape[1]
    cols = np.array(modes.attrs['coeffs'][:])
    labels = np.array([[f"C{y}: {round(x, 1)} x sd" for y in range(n_rows)] for x in cols])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    mp4.images_to_video(dir_png, movie_file, fps=16)