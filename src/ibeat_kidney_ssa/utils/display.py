import numpy as np
import pandas as pd
import zarr

from miblab_plot import pvplot, mp4

def get_outlier_labels(data_path: str, n: int = None, column_idx: int = 0):
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

    if n is None:
        n = len(df)  # Return all if n is not specified
    
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


def recon_err(
    mask_zarr_path=None, 
    dir_png=None, 
    movie_file=None, 
    n_samples=None, 
    n_components=None,
):
    recon = zarr.open(mask_zarr_path, mode='r')
    if n_samples is None:
        masks = recon['masks'][:]   
    else:
        masks = recon['masks'][:n_samples, ...]
    cols = recon.attrs['saved_steps'][:]
    if n_components is not None:
        idx = np.r_[:n_components+1, -1]
        masks = masks[:, idx, ...]
        cols = np.array(cols)[idx].tolist()
    masks = masks.transpose(1, 0, 2, 3, 4)
    n_rows = masks.shape[1] 
    labels = np.array([[f"K{y}: {x}" for y in range(n_rows)] for x in cols])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    mp4.images_to_video(dir_png, movie_file, fps=16)


def modes(
    modes_zarr_path: str = None, 
    dir_png: str = None, 
    movie_file: str = None,
):
    modes = zarr.open(modes_zarr_path, mode='r')
    masks = modes['masks'][:]
    n_rows = masks.shape[1]
    cols = np.array(modes.attrs['coeffs'][:])
    labels = np.array([[f"C{y}: {round(x, 1)} x sd" for y in range(n_rows)] for x in cols])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    mp4.images_to_video(dir_png, movie_file, fps=16)