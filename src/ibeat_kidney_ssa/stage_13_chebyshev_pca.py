import os
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zarr

from ibeat_kidney_ssa.utils import pipe, sdf_cheby, pvplot, movie, ssa

PIPELINE = 'kidney_ssa'

def run(build):
    # Set the stage
    dir_input = os.path.join(build, PIPELINE, 'stage_9_stack_normalized')
    dir_output = pipe.setup_stage(build, PIPELINE, __file__)

    logging.info("Stage 13 --- Chebyshev PCA ---")

    # Input data - stack of normalized masks
    masks_path = os.path.join(dir_input, 'normalized_kidney_masks.zarr')

    # Build feature matrix
    feature_path = os.path.join(dir_output, 'chebyshev_features.zarr')
    ssa.features_from_dataset_zarr(
        sdf_cheby.features_from_mask, 
        masks_path, 
        feature_path, 
        order=16,
    )

    # Build principal components and plot variance explained
    pca_path = os.path.join(dir_output, f"chebyshev_components.zarr")
    var = ssa.pca_from_features_zarr(feature_path, pca_path)
    plot_file = os.path.join(dir_output, f"chebyshev_explained_variance.png")
    plot_explained_variance(plot_file, var)

    # Project kidneys along principal components (note include in step 1)
    coeffs_path = os.path.join(dir_output, f"chebyshev_coefficients.zarr")
    ssa.coefficients_from_features_zarr(feature_path, pca_path, coeffs_path)

    # Generate synthetic kidneys along the principal components
    modes_path = os.path.join(dir_output, f"chebyshev_modes.zarr")
    ssa.modes_from_pca_zarr(
        sdf_cheby.mask_from_features, 
        pca_path, 
        modes_path, 
        n_components=8, 
        max_coeff=4,
    )

    # Display principal modes
    dir_png = os.path.join(dir_output, 'images')
    movie_file = os.path.join(dir_output, 'chebyshev_modes.mp4')
    display_modes(modes_path, dir_png, movie_file)

    logging.info("Stage 12: Chebyshev PCA successfully completed.")


def display_modes(modes_path, dir_png, movie_file):
    modes = zarr.open(modes_path, mode='r')
    masks = modes['modes'][:]
    n_comp = masks.shape[1]
    coeffs = np.array(modes.attrs['coeffs'][:])
    labels = np.array([[f"C{y}: {round(x, 1)} x sd" for y in range(n_comp)] for x in coeffs])
    pvplot.rotating_masks_grid(dir_png, masks, labels, nviews=25)
    movie.images_to_video(dir_png, movie_file, fps=16)


def plot_explained_variance(plot_file, var):
    """
    Plot cumulative explained variance (%) as a function of
    principal components and save the figure.
    """
    var = np.asarray(var)

    # cumulative explained variance in percent
    cum_var = np.cumsum(var)
    cum_var = 100.0 * cum_var

    # component indices, including 0
    pcs = np.arange(0, len(cum_var) + 1)

    # prepend 0 so we explicitly plot (0, 0)
    cum_var = np.concatenate([[0.0], cum_var])

    plt.figure()
    plt.plot(pcs, cum_var, marker='o')
    plt.axhline(100.0, linestyle='--')  # 100% reference line

    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("Cumulative PCA explained variance")
    plt.ylim(0, 105)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()




if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    run(args.build)