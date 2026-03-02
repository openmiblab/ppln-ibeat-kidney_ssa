import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from scipy.fftpack import dctn, idctn
import os
import zarr
import miblab_ssa as ssa

def visualize_spectral_process(mask: np.ndarray, order=16, save_path="kidney_spectral_process.png"):
    """
    Visualizes the 4 stages of the spectral truncation process for a 3D mask.
    """
    # 0. Prep: Get a central slice index for visualization
    mid_idx = mask.shape[0] // 2
    
    # 1. Original Mask (Slice)
    orig_slice = mask[mid_idx].astype(float)

    # 2. Signed Distance Transform (Saturated)
    # We compute the full 3D SDF to stay true to the algorithm
    mask_bool = mask.astype(bool)
    sdf = (distance_transform_edt(~mask_bool) - distance_transform_edt(mask_bool)).astype(np.float32)
    # Saturation matches your features_from_mask logic
    saturation_threshold = 5.0
    sdf_saturated = saturation_threshold * np.tanh(sdf / saturation_threshold)
    sdf_slice = sdf_saturated[mid_idx]

    # 3. Spectral Decomposition (Log-scaled for visibility)
    # We take the DCT of the saturated SDF
    full_coeffs = dctn(sdf_saturated, norm='ortho')
    # We'll visualize the log-magnitude of the coefficients at the same slice
    # Note: Coefficients are in frequency space, so we look at the low-frequency corner
    spec_slice = np.log1p(np.abs(full_coeffs[0, :order*4, :order*4]))

    # 4. Reconstructed Mask (Truncated)
    # Reuse your existing logic to get the smoothed mask
    recon_mask = ssa.sdf_ft.smooth_mask(mask_bool, order=order)
    recon_slice = recon_mask[mid_idx].astype(float)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Panel A: Original Mask
    axes[0, 0].imshow(orig_slice, cmap='gray')
    axes[0, 0].set_title("Original Mask (Cross-section)")
    axes[0, 0].axis('off')

    # Panel B: Signed Distance Transform
    im1 = axes[0, 1].imshow(sdf_slice, cmap='RdBu_r')
    axes[0, 1].set_title("Saturated SDF")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].axis('off')

    # Panel C: Spectral Coefficients (Low Frequencies)
    im2 = axes[1, 1].imshow(spec_slice, cmap='magma')
    axes[1, 1].set_title(f"Spectral Power (Top {order*4}x{order*4} corner)")
    fig.colorbar(im2, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].axis('off')

    # Panel D: Reconstructed Mask
    axes[1, 0].imshow(recon_slice, cmap='gray')
    axes[1, 0].set_title(f"Truncated Mask (Order {order})")
    axes[1, 0].axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure saved to {save_path}")

def figure_features(build):

    dir_input = os.path.join(build, 'kidney_ssa', 'stage_3_normalize')
    masks = os.path.join(dir_input, 'normalized_kidney_masks.zarr')
    png_output = os.path.join(build, 'fig.png')

    input_root = zarr.open(masks, mode='r')
    labels = input_root['labels']
    labels_idx = {label: i for i, label in enumerate(labels)}
    mask_idx = labels_idx["3128_072-R"]
    mask = input_root['masks'][mask_idx]

    visualize_spectral_process(mask, order=16, save_path=png_output)

if __name__ == '__main__':
    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    figure_features(BUILD)