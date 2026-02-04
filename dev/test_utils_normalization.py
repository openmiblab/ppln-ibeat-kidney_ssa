import numpy as np
import miblab_ssa as ssa
from miblab_plot import gui



def test_normalization():

    voxel_size = (1.0,1.0,1.0)
    shape = (128, 128, 128)
    ell_center = (0,30,0)
    sph_center = tuple(np.array(ell_center) + np.array((15,0,0)))
    rot_vec=(1,0,0)
    ell = gui.ellipsoid_mask(
        shape, 
        voxel_sizes=voxel_size, 
        center=ell_center,
        radii=(45,30,15),
        rot_vec=rot_vec,
    )
    sph = gui.ellipsoid_mask(
        shape, 
        voxel_sizes=voxel_size, 
        center=sph_center,
        radii=(30,30,30),
        rot_vec=rot_vec,
    )
    mask = np.logical_or(ell, sph)
    #mask = ell

    # gui.display_volume(mask, voxel_size)

    # Normalize
    spacing_norm = 1.0
    volume_norm = 1e6
    mask_norm, params = ssa.normalize_kidney_mask(mask, voxel_size, 'right')

    print(f"Target normalized mask volume (L): {volume_norm / 1e6}")
    print(f"Actual normalized mask volume (L): {mask_norm.sum() * spacing_norm ** 3 / 1e6}")

    # viewer = napari.Viewer()
    # viewer.add_image(mask_norm.T)
    # napari.run()

    # Visualize
    voxel_size_norm = 3 * [spacing_norm]
    gui.display_volumes_two_panel(mask, mask_norm, voxel_size, voxel_size_norm)

    # # Denormalize
    # mask_recon = normalize.denormalize_mask(mask_norm, params)

    # # Visualize
    # gui.display_volumes(mask, mask_recon, voxel_size)


# Example usage
if __name__ == '__main__':
    test_normalization()