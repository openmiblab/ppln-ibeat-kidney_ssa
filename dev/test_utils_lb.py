import numpy as np
import miblab_ssa as ssa

from miblab_plot import gui



def test_reconstruct_volume_3d():

    ell_1 = gui.ellipsoid_mask(
        (64, 64, 64), 
        voxel_sizes=(2.0,1.0,1.0), 
        center=(10,0,0),
        radii=(30,10,10),
        rot_vec=(1,0,0),
    )
    ell_2 = gui.ellipsoid_mask(
        (64, 64, 64), 
        voxel_sizes=(2.0,1.0,1.0), 
        center=(10,0,0),
        radii=(25,15,10),
        rot_vec=(0,1,1),
    )
    # mask = np.logical_or(ell_1, ell_2)
    mask = np.logical_and(ell_1, np.logical_not(ell_2))
    # gui.display_volume(original_mask)

    mesh = ssa.lb.mask_to_mesh(mask) 
    coeffs, rec_mesh = ssa.lb.process(mesh, k=10)

    # Visualize
    gui.visualize_surface_reconstruction(mesh, rec_mesh)



def test_invariance():
    ell_1 = gui.ellipsoid_mask(
        (128, 128, 128), 
        voxel_sizes=(1.0,1.0,1.0), 
        center=(20,0,0),
        radii=(60,20,20),
        rot_vec=(1,0,0),
    )
    ell_2 = gui.ellipsoid_mask(
        (128, 128, 128), 
        voxel_sizes=(1.0,1.0,1.0), 
        center=(20,0,0),
        radii=(50,30,20),
        rot_vec=(0,1,1),
    )
    mask = np.logical_and(ell_1, np.logical_not(ell_2))
    mesh = ssa.lb.mask_to_mesh(mask) 
    coeffs_ref, _ = ssa.lb.process(mesh, k=5)

    print('ref')
    print(coeffs_ref)

    # ell_1 = gui.ellipsoid_mask(
    #     (128, 128, 128), 
    #     voxel_sizes=(1.0,1.0,1.0), 
    #     center=(20,0,0),
    #     radii=(60,20,20),
    #     rot_vec=(1,0,0),
    # )
    # ell_2 = gui.ellipsoid_mask(
    #     (128, 128, 128), 
    #     voxel_sizes=(1.0,1.0,1.0), 
    #     center=(20,0,0),
    #     radii=(50,30,20),
    #     rot_vec=(0,1,1),
    # )
    # mask = np.logical_and(ell_1, np.logical_not(ell_2))
    mesh = ssa.lb.mask_to_mesh(mask) 
    coeffs, _ = ssa.lb.process(mesh, k=5)

    print('translate')
    print(coeffs)
    
    print(np.linalg.norm(coeffs-coeffs_ref)/np.linalg.norm(coeffs_ref))
    norm_ref = np.array([np.linalg.norm(c) for c in coeffs_ref])
    norm_tr = np.array([np.linalg.norm(c) for c in coeffs])
    print(np.linalg.norm(norm_tr-norm_ref)/np.linalg.norm(norm_ref))




# Example usage
if __name__ == '__main__':
    # test_reconstruct_volume_3d()
    test_invariance()