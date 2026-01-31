import numpy as np
from tqdm import tqdm
import pyvista as pv
import dbdicom as db
import vreg

from ibeat_kidney_ssa.utils import sdf


def mosaic_masks_dcm(masks, imagefile, labels=None, view_vector=(1, 0, 0)):

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    plotter = pv.Plotter(window_size=(ncols*width, nrows*height), shape=(nrows, ncols), border=False, off_screen=True)
    plotter.background_color = 'white'

    row = 0
    col = 0
    for i, mask_series in tqdm(enumerate(masks), desc=f'Building mosaic'):

        # Set up plotter
        plotter.subplot(row,col)
        if labels is not None:
            plotter.add_text(labels[i], font_size=6)
        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1

        # Load data
        vol = db.volume(mask_series, verbose=0)
        mask_norm, _ = sdf.compress(vol.values, (32, 32, 32))

        # Plot tile
        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = vol.spacing
        orig_surface = orig_vol.contour(isosurfaces=[0.5])
        plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
        plotter.camera_position = 'iso'
        plotter.view_vector(view_vector)  # rotate 180° around vertical axis
    
    plotter.screenshot(imagefile)
    plotter.close()



def mosaic_masks_npz(masks, imagefile, labels=None, view_vector=(1, 0, 0)):

    # Plot settings
    aspect_ratio = 16/9
    width = 150
    height = 150

    # Count nr of mosaics
    n_mosaics = len(masks)
    nrows = int(np.ceil(np.sqrt((width*n_mosaics)/(aspect_ratio*height))))
    ncols = int(np.ceil(n_mosaics/nrows))

    plotter = pv.Plotter(
        window_size=(ncols*width, nrows*height), 
        shape=(nrows, ncols), 
        border=False, 
        off_screen=True,
    )
    plotter.background_color = 'white'

    row = 0
    col = 0
    for i, mask_series in tqdm(enumerate(masks), desc=f'Building mosaic'):

        # Set up plotter
        plotter.subplot(row,col)
        if labels is not None:
            plotter.add_text(labels[i], font_size=6)
        if col == ncols-1:
            col = 0
            row += 1
        else:
            col += 1

        # Load data
        vol = vreg.read_npz(mask_series)
        mask_norm, _ = sdf.compress(vol.values.astype(bool), (32, 32, 32))

        # Plot tile
        orig_vol = pv.wrap(mask_norm.astype(float))
        orig_vol.spacing = [1.0, 1.0, 1.0]
        orig_surface = orig_vol.contour(isosurfaces=[0.5])
        plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
        plotter.camera_position = 'iso'
        plotter.view_vector(view_vector)  # rotate 180° around vertical axis
    
    plotter.screenshot(imagefile)
    plotter.close()