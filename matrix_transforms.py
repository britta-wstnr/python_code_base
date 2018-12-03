from copy import deepcopy
import numpy as np


def stc_2_niivol(coords_in, fwd, mri_mgz):
    """Convert stc coordinates to match nilearn plotting on nifti MRI."""
    # convert to mm
    coords_in = deepcopy(coords_in)
    coords_in *= 1000

    # MEG headspace to RAS surface
    ras2meg = deepcopy(fwd['mri_head_t']['trans'])
    ras2meg[0:3, 3] *= 1000.
    coords_ras = transform_coords(coords_in, np.linalg.inv(ras2meg))

    # RAS sruface to mgz voxel space
    vox2ras = mri_mgz.header.get_vox2ras_tkr()
    coords_mgz = transform_coords(coords_ras, np.linalg.inv(vox2ras))

    # mgz voxel space to world space
    mgz2mri = mri_mgz.header.get_affine()
    coords_out = transform_coords(coords_mgz, mgz2mri)

    return coords_out


def transform_coords(coords_in, tfm):
    coords_out = np.dot(tfm[0:3, 0:3], coords_in.T) + tfm[0:3, 3]

    return coords_out
