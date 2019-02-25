from copy import deepcopy
import numpy as np


def stc_2_mgzvol(coords_in, fwd, mri_mgz):
    """Convert stc coordinates to match nilearn plotting on mgz MRI."""
    # convert to mm
    coords_in = deepcopy(coords_in)
    coords_in *= 1000

    # MEG headspace to RAS surface
    ras2meg = deepcopy(fwd['mri_head_t']['trans'])

    coords_ras = transform_coords(coords_in, np.linalg.inv(ras2meg))

    # RAS surface to mgz voxel space
    vox2ras = mri_mgz.header.get_vox2ras_tkr()
    coords_mgz = transform_coords(coords_ras, np.linalg.inv(vox2ras))

    # mgz voxel space to world space
    mgz2mri = mri_mgz.header.get_affine()
    coords_out = transform_coords(coords_mgz, mgz2mri)

    return coords_out


def mgzvol_2_stc(coords_in, fwd, mri_mgz):
    # assumes mm
    coords_in = deepcopy(coords_in)

    # transforms
    ras2meg = deepcopy(fwd['mri_head_t']['trans'])
    ras2meg[0:3, 3] *= 1000.
    vox2ras = mri_mgz.header.get_vox2ras_tkr()
    mgz2mri = mri_mgz.header.get_affine()

    # world space to mgz voxel space
    coords_mgz = transform_coords(coords_in, np.linalg.inv(mgz2mri))

    coords_ras = transform_coords(coords_mgz, vox2ras)

    coords_out = transform_coords(coords_ras, ras2meg)
    coords_out /= 1000
    return coords_out


def transform_coords(coords_in, tfm):
    """Affine coordinate transformations."""
    coords_out = np.dot(tfm[0:3, 0:3], coords_in.T) + tfm[0:3, 3]

    return coords_out


def get_coord_from_peak(stc, fwd):
    """Get the right coordinates belonging to an stc peak."""
    peak_idx = stc.get_peak(mode='pos', vert_as_index=True)
    peak_vert = stc.vertices[peak_idx[0]]
    coords_out = fwd['src'][0]['rr'][peak_vert, ]

    return coords_out


def get_distance(p1, p2):
    """Get the distance between two points in 3D space."""
    p1 = np.array(p1)
    p2 = np.array(p2)

    distance = np.sum((p1-p2)**2)
    distance = np.sqrt(distance)

    return distance
