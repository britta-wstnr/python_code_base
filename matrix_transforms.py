from copy import deepcopy
import numpy as np


def stc_2_mgzvol(coords_in, fwd, mri_mgz):
    """Convert stc coordinates to match nilearn plotting on mgz MRI."""
    # convert to mm
    coords_in = deepcopy(coords_in)
    coords_in *= 1000

    # MEG headspace to RAS surface
    ras2meg = deepcopy(fwd['mri_head_t']['trans'])
    ras2meg[0:3, 3] *= 1000.

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
    coords_out /= 1000.
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


def get_headtransform(lpa, rpa, nas):
    # assumes elekta system
    # adpated from the FieldTrip toolbox
    dir_z = np.cross(rpa - lpa, nas - lpa)
    dir_x = rpa - lpa
    dir_y = np.cross(dir_z, dir_x)
    dir_x = dir_x / np.linalg.norm(dir_x)
    dir_y = dir_y / np.linalg.norm(dir_y)
    dir_z = dir_z / np.linalg.norm(dir_z)
    dirs = np.array((dir_x, dir_y, dir_z))

    origin = lpa + np.dot(np.dot(nas - lpa, dir_x), dir_x)

    rotation = np.eye(4)
    tmp = np.eye(3).dot(np.linalg.inv(dirs))
    rotation[0:3, 0:3] = np.linalg.inv(tmp)

    tra = np.eye(4)
    tra[0:4, 3] = np.append(origin, 1.)

    tfm = np.dot(rotation, tra)
    return tfm


def get_fids_from_raw(raw):
    for row in raw.info['dig']:
        if row['kind'] == 1:
            if row['ident'] == 1:
                lpa_meg = row['r']
            elif row['ident'] == 2:
                nas_meg = row['r']
            elif row['ident'] == 3:
                rpa_meg = row['r']
    fids = {'lpa' : lpa_meg, 'rpa' : rpa_meg, 'nas' : nas_meg}
    return fids
