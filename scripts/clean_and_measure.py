# save as scripts/clean_and_measure.py
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from skimage.morphology import ball
import math

def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata()
    zooms = tuple(map(float, nii.header.get_zooms()[:3]))
    return arr, zooms

def threshold_mask(prob_map, thr):
    return (prob_map >= thr).astype(np.uint8)

def keep_largest_component(mask):
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest).astype(np.uint8)

def remove_small_components(mask, min_vox):
    labeled, n = ndi.label(mask)
    out = np.zeros_like(mask)
    for lab in range(1, n+1):
        comp = (labeled == lab)
        if comp.sum() >= min_vox:
            out |= comp
    return out.astype(np.uint8)

def morph_cleanup(mask, opening_radius=1, closing_radius=2):
    # 3D structuring element
    open_se = ball(opening_radius)
    close_se = ball(closing_radius)
    m = ndi.binary_opening(mask, structure=open_se)
    m = ndi.binary_closing(m, structure=close_se)
    # fill holes slice-wise (axial)
    for z in range(m.shape[2]):
        m[:,:,z] = ndi.binary_fill_holes(m[:,:,z])
    return m.astype(np.uint8)

def compute_volume(mask, zooms):
    voxel_vol = zooms[0]*zooms[1]*zooms[2]  # mm^3
    voxels = int(mask.sum())
    vol_mm3 = voxels * voxel_vol
    return voxels, float(vol_mm3), float(vol_mm3/1000.0)

def compute_max_diameter_world(mask, zooms, sample_limit=2000):
    # get surface/coords of mask voxels
    coords = np.array(np.where(mask)).T  # (N,3) in voxel indices (z last maybe)
    if coords.shape[0] == 0:
        return None
    # convert to world mm coords (assuming identity spacing * zooms)
    world = coords.astype(float)
    world[:,0] *= zooms[0]
    world[:,1] *= zooms[1]
    world[:,2] *= zooms[2]
    n = world.shape[0]
    # if too many points, sample boundary points
    if n > sample_limit:
        idx = np.linspace(0, n-1, sample_limit).astype(int)
        pts = world[idx]
    else:
        pts = world
    # compute pairwise distances (may be O(n^2))
    dists = cdist(pts, pts, metric='euclidean')
    maxd = dists.max()
    return float(maxd)  # mm

def pipeline(nifti_path, mask_path,
             thresholds=[0.3,0.4,0.5,0.6],
             min_sizes=[500,1000,2000,5000],
             open_r=1, close_r=2):
    img, zooms = load_nifti(nifti_path)
    mask_raw, _ = load_nifti(mask_path)

    results = []
    for thr in thresholds:
        # if mask is binary already, thr=0.5 will be fine
        bin_mask = threshold_mask(mask_raw, thr) if mask_raw.dtype != np.uint8 or mask_raw.max()>1 else (mask_raw>0).astype(np.uint8)
        bin_mask = keep_largest_component(bin_mask)
        bin_mask = morph_cleanup(bin_mask, opening_radius=open_r, closing_radius=close_r)
        for minv in min_sizes:
            clean = remove_small_components(bin_mask, minv)
            voxels, vol_mm3, vol_cm3 = compute_volume(clean, zooms)
            maxd_mm = compute_max_diameter_world(clean, zooms)
            results.append({
                'thr': thr,
                'min_vox': minv,
                'voxels': voxels,
                'vol_mm3': vol_mm3,
                'vol_cm3': vol_cm3,
                'max_diam_mm': maxd_mm
            })
    # sort results by vol_mm3
    results_sorted = sorted(results, key=lambda x: x['vol_mm3'])
    return results_sorted

if __name__ == "__main__":
    import sys, json
    nif = sys.argv[1]
    msk = sys.argv[2]
    out = pipeline(nif, msk)
    print(json.dumps(out[:10], indent=2))
