# tumor/utils/preview3d.py
import os
import uuid
import logging
from typing import Optional

import nibabel as nib
import numpy as np
from skimage import measure
from django.conf import settings

# try pyvista first, fallback to matplotlib 3D
try:
    import pyvista as pv
    _have_pyvista = True
except Exception:
    _have_pyvista = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _have_mpl = True
except Exception:
    _have_mpl = False

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def generate_3d_preview(mask_nifti_path: str, out_path: Optional[str] = None) -> str:
    """
    Generate 3D mesh snapshot PNG from mask NIfTI.
    Returns relative path to MEDIA_ROOT.
    """
    meshes_dir = os.path.join(settings.MEDIA_ROOT, "results", "meshes_previews")
    _ensure_dir(meshes_dir)

    if out_path is None:
        out_filename = f"preview3d_{uuid.uuid4().hex}.png"
        out_path = os.path.join(meshes_dir, out_filename)
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        nii = nib.load(mask_nifti_path)
        data = (nii.get_fdata() > 0.5).astype(np.uint8)

        if _have_pyvista:
            try:
                verts, faces, normals, values = measure.marching_cubes(data, level=0.5, spacing=nii.header.get_zooms()[:3])
                faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()
                mesh = pv.PolyData(verts, faces_pv)
                plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
                plotter.add_mesh(mesh, color='red', opacity=1.0)
                plotter.set_background('white')
                plotter.camera_position = 'iso'
                plotter.show(screenshot=out_path)
                plotter.close()
                rel = os.path.relpath(out_path, settings.MEDIA_ROOT)
                return rel
            except Exception as e:
                logger.exception("pyvista preview failed, falling back to matplotlib: %s", e)

        # fallback to matplotlib trisurf
        if not _have_mpl:
            raise RuntimeError("Neither pyvista nor matplotlib available for 3D preview generation")

        verts, faces, normals, values = measure.marching_cubes(data, level=0.5, spacing=nii.header.get_zooms()[:3])
        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        # convert faces to triangular vertices
        mesh_x = verts[:, 0]
        mesh_y = verts[:, 1]
        mesh_z = verts[:, 2]
        tri = faces
        ax.plot_trisurf(mesh_x, mesh_y, tri, mesh_z, linewidth=0.2, antialiased=True, color='red', alpha=1.0)
        ax.set_axis_off()
        # autoscale
        try:
            ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
        except Exception:
            pass
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        rel = os.path.relpath(out_path, settings.MEDIA_ROOT)
        return rel

    except Exception as e:
        logger.exception("3D preview generation error: %s", e)
        raise
