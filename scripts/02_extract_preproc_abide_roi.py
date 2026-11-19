import os
import sys
import glob
import tempfile
from pathlib import Path
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_abide import (
    TEMPLATE_PATH,
    canonical,
    download_template,
    register,
    save_t1,
    sitk_to_array,
)

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

def prepare_template():
    """Canonical FSL MNI152 1 mm grid. The loader pads a crop of this grid."""
    download_template()
    template = canonical(TEMPLATE_PATH)
    nib.save(template, str(TEMPLATE_PATH))
    return str(TEMPLATE_PATH)

def standardize_t1(t1_path, t1_out, template_path):
    """Affinely register a FreeSurfer skull-stripped T1 onto the FSL MNI152
    1 mm grid, then crop to 167 x 212 x 160 and scale intensities to [0, 1].
    """
    brain = nib.as_closest_canonical(nib.load(t1_path))
    with tempfile.TemporaryDirectory() as tmp:
        moving_path = os.path.join(tmp, "brain.nii.gz")
        nib.save(brain, moving_path)
        fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
        moving = sitk.ReadImage(moving_path, sitk.sitkFloat32)
        transform = register(fixed, moving, "affine")
        warped = sitk.Resample(
            moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        warped_array = sitk_to_array(warped).astype(np.float32)

    template = nib.load(template_path)
    if warped_array.shape != template.shape:
        raise RuntimeError(
            f"warped T1 shape {warped_array.shape} != template {template.shape}"
        )
    bright = np.argwhere(warped_array > np.percentile(warped_array, 90))
    center = bright.mean(axis=0)
    if np.any(np.abs(center - np.array(warped_array.shape) / 2) > np.array(warped_array.shape) * 0.2):
        raise RuntimeError(f"warped brain center {center.round(1)} is outside the MNI volume")
    save_t1(warped_array, template, Path(t1_out))

def main():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17)
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize='zscore_sample', verbose=0)
    template_path = prepare_template()
    
    fmri_files = glob.glob(os.path.join(RAW_DIR, "fmri", "*_func.nii.gz"))
    
    for fmri_path in tqdm(fmri_files, desc="Formatting Modalities"):
        sid = os.path.basename(fmri_path).replace('_func.nii.gz', '')
        t1_path = os.path.join(RAW_DIR, "t1", f"{sid}_brain.mgz")
        
        fmri_out = os.path.join(PROCESSED_DIR, "fmri", sid, f"{sid}_Schaefer17n400p.csv.gz")
        t1_out = os.path.join(PROCESSED_DIR, "t1", f"{sid}_T1.nii.gz")

        # 1. Process fMRI into transposed csv.gz
        if not os.path.exists(fmri_out):
            try:
                time_series = masker.fit_transform(fmri_path)
                df = pd.DataFrame(time_series.T, index=[f"ROI_{i}" for i in range(1, 401)])
                os.makedirs(os.path.dirname(fmri_out), exist_ok=True)
                df.to_csv(fmri_out, compression="gzip")
            except Exception as e: 
                print(f"Failed fMRI {sid}: {e}")

        # 2. Standardize T1 to match Brain Harmony's MNI specifications
        if os.path.exists(t1_path) and not os.path.exists(t1_out):
            try:
                os.makedirs(os.path.dirname(t1_out), exist_ok=True)
                standardize_t1(t1_path, t1_out, template_path)
            except Exception as e:
                print(f"Failed T1 {sid}: {e}")

if __name__ == "__main__":
    main()
