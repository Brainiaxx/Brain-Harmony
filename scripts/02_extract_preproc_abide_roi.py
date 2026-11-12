import os
import glob
import shutil
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, new_img_like, coord_transform
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from tqdm import tqdm
import numpy as np

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

def standardize_t1(t1_path, t1_out):
    """
    Reorients and resamples the T1 image to match the standard MNI dimensions 
    (167 x 212 x 160) required by Brain Harmony's T1 dataloader.
    """
    # Load the raw T1 NIfTI image
    img = nib.load(t1_path)
    
    # 1. Reorient to canonical FSL/MNI orientation (+LPS / standard RAS space)
    img_canonical = nib.as_closest_canonical(img)
    
    # 2. Target shape corresponding to the paper's MNI specification (167, 212, 160)
    target_shape = (167, 212, 160)
    
    # If the current shape differs, resample/resize smoothly to match target shape
    if img_canonical.shape != target_shape:
        # Calculate scaling factors to match target dimensions
        current_data = img_canonical.get_fdata()
        zoom_factors = [t / c for t, c in zip(target_shape, current_data.shape)]
        
        from scipy.ndimage import zoom
        resampled_data = zoom(current_data, zoom_factors, order=1) # Trilinear interpolation for MRI
        
        # Normalize voxel intensities to [0, 1] range as specified in Appendix A.1
        min_val, max_val = resampled_data.min(), resampled_data.max()
        if max_val > min_val:
            resampled_data = (resampled_data - min_val) / (max_val - min_val)
            
        # Create a new NIfTI image maintaining the correct affine matrix
        img_standardized = new_img_like(img_canonical, resampled_data)
        nib.save(img_standardized, t1_out)
    else:
        # If it already matches, just normalize intensity and save
        data = img_canonical.get_fdata()
        min_val, max_val = data.min(), data.max()
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        img_normalized = new_img_like(img_canonical, data)
        nib.save(img_normalized, t1_out)

def main():
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17)
    masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize='zscore_sample', verbose=0)
    
    fmri_files = glob.glob(os.path.join(RAW_DIR, "fmri", "*_func.nii.gz"))
    
    for fmri_path in tqdm(fmri_files, desc="Formatting Modalities"):
        sid = os.path.basename(fmri_path).replace('_func.nii.gz', '')
        t1_path = os.path.join(RAW_DIR, "t1", f"{sid}_anat.nii.gz")
        
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
                standardize_t1(t1_path, t1_out)
            except Exception as e:
                print(f"Failed T1 {sid}: {e}")

if __name__ == "__main__":
    main()