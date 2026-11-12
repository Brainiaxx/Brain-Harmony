import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PHENOTYPE_URL = f"{BASE_URL}/Phenotypic_V1_0b_preprocessed1.csv"

RAW_DIR = "./data/raw"
os.makedirs(os.path.join(RAW_DIR, "fmri"), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, "t1"), exist_ok=True)

def download_file(url, dest_path):
    if os.path.exists(dest_path): return True
    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            return True
        return False
    except: return False

def main():
    pheno_path = os.path.join(RAW_DIR, "ABIDE_Phenotype.csv")
    download_file(PHENOTYPE_URL, pheno_path)
    
    df = pd.read_csv(pheno_path)
    
    # QA Filtering: Only download subjects that passed manual anatomical and functional QC
    # Adjust or remove these filters based on how strict you want your dataset to be
    df_clean = df[(df['qc_anat_rater_2'] == 'OK') & (df['qc_func_rater_2'] == 'OK')]
    
    file_ids = [str(fid) for fid in df_clean['FILE_ID'].dropna() if str(fid) != 'no_filename']
    print(f"Found {len(file_ids)} subjects passing QA.")

    tasks = []
    for fid in file_ids:
        # fMRI: Download the FULLY PREPROCESSED functional data
        func_url = f"{BASE_URL}/Outputs/cpac/filt_global/func_preproc/{fid}_func_preproc.nii.gz"
        tasks.append((func_url, os.path.join(RAW_DIR, "fmri", f"{fid}_func.nii.gz")))
        
        # T1: ANTs 3D Voxel-wise Cortical Thickness Volume
        anat_url = f"{BASE_URL}/Outputs/ants/anat_thickness/{fid}_anat_thickness.nii.gz"
        tasks.append((anat_url, os.path.join(RAW_DIR, "t1", f"{fid}_anat.nii.gz")))

    print(f"Downloading {len(tasks)} files concurrently...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, dest) for url, dest in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures)): pass

if __name__ == "__main__":
    main()