import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

def main():
    pheno_path = os.path.join(RAW_DIR, "ABIDE_Phenotype.csv")
    df = pd.read_csv(pheno_path)
    
    valid_subjects = []
    
    for _, row in df.iterrows():
        sid = str(row['FILE_ID'])
        if sid == 'no_filename': continue
        
        t1_path = os.path.join(PROCESSED_DIR, "t1", f"{sid}_T1.nii.gz")
        fmri_path = os.path.join(PROCESSED_DIR, "fmri", sid, f"{sid}_Schaefer17n400p.csv.gz")
        
        if os.path.exists(t1_path) and os.path.exists(fmri_path):
            site_id = row.get('SITE_ID', 'Unknown')
            valid_subjects.append({
                # Prepend the site and a slash so datasets.py can parse it correctly
                "id": f"{site_id}/sub-{sid}", 
                "label": int(row['DX_GROUP']),
                "site": site_id
            })

    labels = [s['label'] for s in valid_subjects]
    train_val, test = train_test_split(valid_subjects, test_size=0.15, stratify=labels, random_state=42)
    
    train_val_labels = [s['label'] for s in train_val]
    train, val = train_test_split(train_val, test_size=0.176, stratify=train_val_labels, random_state=42)

    splits = {"train": train, "val": val, "test": test}
    
    out_path = os.path.join(PROCESSED_DIR, "data_splits_seed0.json")
    with open(out_path, 'w') as f: json.dump(splits, f, indent=2)

if __name__ == "__main__":
    main()