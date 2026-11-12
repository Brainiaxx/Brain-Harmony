import os
import json

PROCESSED_DIR = "./data/processed"

def main():
    # Standard ABIDE-I Repetition Times (in seconds) by Site ID
    tr_dict = {
        "CALTECH": 2.0,
        "CMU_a": 2.0,
        "CMU_b": 2.0,
        "KKI": 2.5,
        "LEUVEN_1": 1.667,
        "LEUVEN_2": 1.667,
        "MAX_MUN": 3.0,
        "NYU": 2.0,
        "OLIN": 1.5,
        "PITT": 1.5,
        "SBL": 2.2,
        "SDSU": 2.0,
        "TRINITY": 2.0,
        "UCLA_1": 3.0,
        "UCLA_2": 3.0,
        "UM_1": 2.0,
        "UM_2": 2.0,
        "USM": 2.0,
        "YALE": 2.0
    }
    
    out_path = os.path.join(PROCESSED_DIR, "TR.json")
    with open(out_path, 'w') as f:
        json.dump(tr_dict, f, indent=4)
        
    print(f"Created TR.json at {out_path}")

if __name__ == "__main__":
    main()