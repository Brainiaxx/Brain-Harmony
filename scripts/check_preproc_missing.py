import json
import os

splits_file = "data/abide_i/processed/data_splits_seed0.json"
embed_dir = "experiments/stage0_embed/downstream_embed/AbideI"

# Load the expected subjects from the split file
with open(splits_file, 'r') as f:
    splits = json.load(f)

# Combine train, val, and test IDs
expected_sids = splits.get('train', []) + splits.get('val', []) + splits.get('test', [])

missing_sids = []
for sid in expected_sids:
    # Handle if JSON stores dictionaries or raw strings
    sid_name = sid['id'] if isinstance(sid, dict) else sid
    
    expected_path = os.path.join(embed_dir, f"{sid_name}.npz")
    if not os.path.exists(expected_path):
        missing_sids.append(sid_name)

print(f"Total expected subjects in JSON: {len(expected_sids)}")
print(f"Total successfully embedded (.npz): {len(expected_sids) - len(missing_sids)}")
print(f"Total MISSING: {len(missing_sids)}")

if missing_sids:
    print("\nFirst 10 missing subjects:")
    for m in missing_sids[:10]:
        print(f" - {m}")