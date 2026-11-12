# Preprocessed ABIDE-I download and model compatibility notes

## Getting the data

### 01_download_preproc_abide
Creates directories for fmri and t1 data and downloads it. Filters on sample rating to avoid processing subjects that will fail downstream.

### 02_extract_preproc_abide_roi
Reshapes T1 tensor to (167,212,160) to match dataloader if needed and normalizes voxel intensities as specified in Appendix A.

### 03_prepare_preproc_abide_splits
Generate train test splits.

### 04_create_preproc_abide_tr_file
Create the TR file with the sampling rates of each institution

## Running the model

1. Download checkpoints
2. If running on local, some changes need to be made.
    - libs/flex_transformer.py
        - Fix module call to expand mask
            ´´´
            def forward(self, x, attention_mask, output_attentions=False):
                if attention_mask is not None:
                    #from attn_utils import expand_mask
                    from .attn_utils.expand_mask import _expand_mask

                    attention_mask = _expand_mask(attention_mask, x.dtype)
                ´´´
        - Comment out breakpoint() further down to avoid spamming c during run
    
    - datasets/datasets.pt
        - Added try except block to catch subjects that failed TR filtering in stage 0
        - Use scripts/check_preproc_missing.py to find out how many, currently 23 are failing. Should look into why.
    - modules/harmonizer/stage2_finetune/main_finetune.py
        - As im running on the latest version of cuda for the blackwell architecture, the timm module has to be replaced with python code to avoid compatibility issues. The timm module has been removed in latest pytorch as its behaviour can now be fully replicated in python. 