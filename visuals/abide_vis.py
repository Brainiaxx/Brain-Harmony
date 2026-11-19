from nilearn import plotting

# Overlay the ANTs cortical thickness map onto the standard MNI152 template
plotting.plot_stat_map(
    #"Caltech_0051480_anat.nii.gz",
    "Caltech_0051480_T1_transformed.nii.gz",
    threshold=0.1,          # Mask out 0.0 non-cortical background voxels
    cmap="hot",             # Color map for thickness (in mm)
    display_mode="ortho",   # Axial, coronal, and sagittal slices
    title="ANTs Cortical Thickness (mm)"
)
plotting.show()