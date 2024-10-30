# -*- coding:utf-8 -*-
# @Script: 04-painexplo_makegroupmask.py
# @Description: Create a group mask from individual masks


from nilearn.image import load_img, new_img_like
import os
import nilearn.plotting
import numpy as np
from os.path import join as opj
import numpy as np
from painexplo_config import global_parameters  # noqa

from nilearn.plotting import plot_stat_map
import nilearn
import matplotlib.pyplot as plt

# Fetch anatomical image for plots
anat = nilearn.datasets.fetch_icbm152_2009()["t1"]

###################################################################
# Paths
###################################################################
param = global_parameters()
basepath = param.bidspath
prepdir = opj(basepath, "derivatives/fmriprep")

###################################################################
# Get sub
###################################################################

# Get all subjects
subs_id = [
    s
    for s in os.listdir(prepdir)
    if "sub-" in s and ".html" not in s and "sub-010" not in s
]  # Exclude sub-010 because box cut off

# Build group mask
msk_thrs = 1  # Proportion of sessions mask with this voxel to incude

# Load all part masks data
all_masks, files = [], []
for s in subs_id:
    part_mask = []
    funcdir = opj(prepdir, s, "func")
    # Get masks for all sessions
    all_masks += [
        load_img(opj(funcdir, f)).get_fdata()
        for f in os.listdir(funcdir)
        if "_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz" in f
    ]

    part_mask += [
        load_img(opj(funcdir, f)).get_fdata()
        for f in os.listdir(funcdir)
        if "_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz" in f
    ]

    files += [
        opj(funcdir, f)
        for f in os.listdir(funcdir)
        if "_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz" in f
    ]

    # Save for each part
    part_mask = np.stack(part_mask)
    part_mask = np.where(
        np.sum(part_mask, axis=0) / part_mask.shape[0] >= msk_thrs, 1, 0
    )
    part_mask = new_img_like(files[0], part_mask)

    # PLot part mask
    fig = plot_stat_map(part_mask, bg_img=anat)
    plt.savefig(opj(funcdir, s + "_bold_total_mask.png"))


# Stack all in a single array
all_masks = np.stack(all_masks)

# Get proportion of voxels in mask and threshold
group_mask = np.where(
    (np.sum(all_masks, axis=0) / all_masks.shape[0]) >= msk_thrs, 1, 0
)

# Make nifti and save
group_mask = new_img_like(files[0], group_mask)
# nilearn.plotting.plot_stat_map(group_mask, bg_img=anat)

group_mask.to_filename(opj(basepath, "derivatives/group_mask.nii.gz"))

# Plot group mask and save
fig = plot_stat_map(group_mask, bg_img=anat)
plt.savefig(opj(basepath, "derivatives/group_mask.png"))
