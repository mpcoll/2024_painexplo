from nilearn.image import load_img, new_img_like
import os
import nilearn.plotting
import numpy as np
from os.path import join as opj
import numpy as np
from nilearn import masking
from nilearn.image.resampling import coord_transform
from painexplo_config import global_parameters  # noqa

from nilearn.plotting import plot_stat_map
import nilearn
import matplotlib.pyplot as plt

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

subs_id = [
    s
    for s in os.listdir(prepdir)
    if "sub-" in s and ".html" not in s and "sub-010" not in s
]  # Exclude sub-010 because box cut off

# Build group mask
# TODO replace with 1.
msk_thrs = 1  # Proportion of sessions mask with this voxel to incude


gm_mask = load_img(
    opj(basepath, "external/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii")
)
gm_mask = np.where(gm_mask.get_fdata() > 0.2, 1, 0)


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

    fig = plot_stat_map(part_mask, bg_img=anat)
    plt.savefig(opj(funcdir, s + "_bold_total_mask.png"))


# In a single array
all_masks = np.stack(all_masks)

# Get proportion of voxels in mask and threshold
group_mask = np.where(
    (np.sum(all_masks, axis=0) / all_masks.shape[0]) >= msk_thrs, 1, 0
)

# # Grey matter thresholding (maybe not)
# group_mask = group_mask + gm_mask
# group_mask = np.where(group_mask == 2, 1, 0)

# Make nifti and save
group_mask = new_img_like(files[0], group_mask)
# nilearn.plotting.plot_stat_map(group_mask, bg_img=anat)

group_mask.to_filename(opj(basepath, "derivatives/group_mask.nii.gz"))

# Plot group mask
fig = plot_stat_map(group_mask, bg_img=anat)
plt.savefig(opj(basepath, "derivatives/group_mask.png"))
nilearn.plotting.view_img(group_mask, bg_img=anat)
