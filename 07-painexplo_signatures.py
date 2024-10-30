# -*- coding:utf-8 -*-
# @Script: 03-painexplo_preprocess.py
# @Description: Apply brain signatures to single trial data and average images
# and create some plots to assess pattern expression of each signature in the data

# TODO: Improve plots

from nilearn.image import load_img, resample_to_img
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask

from tqdm.notebook import tqdm
import seaborn as sns
from scipy.spatial.distance import cosine
from scipy.stats import zscore, pearsonr, spearmanr
import matplotlib.pyplot as plt
from painexplo_config import global_parameters, pattern_expression_nocv

###################################################################
# Paths
###################################################################
# Get global parameters
param = global_parameters()
basepath = opj(param.bidspath)
wager_maps_path = opj(param.bidspath, "external/wager_maps")
preppath = opj(param.bidspath, "derivatives/fmriprep")
fig_path = opj(param.bidspath, "derivatives/figures")
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

###################################################################
# Parameters
###################################################################
# VIF thresold to exclude high multicollinearity trials
vif_threshold = 4

# Analysis mask
group_mask = load_img(opj(basepath, "derivatives/group_mask.nii.gz"))

# Signatures to use
signatures = {
    "nps": opj(wager_maps_path, "weights_NSF_grouppred_cvpcr.nii.gz"),
    "siips": opj(
        wager_maps_path, "2017_Woo_SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii"
    ),
    "pines": opj(
        wager_maps_path, "2015_Chang_PLoSBiology_PINES/Rating_Weights_LOSO_2.nii"
    ),
}


###################################################################
# Apply signatures to single trial data
###################################################################

subs = [s for s in os.listdir(preppath) if "sub" in s and "html" not in s]
all_data = []
for p in tqdm(subs):

    # Create output folder if not existing
    out_path = opj(basepath, "derivatives", "behav", p)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get main task file for each plot
    files = [
        f
        for f in os.listdir(opj(basepath, p, "func"))
        if "events" in f and ".tsv" in f and "._" not in f
    ]
    files.sort()

    # Create single dataframe from 10 runs
    data = []
    for f in files:
        data.append(pd.read_csv(opj(basepath, p, "func", f), sep="\t"))
    data = pd.concat(data)
    # BIDS files have multiple events per trial, keep only one row per trial
    data = data[data["trial_type"] == "cues"]

    # # Get images for pain, rating and cues
    X_data = dict()
    for idx, type in enumerate(["pain", "rating", "cues"]):
        X_data[type] = []
        img = [
            opj(basepath, "derivatives", "glm_strials_" + type, f)
            for f in os.listdir(opj(basepath, "derivatives", "glm_strials_" + type))
            if f.endswith("_t.nii.gz") and p in f
        ]
        # Sort images
        img.sort()
        # Add to data
        data["imgpath_" + type] = img
        # Get VIFs
        data["vifs_" + type] = pd.read_csv(
            opj(basepath, "derivatives", "glm_strials_" + type, p + "_vifs.csv")
        )["vif"]

        # Load images in a numpy array
        X_data[type] += [
            apply_mask(fname, group_mask) for fname in data["imgpath_" + type]
        ]
        X_data[type] = np.squeeze(np.stack(X_data[type]))
        X_data[type] = X_data[type].astype(np.float32)

    for name, path in signatures.items():

        # Load signature, resample to mask, apply mask
        pattern = apply_mask(
            resample_to_img(load_img(path), group_mask),
            group_mask,
        )
        # Apply to data
        data = pattern_expression_nocv(
            X_data["rating"], pattern, data, name + "_rating"
        )
        data = pattern_expression_nocv(X_data["pain"], pattern, data, name + "_pain")
        data = pattern_expression_nocv(X_data["cues"], pattern, data, name + "_cues")

    # Add correaltions between temp/rating with pattern expression
    for name, _ in signatures.items():

        for type in ["rating", "pain", "cues"]:
            data_corr = data[data["vifs_" + type] < vif_threshold]
            data[type + "_" + name + "_cosine_rating_cor"] = spearmanr(
                data_corr[name + "_" + type + "_cosine"], data_corr["rating"]
            )[0]
            data[type + "_" + name + "_cosine_temp_cor"] = spearmanr(
                data_corr[name + "_" + type + "_cosine"], data_corr["stim_intensite"]
            )[0]
    # Append to all data
    all_data.append(data)

# Concatenate all data
all_data_df = pd.concat(all_data)
# Save
all_data_df.to_csv(opj(basepath, "derivatives", "signatures_full_df.csv"))

# Mean correlations
all_data_df = (
    all_data_df.groupby("participant_id").mean(numeric_only=True).reset_index()
)

# Columns to keep
cols = []
for type in ["rating", "pain", "cues"]:
    for name, _ in signatures.items():
        cols.append(type + "_" + name + "_cosine_temp_cor")
        cols.append(type + "_" + name + "_cosine_rating_cor")

other_cols = []
for type in ["rating", "pain", "cues"]:
    for name, _ in signatures.items():
        other_cols.append(name + "_" + type + "_cosine")
# Save
all_data_df = all_data_df[["participant_id"] + other_cols + cols]
all_data_df.to_csv(opj(basepath, "derivatives", "signatures_corr_df.csv"))

# Plot
df_plot = all_data_df.melt(id_vars="participant_id", value_vars=cols)


# Add phase, signature and cor type
phase, signature, cor_type = [], [], []
for i in df_plot["variable"].str.split("_").values:
    phase.append(i[0])
    signature.append(i[1])
    cor_type.append(i[3])

df_plot["phase"] = phase
df_plot["signature"] = signature
df_plot["cor_type"] = cor_type

# Plot
plt.figure()
sns.catplot(
    data=df_plot, x="phase", y="value", hue="cor_type", col="signature", kind="bar"
)
plt.savefig(
    opj(basepath, "derivatives", "figures", "strials_signatures_correlations_bar.png")
)


plt.figure()
sns.catplot(
    data=df_plot, x="phase", y="value", hue="cor_type", col="signature", kind="swarm"
)
plt.savefig(
    opj(basepath, "derivatives", "figures", "strials_signatures_correlations_swarm.png")
)


###################################################################
# Apply signatures to average image
###################################################################
# Get all univariate images
univariate_path = opj(basepath, "derivatives", "glm_model_univariate")
files = [f for f in os.listdir(univariate_path) if "t.nii.gz" in f and ".nii.gz" in f]
subs, phases, img_path = [], [], []
for f in tqdm(files):
    subs.append(f.split("_")[0])
    phases.append(f.split("_")[1])
    img_path.append(opj(univariate_path, f))
# Create data frame
data = pd.DataFrame({"participant_id": subs, "phase": phases, "imgpath": img_path})
# Load images
X = []
X += [apply_mask(fname, group_mask) for fname in tqdm(data["imgpath"])]
X = np.squeeze(np.stack(X))
X = X.astype(np.float32)

# Apply signatures
for name, path in signatures.items():

    # Load signature, resample to mask, apply mask
    pattern = apply_mask(
        resample_to_img(load_img(path), group_mask),
        group_mask,
    )
    # Apply to data
    data = pattern_expression_nocv(X, pattern, data, name)

# plot
df_plot = data.melt(
    id_vars=["participant_id", "phase"],
    value_vars=[name + "_cosine" for name in signatures.keys()],
)
df_plot["signature"] = df_plot["variable"].str.split("_").str[0]

plt.figure()
g = sns.catplot(data=df_plot, x="phase", y="value", hue="signature", kind="bar")
g.set_ylabels("Cosine similarity")
plt.savefig(opj(basepath, "derivatives", "figures", "univariate_signatures_bar.png"))

plt.figure()
g = sns.catplot(data=df_plot, x="phase", y="value", hue="signature", kind="swarm")
g.set_ylabels("Cosine similarity")
plt.savefig(opj(basepath, "derivatives", "figures", "univariate_signatures_swarm.png"))

# Zscore within participant
data_zscore = df_plot.copy()
for p in data_zscore["participant_id"].unique():
    data_zscore.loc[data_zscore["participant_id"] == p, "value"] = zscore(
        data_zscore.loc[data_zscore["participant_id"] == p, "value"]
    )

plt.figure()
g = sns.catplot(data=data_zscore, x="phase", y="value", hue="signature", kind="bar")
g.set_ylabels("Z-scored within part cosine similarity")
plt.savefig(opj(basepath, "derivatives", "figures", "univariate_signatures_bar_z.png"))

plt.figure()
g = sns.catplot(data=data_zscore, x="phase", y="value", hue="signature", kind="swarm")
g.set_ylabels("Z-scored within part cosine similarity")
plt.savefig(
    opj(basepath, "derivatives", "figures", "univariate_signatures_swarm_z.png")
)
