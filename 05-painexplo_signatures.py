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
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

###################################################################
# Paths
###################################################################

# TODO
# Get distribution of correlations for different signatures across participants for each participant
# for both pain and rating, cues and

# import global parameters
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
from painexplo_config import global_parameters  # noqa

param = global_parameters()
vif_threshold = 4

basepath = opj(param.bidspath)
wager_maps_path = opj(param.bidspath, "external/wager_maps")
preppath = opj(param.bidspath, "derivatives/fmriprep")
fig_path = opj(param.bidspath, "derivatives/figures")
if not os.path.exists(fig_path):
    os.makedirs(fig_path)


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


# Helper function to get pattern expression
def pattern_expression_nocv(dat, pattern, stats, name):
    """Calculate similarity between maps using dot product and cosine product.
       Non-crossvalidated - to use with external data/patterns.

    Args:
        dat ([array]): images to calculate similarity on (array of shape n images x n voxels)
        pattern ([array]): Pattern weights
        stats ([pd df]): Data frame with subejct id and fods for each in columns
        name ([string]): Name to add to ouput columns
    Returns:
        [df]: stats df with dot and cosine columns added
    """
    pexpress = np.zeros(dat.shape[0]) + 9999
    cosim = np.zeros(dat.shape[0]) + 9999

    for xx in range(dat.shape[0]):
        pexpress[xx] = np.dot(dat[xx, pattern != 0], pattern[pattern != 0])
        cosim[xx] = 1 - cosine(dat[xx, pattern != 0], pattern[pattern != 0])
    stats[name + "_dot"] = pexpress
    stats[name + "_cosine"] = cosim

    return stats


###################################################################
# Apply signatures to single trial data
###################################################################

subs = [s for s in os.listdir(preppath) if "sub" in s and "html" not in s]
subs = ["sub-009"]
all_data = []
for p in tqdm(subs):

    # Create output folder if not existing
    out_path = opj(basepath, "derivatives", "behav", p)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get main task file for each plot
    files = [
        f for f in os.listdir(opj(basepath, p, "func")) if "events" in f and ".tsv" in f
    ]
    files.sort()

    data = []
    for f in files:
        data.append(pd.read_csv(opj(basepath, p, "func", f), sep="\t"))
    data = pd.concat(data)
    # BIDS files have multiple events per trial, keep only one row per trial
    data = data[data["trial_type"] == "cues"]

    # # Get images for pain
    strials_img = [
        opj(basepath, "derivatives", "glm_strials_rating", f)
        for f in os.listdir(opj(basepath, "derivatives", "glm_strials_rating"))
        if f.endswith("_t.nii.gz") and p in f
    ]
    strials_img.sort()
    data["imgpatht_rating"] = strials_img

    strials_img = [
        opj(basepath, "derivatives", "glm_strials_pain", f)
        for f in os.listdir(opj(basepath, "derivatives", "glm_strials_pain"))
        if f.endswith("_t.nii.gz") and p in f
    ]
    strials_img.sort()
    data["imgpatht_pain"] = strials_img

    strials_img = [
        opj(basepath, "derivatives", "glm_strials_cues", f)
        for f in os.listdir(opj(basepath, "derivatives", "glm_strials_cues"))
        if f.endswith("_t.nii.gz") and p in f
    ]
    strials_img.sort()
    data["imgpatht_cues"] = strials_img

    vifs_pain = pd.read_csv(
        opj(basepath, "derivatives", "glm_strials_pain", p + "_vifs.csv")
    )
    data["vifs_pain"] = vifs_pain["vif"]

    vifs_rating = pd.read_csv(
        opj(basepath, "derivatives", "glm_strials_rating", p + "_vifs.csv")
    )
    data["vifs_rating"] = vifs_pain["vif"]

    vifs_cues = pd.read_csv(
        opj(basepath, "derivatives", "glm_strials_cues", p + "_vifs.csv")
    )
    data["vifs_cues"] = vifs_pain["vif"]

    # Brain Data
    X_rating = []
    X_rating += [apply_mask(fname, group_mask) for fname in data["imgpatht_rating"]]
    X_rating = np.squeeze(np.stack(X_rating))
    X_rating = X_rating.astype(np.float32)

    X_pain = []
    X_pain += [apply_mask(fname, group_mask) for fname in data["imgpatht_pain"]]
    X_pain = np.squeeze(np.stack(X_pain))
    X_pain = X_pain.astype(np.float32)

    X_cues = []
    X_cues += [apply_mask(fname, group_mask) for fname in data["imgpatht_cues"]]
    X_cues = np.squeeze(np.stack(X_cues))
    X_cues = X_cues.astype(np.float32)

    for name, path in signatures.items():

        # Load signature, resample to mask, apply mask
        pattern = apply_mask(
            resample_to_img(load_img(path), group_mask),
            group_mask,
        )
        # Apply to data
        data = pattern_expression_nocv(X_rating, pattern, data, name + "_rating")
        data = pattern_expression_nocv(X_pain, pattern, data, name + "_pain")
        data = pattern_expression_nocv(X_cues, pattern, data, name + "_cues")

    # Add correaltions with pattern expression
    for name, _ in signatures.items():

        for type in ["rating", "pain", "cues"]:
            data_corr = data[data["vif_" + type] < vif_threshold]
            data[type + "_" + name + "_cosine_rating_cor"] = spearmanr(
                data_corr[name + "_" + type + "_cosine"], data_corr["rating"]
            )[0]
            data[type + "_" + name + "_cosine_temp_cor"] = spearmanr(
                data_corr[name + "_" + type + "_cosine"], data_corr["stim_intensite"]
            )[0]

    all_data.append(data)

# Concatenate all data
all_data_df = pd.concat(all_data)
# Save
all_data_df.to_csv(opj(basepath, "derivatives", "signatures_full_df.csv"))

# Mean correlations
all_data_df = (
    all_data_df.groupby("participant_id").mean(numeric_only=True).reset_index()
)

cols = []
for type in ["rating", "pain", "cues"]:
    for name, _ in signatures.items():
        cols.append(type + "_" + name + "_cosine_temp_cor")
        cols.append(type + "_" + name + "_cosine_rating_cor")

other_cols = []
for type in ["rating", "pain", "cues"]:
    for name, _ in signatures.items():
        other_cols.append(name + "_" + type + "_cosine")

all_data_df = all_data_df[["participant_id"] + other_cols + cols]

all_data_df.to_csv(opj(basepath, "derivatives", "signatures_corr_df.csv"))


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

univariate_path = opj(basepath, "derivatives", "glm_model_univariate")
files = [f for f in os.listdir(univariate_path) if "t.nii.gz" in f and ".nii.gz" in f]
subs, phases, img_path = [], [], []
for f in tqdm(files):
    subs.append(f.split("_")[0])
    phases.append(f.split("_")[1])
    img_path.append(opj(univariate_path, f))

data = pd.DataFrame({"participant_id": subs, "phase": phases, "imgpath": img_path})

X = []
X += [apply_mask(fname, group_mask) for fname in tqdm(data["imgpath"])]
X = np.squeeze(np.stack(X))
X = X.astype(np.float32)

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
