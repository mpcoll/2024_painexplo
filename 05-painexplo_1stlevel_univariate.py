# -*- coding:utf-8 -*-
# @Script: 04-painexplo_makegroupmask.py
# @Description: Univariate analyses. First level and second level models.
# Models are created from BIDS data using nilearn. First level models are
# created for each participant and some contrasts are saved with different
# reports to inspect the model and data. Second level models are created
# for each contrast and saved with different reports and maps.

# An optional neurosynth decode is performed on the contrasts. The decoder
# is loaded from a file created with the neurosynth_prep function in the
# painexplo_config.py file.

from nilearn.image import load_img
from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.second_level import make_second_level_design_matrix, SecondLevelModel
from nilearn.masking import apply_mask, unmask
from os.path import join as opj
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from nltools import Design_Matrix
from mne.report import Report
import matplotlib.pyplot as plt
import seaborn as sns
from atlasreader import create_output
from nltools.stats import fdr
import nimare

# Get global parameters
from painexplo_config import global_parameters  # noqa

param = global_parameters()


# Paths
basepath = opj(param.bidspath)
preppath = opj(param.bidspath, "derivatives/fmriprep")
outpath = opj(param.bidspath, "derivatives/glm_model_univariate")
if not os.path.exists(outpath):
    os.mkdir(outpath)

# Load group mask
group_mask = load_img(opj(param.bidspath, "derivatives/group_mask.nii.gz"))

# Define MNI space
space = "MNI152NLin2009cAsym"

# Get subjects
subs = [s for s in os.listdir(preppath) if "sub" in s and "html" not in s]

# Whether to run neurosynth decode
neurosyth_decode = True
if neurosyth_decode:
    nsynth_decoder = nimare.decode.continuous.CorrelationDecoder.load(
        param.nsynth_decoder_path
    )
    # Adjust number of cores
    nsynth_decoder.n_cores = param.ncpus

# Check if we should overwrite, if not remove already processed
overwrite = True
if not overwrite:
    # Remove already processed
    subs = [s for s in subs if not os.path.exists(opj(outpath, s + "_glm_report.html"))]

###################################################################
# Generate first level models
###################################################################

if subs == []:
    print("All subjects already processed, running only 2nd level")
    # Use this because first level from bids is not working with empty list
    models, imgs, events, confounds = [], [], [], []
else:
    models, imgs, events, confounds = first_level_from_bids(
        dataset_path=param.bidspath,  # Path to BIDS directory
        task_label="painexplo",  # Task
        space_label=space,  # Space
        img_filters=[("desc", "preproc")],  # Use preprocessed images
        high_pass=param.highpass,  # High pass filter (used with default drift model)
        mask_img=group_mask,  # Mask
        slice_time_ref=0.41 / 0.885,  # Slice time reference, as a proportion of TR
        hrf_model="glover",  # Use default hrf
        smoothing_fwhm=param.fwhm,  # Smoothing kernel size
        n_jobs=param.ncpus,  # Number of CPUs to use
        derivatives_folder=preppath,  # Path to derivatives
        sub_labels=[s.replace("sub-", "") for s in subs],  # Parts to process
    )


###################################################################
# Fit models
###################################################################

# Events in events file to not model
do_not_model = [
    "j2_b4pain_cross",
    "j3_aftpain_cross",
    "feedback",
]
# Create a dataframe to store mean framewise displacement
fwd_all = pd.DataFrame(
    index=[s.subject_label for s in models], data=dict(mean_fwd=[999.0] * len(models))
)

# Loop models
for model, img, event, conf in zip(models, imgs, events, confounds):
    s = model.subject_label

    # Create a design matrix
    conf_out, events_out, fwd_sub = [], [], []
    for ev, cf in zip(event, conf):

        if "motion_outliers" in param.confounds:
            # Remove generic motion outliers and add real ones
            confounds = list(param.confounds[:-1]) + list(
                cf.columns[cf.columns.str.contains("motion_outlier")]
            )
        else:
            confounds = param.confounds

        # Select confounds from confounds file
        conf_in = cf[confounds]
        # Replace nan for first scan
        conf_out.append(conf_in.fillna(0))

        # Arrange events
        ev["subject_id"] = s
        # Do not model first cross
        ev = ev[~ev["trial_type"].isin(do_not_model)].reset_index()

        # Add to list
        events_out.append(ev)
        # Get mean framewise displacement for this session
        fwd_sub.append(np.nanmean(cf["framewise_displacement"]))

    # Add mean framewise displacement
    fwd_all.loc[s] = np.mean(fwd_sub)

    # Fit model
    model.fit(img, events_out, conf_out)

    # Create a report with nilearn with some contrasts
    ncols = len(model.design_matrices_[0].columns)
    model.generate_report(
        contrasts=["pain", "cues", "rating", "j1_b4trial_cross"],
    ).save_as_html(opj(outpath, "sub-" + s + "_glm_report.html"))

    # Create a more detailed report with nltools
    report = Report(verbose=True, subject=s, title=s + " first level model")

    # Vifs and more exhaustive report
    vifs = []
    for ridx, d in enumerate(model.design_matrices_):
        # Use same design matrix as nilearn (already convolved)
        dm = Design_Matrix(
            d,  # Design matrix for one run
            sampling_freq=1 / models[0].t_r,  # Sampling frequency
            # Polynomials to include
            polys=[c for c in d.columns.to_list() if "drift" in c] + ["constant"],
        )

        # Calculate vifs
        vifs.append(
            pd.DataFrame(
                dict(
                    vif=dm.vif(),
                    run=ridx + 1,
                    regressor=dm.columns.to_list()[: -len(dm.polys)],
                )
            )
        )

        # Z score matrix for plots
        dm = dm.apply(zscore)
        # Plot design matrix
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(dm, ax=ax, cbar=False, cmap="viridis")
        report.add_figure(
            fig, title="First Level Design Matrix for run " + str(ridx + 1)
        )

        # Plot correlation matrix for regressors
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(dm.corr(), vmin=-1, cmap="viridis")
        report.add_figure(
            fig, title="Events regressor correlation for run" + str(ridx + 1)
        )

        # Plot VIFS for regressors
        fig, ax = plt.subplots(figsize=(10, 15))
        plt.plot(dm.columns[: len(vifs[-1])], vifs[-1]["vif"], linewidth=3)
        plt.xticks(rotation=90)
        plt.ylim(0, 50)
        plt.ylabel("Variance Inflation Factor")
        plt.axhline(2.5, color="g", label="VIF 2.5")
        plt.axhline(4, color="y", label="VIF 4")
        plt.axhline(10, color="r", label="VIF 10")
        plt.legend()
        report.add_figure(fig, title="Events VIF " + str(ridx + 1))
        plt.close("all")

    # Perform contrasts
    all_cols, all_len = [], []
    for ridx, d in enumerate(model.design_matrices_):
        all_cols += d.columns.to_list()
        all_len.append(len(d.columns))

    for contrast in ["pain", "rating", "cues"]:
        # Main effect of pain, rating and cues
        contrast_vector = np.zeros(np.sum(all_len))  # Initialize contrast vector
        # Set contrast vector to 1 for the regressor of interest
        contrast_vector = np.where(np.asarray(all_cols) == contrast, 1, contrast_vector)

        # Split by runs
        contrast_vector = np.split(contrast_vector, np.cumsum(all_len)[:-1])

        # Compute contrast - Betas
        model.compute_contrast(contrast_vector, output_type="effect_size").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + ".nii.gz")
        )

        # Decode with neurosynth
        if neurosyth_decode:
            out = nsynth_decoder.transform(
                opj(outpath, "sub-" + s + "_" + contrast + ".nii.gz")
            ).sort_values(by="r", ascending=False)
            out.to_csv(opj(outpath, "sub-" + s + "_" + contrast + "_nsynth.csv"))

        # Compute contrast - Z scores
        model.compute_contrast(contrast_vector, output_type="z_score").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_z.nii.gz")
        )

        # Compute contrast - T scores
        model.compute_contrast(contrast_vector, output_type="stat").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_t.nii.gz")
        )

        # Same contrasts but vs cross
        contrast_vector = np.zeros(np.sum(all_len))  # Initialize contrast vector
        contrast_vector = np.where(np.asarray(all_cols) == contrast, 1, contrast_vector)

        contrast_vector = np.where(
            np.asarray(all_cols) == "j1_b4trial_cross", -1, contrast_vector
        )

        # Split by runs
        contrast_vector = np.split(contrast_vector, np.cumsum(all_len)[:-1])

        # Contrast
        model.compute_contrast(contrast_vector, output_type="effect_size").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross.nii.gz")
        )
        # Decode with neurosynth
        if neurosyth_decode:
            out = nsynth_decoder.transform(
                opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross.nii.gz")
            ).sort_values(by="r", ascending=False)
            out.to_csv(
                opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross_nsynth.csv")
            )
            report.add_html(
                out.head(50).to_html(),
                title="Neurosynth decode for " + contrast + "_vs_cross",
            )

        model.compute_contrast(contrast_vector, output_type="z_score").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross_z.nii.gz")
        )

        model.compute_contrast(contrast_vector, output_type="stat").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross_t.nii.gz")
        )

    # Save report
    report.save(
        opj(outpath, "sub-" + s + "_glm_report_custom.html"),
        overwrite=True,
        open_browser=False,
    )

# Save mean framewise displacement
fwd_all.to_csv(opj(outpath, "mean_fwd.csv"))


# ###################################################################
# # Second level model
# ###################################################################

# Group contrasts
all_files = os.listdir(outpath)
for condition in [
    "pain",
    "cues",
    "rating",
    "pain_vs_cross",
    "rating_vs_cross",
    "cues_vs_cross",
]:
    # Get all beta maps for this contrast
    mod_files_all = sorted(
        [opj(outpath, f) for f in all_files if condition + ".nii.gz" in f]
    )
    # Create output folder
    outcond = opj(outpath, "2nd_" + condition)
    if not os.path.exists(outcond):
        os.mkdir(outcond)

    # Get subjects
    subs = [f.split("/")[-1][0:7] for f in mod_files_all]
    # Get mean framewise displacement
    fwd = np.asarray(pd.read_csv(opj(outpath, "mean_fwd.csv"))["mean_fwd"])

    # Save 2nd level
    extra_info_subjects = pd.DataFrame({"subject_label": subs})

    # Create design matrix
    dm = make_second_level_design_matrix(subs, extra_info_subjects)
    model = SecondLevelModel(mask_img=group_mask)
    # Fit model
    model.fit(second_level_input=mod_files_all, design_matrix=dm)
    # Generate report
    report = model.generate_report(
        "intercept"
    )  # Intercept is the only contrast (effect against 0)

    # Save t values
    tvals = model.compute_contrast("intercept", output_type="stat")
    pvals = model.compute_contrast("intercept", output_type="p_value")
    tvals.to_filename(opj(outcond, condition + "_tvals.nii"))

    # Decode with neurosynth
    if neurosyth_decode:
        out = nsynth_decoder.transform(
            opj(outcond, condition + "_tvals.nii")
        ).sort_values(by="r", ascending=False)
        out.to_csv(opj(outcond, condition + "_nsynth.csv"))

    # Save report
    report.save_as_html(opj(outcond, "2ndlevel_within" + condition + "_report.html"))

    # Get p_values and t-values in an array
    pvals_msk = apply_mask(pvals, group_mask)
    tvals_msk = apply_mask(tvals, group_mask)

    # Constrast is one tailed so flip pvals if negative and double p-values (see Chen et al. 2019)
    pvals_msk = np.where(tvals_msk < 0, 2 * (1 - pvals_msk), 2 * (pvals_msk))
    # Put back in nifti
    unmask(pvals_msk, group_mask).to_filename(opj(outcond, condition + "_pvals.nii"))

    # Apply FDR and Bonferroni correction
    tval_fdr = np.where(pvals_msk < fdr(pvals_msk, q=0.05), tvals_msk, 0)
    tval_bonf = np.where(pvals_msk * len(pvals_msk) < 0.05, tvals_msk, 0)
    tval_001 = np.where(pvals_msk < 0.001, tvals_msk, 0)
    tval_005 = np.where(pvals_msk < 0.005, tvals_msk, 0)

    unmask(tval_fdr, group_mask).to_filename(
        opj(outcond, condition + "_tvals_fdr05.nii")
    )
    unmask(tval_bonf, group_mask).to_filename(
        opj(outcond, condition + "_tvals_fwe05.nii")
    )
    unmask(tval_001, group_mask).to_filename(
        opj(outcond, condition + "_tvals_unc001.nii")
    )
    unmask(tval_005, group_mask).to_filename(
        opj(outcond, condition + "_tvals_unc005.nii")
    )

    # Run atlasreader to plot clusters
    if not os.path.exists(opj(outcond, "atlas_" + condition + "_p001")):
        os.mkdir(opj(outcond, "atlas_" + condition + "_p001"))
        os.mkdir(opj(outcond, "atlas_" + condition + "_pfdr05"))
        os.mkdir(opj(outcond, "atlas_" + condition + "_pfwe05"))

    create_output(
        opj(outcond, condition + "_tvals_unc001.nii"),
        0,
        outdir=opj(
            opj(outcond, "atlas_" + condition + "_p001"),
        ),
    )
    create_output(
        opj(outcond, condition + "_tvals_fdr05.nii"),
        0,
        outdir=opj(opj(outcond, "atlas_" + condition + "_pfdr05")),
    )
    create_output(
        opj(outcond, condition + "_tvals_fwe05.nii"),
        0,
        outdir=opj(opj(outcond, "atlas_" + condition + "_pfwe05")),
    )
