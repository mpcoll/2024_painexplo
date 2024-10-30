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

import nilearn as nl
from nilearn.image import load_img
from nilearn.glm.first_level import first_level_from_bids
from os.path import join as opj
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from nltools import Design_Matrix
from mne.report import Report
import matplotlib.pyplot as plt
import seaborn as sns

# import global parameters
from painexplo_config import global_parameters  # noqa

# Turn on to overwrite existing data
overwrite = True

param = global_parameters()

# Define path
preppath = opj(param.bidspath, "derivatives/fmriprep")

# Load group mask
group_mask = load_img(opj(param.bidspath, "derivatives/group_mask.nii.gz"))

# Define MNI space
space = "MNI152NLin2009cAsym"


# Events to model as single trials (Least square all single trials)
strials_events = ["pain", "rating", "cues"]

# Loop over events type
for strials_event in strials_events:

    # Outpath
    outpath = opj(param.bidspath, "derivatives/glm_strials_" + strials_event)
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Get subjects in fmriprep out folder
    subs = [
        s.replace("sub-", "")
        for s in os.listdir(preppath)
        if "sub" in s and "html" not in s
    ]

    # Remove subjects already processed if not overwrite
    if not overwrite:
        subs_done = [
            s.replace("sub-", "")[0:3]
            for s in os.listdir(outpath)
            if "sub" in s and "_report_custom.html" in s
        ]
        subs = [s for s in subs if s not in subs_done]
    ###################################################################
    # Generate models
    ###################################################################

    if subs == []:
        print("All subjects already processed")
        # Use this because first level from bids is not working with empty list
        models, imgs, events, confounds = [], [], [], []
    else:
        # Generate models
        models, imgs, events, confounds = first_level_from_bids(
            dataset_path=param.bidspath,  # Path to BIDS directory
            task_label="painexplo",  # Task
            space_label=space,  # Space
            img_filters=[("desc", "preproc")],  # Use preprocessed images
            high_pass=param.highpass,  # High pass filter
            mask_img=group_mask,  # Mask
            slice_time_ref=0.46327683615819204,  # Slice time reference
            hrf_model="glover",  # Use default hrf
            smoothing_fwhm=param.fwhm,  # Smoothing kernel size
            n_jobs=param.ncpus,  # Number of CPUs to use
            derivatives_folder=preppath,  # Path to derivatives
            sub_labels=subs,  # Parts to process
        )

    # Events in events file to not model
    do_not_model = [
        "j1_b4trial_cross",
        "j2_b4pain_cross",
        "j3_aftpain_cross",
        "feedback",
    ]
    # List to keep mean framewise displacement for each participant
    fwd_all = pd.DataFrame(
        index=[s.subject_label for s in models],
        data=dict(mean_fwd=[999.0] * len(models)),
    )

    ###################################################################
    # Fit models
    ###################################################################
    # Loop over participants and sessions
    for model, img, event, conf in zip(models, imgs, events, confounds):
        s = model.subject_label  # Get subject label

        # Count runs
        run_count = 0
        # List to keep confounds and events
        conf_out, events_out, fwd_sub = [], [], []
        # Loop over runs
        for ev, cf in zip(event, conf):

            # Select confounds from confounds file
            conf_in = cf[param.confounds]

            # Replace nan for first scan
            conf_out.append(conf_in.fillna(0))

            # Arrange events
            ev["subject_id"] = s

            # Do not model some events
            ev = ev[~ev["trial_type"].isin(do_not_model)].reset_index()

            # Keep original trial type for sanity check
            ev["trial_type_original"] = ev["trial_type"].copy()

            # Least-square all single trial
            # Add trial number to get a different regressor for each pain stimulus
            count = 1
            for i in range(len(ev)):
                if strials_event in ev["trial_type"][i]:
                    ev.loc[i, "trial_type"] = ev.loc[i, "trial_type"] + str(
                        int(
                            count
                            + run_count
                            * (len(ev) / len(np.unique(ev["trial_type_original"])))
                        )
                    ).zfill(3)
                    count += 1
            # Append to all events
            events_out.append(ev)

            # Update run count
            run_count += 1

        # Total trials (if some participants have less trials)
        total_trials = int(len(pd.concat(events_out)) / 3)

        # Fit model on all runs
        model.fit(img, events_out, conf_out)

        # Create and save a report with nilearn (to double check)
        report = model.generate_report(
            "trans_x"
        )  # Use trans_x to plot because always present
        report.save_as_html(opj(outpath, "sub-" + s + "_glm_report.html"))

        # Create a more detailed report with nltools
        report = Report(verbose=True, subject=s, title=s + " first level model")

        # Vifs and more exhaustive report
        vifs = []
        for ridx, d in enumerate(model.design_matrices_):
            # Create design matrix
            dm = Design_Matrix(
                d,
                sampling_freq=1 / models[0].t_r,
                polys=[c for c in d.columns.to_list() if "drift" in c] + ["constant"],
            )

            # Calculate VIF
            vifs.append(
                pd.DataFrame(
                    dict(
                        vif=dm.vif(),
                        run=ridx + 1,
                        regressor=dm.columns.to_list()[: -len(dm.polys)],
                    )
                )
            )

            # Apply zscore for plot
            dm = dm.apply(zscore)

            # Plot design matrix
            fig, ax = plt.subplots(figsize=(10, 15))
            sns.heatmap(dm, ax=ax, cbar=False, cmap="viridis")
            report.add_figure(
                fig, title="First Level Design Matrix for run " + str(ridx + 1)
            )

            # Plot correlation matrix between regressors
            fig, ax = plt.subplots(figsize=(10, 15))
            sns.heatmap(dm.corr(), vmin=-1, vmax=1, cmap="viridis")
            report.add_figure(
                fig, title="Events regressor correlation for run" + str(ridx + 1)
            )

            # Plot VIF for each regressor
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

        # Concatenate VIFs
        vifs = pd.concat(vifs)
        # Keep only regressors of interest
        vifs = vifs[vifs["regressor"].str.contains(strials_event)]

        # Save report
        report.save(
            opj(outpath, "sub-" + s + "_glm_report_custom.html"),
            overwrite=True,
            open_browser=False,
        )

        # Calcualte contrasts for each trial and save
        contrast_file = []
        for trials in range(1, total_trials + 1):  # Loop 200 trials
            # Get all the columns name across all sessions
            cols = []
            for dm in model.design_matrices_:
                cols = cols + dm.columns.to_list()

            # Define regressor of interest (single trial)
            regressor = strials_event + str(trials).zfill(3)

            # Single trial contrast (1 0, ....) for this regressor
            contrast_vector = np.zeros(len(cols))
            contrast_vector = np.where(
                np.asarray(cols) == regressor, 1, contrast_vector
            )

            # Split coefficeints by session
            contrast_vector = np.split(contrast_vector, len(events_out))

            # Compute and save contrasts
            model.compute_contrast(
                contrast_vector, output_type="effect_size"
            ).to_filename(opj(outpath, "sub-" + s + "_" + regressor + ".nii.gz"))
            model.compute_contrast(contrast_vector, output_type="z_score").to_filename(
                opj(outpath, "sub-" + s + "_" + regressor + "_z.nii.gz")
            )
            model.compute_contrast(contrast_vector, output_type="stat").to_filename(
                opj(outpath, "sub-" + s + "_" + regressor + "_t.nii.gz")
            )
            # Append to contrast file to save in metadata
            contrast_file.append("sub-" + s + "_" + regressor + "_z.nii.gz")

        # Add files to vifs
        vifs["file"] = contrast_file
        # Save VIFs
        vifs.reset_index(drop=True).to_csv(opj(outpath, "sub-" + s + "_vifs.csv"))
