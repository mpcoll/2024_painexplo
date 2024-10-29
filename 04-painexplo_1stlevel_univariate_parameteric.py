from nilearn.image import load_img
from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.second_level import make_second_level_design_matrix, SecondLevelModel
from nilearn.plotting import view_img, plot_stat_map
from nilearn.masking import apply_mask, unmask
from nilearn import plotting
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

from painexplo_config import global_parameters  # noqa

param = global_parameters()


# Paths
basepath = opj(param.bidspath)
preppath = opj(param.bidspath, "derivatives/fmriprep")
outpath = opj(param.bidspath, "derivatives/glm_model_univariate_parameteric")
if not os.path.exists(outpath):
    os.mkdir(outpath)
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

# Check if we should overwrite, if not remove already processed
overwrite = False
if not overwrite:
    subs = [s for s in subs if not os.path.exists(opj(outpath, s + "_glm_report.html"))]


# Confounds to use
confounds_use = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "trans_x_derivative1",
    "trans_y_derivative1",
    "trans_z_derivative1",
    "trans_x_power2",
    "trans_y_power2",
    "trans_z_power2",
    "trans_x_derivative1_power2",
    "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1",
    "rot_y_derivative1",
    "rot_z_derivative1",
    "rot_x_power2",
    "rot_y_power2",
    "rot_z_power2",
    "rot_x_derivative1_power2",
    "rot_y_derivative1_power2",
    "rot_z_derivative1_power2",
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
]


###################################################################
# Generate models
###################################################################
# Generate first level models
models, imgs, events, confounds = first_level_from_bids(
    dataset_path=param.bidspath,  # Path to BIDS directory
    task_label="painexplo",  # Task
    space_label=space,  # Space
    img_filters=[("desc", "preproc")],  # Use preprocessed images
    high_pass=1 / 128,  # High pass filter
    mask_img=group_mask,  # Mask
    slice_time_ref=0.46327683615819204,  # Slice time reference
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

        # Select confounds from confounds file
        conf_in = cf[confounds_use]
        # Replace nan for first scan
        conf_out.append(conf_in.fillna(0))

        # Arrange events
        ev["subject_id"] = s
        # Do not model some events
        ev = ev[~ev["trial_type"].isin(do_not_model)].reset_index()

        ev_pain = ev[ev["trial_type"] == "pain"].copy()
        ev_pain["trial_type"] = "pain_mod"
        ev = pd.concat([ev, ev_pain])
        ev.sort_values(by="onset", inplace=True)

        #
        # mean-center modulation to orthogonalize w.r.t. main effect of condition
        temp_modulation = (
            ev["stim_intensite"].values - ev["stim_intensite"].values.mean()
        )
        # Add to events
        ev["modulation"] = np.where(ev["trial_type"] == "pain_mod", temp_modulation, 1)

        events_out.append(ev)
        fwd_sub.append(np.nanmean(cf["framewise_displacement"]))

    # Add mean framewise displacement
    fwd_all.loc[s] = np.mean(fwd_sub)

    # Fit model
    model.fit(img, events_out, conf_out)

    # Create a report with nilearn with some contrasts
    ncols = len(model.design_matrices_[0].columns)
    stophere
    model.generate_report(
        contrasts={
            # Pad manually cause nilearn crashes when padding automatically
            "pain_vs_cross": np.asarray([0, -1, 1, 0, 0] + [0] * (ncols - 5)),
            "painmod_vs_cross": np.asarray([0, -1, 0, 1, 0] + [0] * (ncols - 5)),
            "cues_vs_cross": np.asarray([1, -1, 0, 0, 0] + [0] * (ncols - 5)),
            "rating_vs_cross": np.asarray([0, -1, 0, 0, 1] + [0] * (ncols - 5)),
            "pain": np.asarray([0, 0, 1, 0, 0] + [0] * (ncols - 5)),
            "pain_mod": np.asarray([0, 0, 0, 1, 0] + [0] * (ncols - 5)),
            "cues": np.asarray([1, 0, 0, 0, 0] + [0] * (ncols - 5)),
            "rating": np.asarray([0, 0, 0, 0, 1] + [0] * (ncols - 5)),
            "cross": np.asarray([0, 1, 0, 0, 0] + [0] * (ncols - 5)),
        }
    ).save_as_html(opj(outpath, "sub-" + s + "_glm_report.html"))

    # Create a more detailed report with nltools
    report = Report(verbose=True, subject=s, title=s + " first level model")

    # Vifs and more exhaustive report
    vifs = []
    for ridx, d in enumerate(model.design_matrices_):

        # Use same design matrix as nilearn (already convolved)
        dm = Design_Matrix(
            d,
            sampling_freq=1 / models[0].t_r,
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

        # Z score for plots
        dm = dm.apply(zscore)
        # Plot design matrix
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(dm, ax=ax, cbar=False, cmap="binary")
        report.add_figure(
            fig, title="First Level Design Matrix for run " + str(ridx + 1)
        )

        # Plot correlation matrix for regressors
        fig, ax = plt.subplots(figsize=(10, 15))
        sns.heatmap(dm.corr(), vmin=-1, vmax=1, cmap="RdBu_r")
        report.add_figure(
            fig, title="Events regressor correlation for run" + str(ridx + 1)
        )

        # Plot VIFS
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

    # Save report
    report.save(
        opj(outpath, "sub-" + s + "_glm_report_custom.html"),
        overwrite=True,
        open_browser=False,
    )

    # Perform contrasts
    cols = dm.columns.to_list()
    for contrast in ["pain", "rating", "cues"]:
        # Event vs cross for pain, rating and cues
        contrast_vector = np.zeros(len(cols))
        contrast_vector = np.where(np.asarray(cols) == contrast, 1, contrast_vector)

        # Contrast
        model.compute_contrast(contrast_vector, output_type="effect_size").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + ".nii.gz")
        )
        model.compute_contrast(contrast_vector, output_type="z_score").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + ".nii.gz")
        )

        model.compute_contrast(contrast_vector, output_type="stat").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + ".nii.gz")
        )

        contrast_vector = np.where(
            np.asarray(cols) == "j1_b4trial_cross", -1, contrast_vector
        )

        # Contrast
        model.compute_contrast(contrast_vector, output_type="effect_size").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross.nii.gz")
        )
        model.compute_contrast(contrast_vector, output_type="z_score").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross_z.nii.gz")
        )

        model.compute_contrast(contrast_vector, output_type="stat").to_filename(
            opj(outpath, "sub-" + s + "_" + contrast + "_vs_cross_t.nii.gz")
        )
# Save mean framewise displacement
fwd_all.to_csv(opj(outpath, "mean_fwd.csv"))


# ###################################################################
# # Second level model
# ###################################################################

if neurosyth_decode:
    # Download neurosynth data if not already downloaded
    if not os.path.exists(
        opj(basepath, "external", "neurosynth", "neurosynth_dataset.pkl.gz")
    ):
        files = nimare.extract.fetch_neurosynth(
            data_dir=opj(basepath, "external"),
            overwrite=True,
            source="abstract",
            vocab="terms",
        )
        neurosynth_db = files[0]
        neurosynth_dset = nimare.io.convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        neurosynth_dset.save(
            opj(basepath, "external", "neurosynth", "neurosynth_dataset_LDA100.pkl.gz")
        )
        neurosynth_dset.update_path(opj(basepath, "external", "neurosynth"))
        len(neurosynth_dset.ids)
        # neurosynth_dset = nimare.extract.download_abstracts(
        #     neurosynth_dset, "michelpcoll@gmail.com"
        # )
        # neurosynth_dset.save(
        #     opj(
        #         basepath,
        #         "external",
        #         "neurosynth",
        #         "neurosynth_daaset_with_abstracts.pkl.gz",ru
        #     )
        # )
    # Use pain signature decoder
    decoder = nimare.decode.continuous.CorrelationDecoder(
        feature_group=None,
        features=None,
        frequency_threshold=0.005,
    )
    decoder.fit(neurosynth_dset)
    decoder.save(
        opj(basepath, "external", "neurosynth", "CorrelationDecoder_LAD100.pkl")
    )
    decoder.features_

# Group contrasts
all_files = os.listdir(outpath)
for condition in ["_pain_vs_cross", "_rating_vs_cross", "_cues_vs_cross"]:
    mod_files_all = sorted(
        [opj(outpath, f) for f in all_files if condition + ".nii.gz" in f]
    )
    outcond = opj(outpath, "2nd_" + condition)
    if not os.path.exists(outcond):
        os.mkdir(outcond)

    subs = [f.split("/")[-1][0:7] for f in mod_files_all]
    group = [f.split("/")[-1].split("_")[1] for f in mod_files_all]
    fwd = np.asarray(pd.read_csv(opj(outpath, "mean_fwd.csv"))["mean_fwd"])

    # Save 2nd level
    extra_info_subjects = pd.DataFrame({"subject_label": subs})

    dm = make_second_level_design_matrix(subs, extra_info_subjects)
    model = SecondLevelModel(mask_img=group_mask)
    model.fit(second_level_input=mod_files_all, design_matrix=dm)
    report = model.generate_report("intercept")
    report.save_as_html(opj(outcond, "2ndlevel_within" + condition + "_report.html"))

    tvals = model.compute_contrast("intercept", output_type="stat")
    pvals = model.compute_contrast("intercept", output_type="p_value")
    tvals.to_filename(opj(outcond, condition + "_tvals.nii"))

    pvals_msk = apply_mask(pvals, group_mask)
    tvals_msk = apply_mask(tvals, group_mask)

    # Constrast is one tailed so flip pvals if negative and double p-values (see Chen et al. 2019)
    pvals_msk = np.where(tvals_msk < 0, 2 * (1 - pvals_msk), 2 * (pvals_msk))
    unmask(pvals_msk, group_mask).to_filename(opj(outcond, condition + "_pvals.nii"))

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

    # Add neurosynth decode
    out = decoder.transform(
        "/Volumes/mirexplo/2024_painexplorationMRI/derivatives/glm_model_univariate/2nd__pain_vs_cross/_pain_vs_cross_tvals_unc001.nii"
    )
    out.sort_values(by="r", ascending=False).head(10)
# ##################################################################
# # Slices plot
# ##################################################################

# # Within in CTL group

# # Label size
# labelfontsize = 7
# titlefontsize = np.round(labelfontsize * 1.5)
# ticksfontsize = np.round(labelfontsize * 0.8)
# legendfontsize = np.round(labelfontsize * 0.8)
# bgimg = opj(
#     param.datadrive, "external/tpl-MNI152NLin2009cAsym_space-MNI_res-01_T1w_brain.nii"
# )

# # Load corrected map
# painmapfdr = load_img(
#     opj(outpath, "2nd__PainvsNeutral", "_PainvsNeutral_withinCTL_tvals_unc001.nii")
# )
# view_img(painmapfdr, bg_img=bgimg)
# # PLot slices
# to_plot = {"x": [5, -48, -14, -30, 45, -8], "y": [], "z": [-4, 47]}

# cmap = plotting.cm.cold_hot
# for axis, coord in to_plot.items():
#     for c in coord:
#         fig, ax = plt.subplots(figsize=(1.5, 1.5))
#         disp = plot_stat_map(
#             painmapfdr,
#             cmap=cmap,
#             colorbar=False,
#             bg_img=bgimg,
#             dim=-0.3,
#             black_bg=False,
#             display_mode=axis,
#             axes=ax,
#             vmax=8,
#             cut_coords=(c,),
#             alpha=1,
#             annotate=False,
#         )
#         disp.annotate(size=ticksfontsize, left_right=False)
#         fig.savefig(
#             opj(outpath, "withinCTL_001_" + axis + str(c) + ".svg"),
#             transparent=True,
#             bbox_inches="tight",
#             dpi=600,
#         )


# # Plot a last random one to get the colorbar
# fig, ax = plt.subplots(figsize=(1.5, 1.5))
# thr = np.min(np.abs(painmapfdr.get_fdata()[painmapfdr.get_fdata() != 0]))
# disp = plot_stat_map(
#     painmapfdr,
#     cmap=cmap,
#     colorbar=True,
#     bg_img=bgimg,
#     dim=-0.3,
#     black_bg=False,
#     symmetric_cbar=True,
#     display_mode="x",
#     threshold=thr,
#     axes=ax,
#     vmax=8,
#     cut_coords=(-30,),
#     alpha=1,
#     annotate=False,
# )
# disp.annotate(size=ticksfontsize, left_right=False)
# disp._colorbar_ax.set_ylabel("T value", rotation=90, fontsize=labelfontsize, labelpad=5)
# # lab = disp._colorbar_ax.get_yticklabels()
# # disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)
# disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

# fig.savefig(
#     opj(outpath, "withinCTL_001_slicescbar.svg"),
#     dpi=600,
#     bbox_inches="tight",
#     transparent=True,
# )


# # Load corrected map
# painmapfdr = load_img(
#     opj(outpath, "2nd__PainvsNeutral", "_PainvsNeutral_withinEXP_tvals_unc001.nii")
# )
# view_img(painmapfdr, bg_img=bgimg)
# # PLot slices
# to_plot = {"x": [-12, 41, -36, 59, -48], "y": [], "z": [3]}

# cmap = plotting.cm.cold_hot
# for axis, coord in to_plot.items():
#     for c in coord:
#         fig, ax = plt.subplots(figsize=(1.5, 1.5))
#         disp = plot_stat_map(
#             painmapfdr,
#             cmap=cmap,
#             colorbar=False,
#             bg_img=bgimg,
#             dim=-0.3,
#             black_bg=False,
#             display_mode=axis,
#             axes=ax,
#             vmax=8,
#             cut_coords=(c,),
#             alpha=1,
#             annotate=False,
#         )
#         disp.annotate(size=ticksfontsize, left_right=False)
#         fig.savefig(
#             opj(outpath, "withinEXP_001_" + axis + str(c) + ".svg"),
#             transparent=True,
#             bbox_inches="tight",
#             dpi=600,
#         )


# # Plot a last random one to get the colorbar
# fig, ax = plt.subplots(figsize=(1.5, 1.5))
# thr = np.min(np.abs(painmapfdr.get_fdata()[painmapfdr.get_fdata() != 0]))
# disp = plot_stat_map(
#     painmapfdr,
#     cmap=cmap,
#     colorbar=True,
#     bg_img=bgimg,
#     dim=-0.3,
#     black_bg=False,
#     symmetric_cbar=True,
#     display_mode="x",
#     threshold=thr,
#     axes=ax,
#     vmax=8,
#     cut_coords=(0,),
#     alpha=1,
#     annotate=False,
# )
# disp.annotate(size=ticksfontsize, left_right=False)
# disp._colorbar_ax.set_ylabel("T value", rotation=90, fontsize=labelfontsize, labelpad=5)
# lab = disp._colorbar_ax.get_yticklabels()
# disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)
# disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

# fig.savefig(
#     opj(outpath, "withinEXP_001_slicescbar.svg"),
#     dpi=600,
#     bbox_inches="tight",
#     transparent=True,
# )


# # Load corrected map
# painmapfdr = load_img(
#     opj(outpath, "2nd__PainvsNeutral", "_PainvsNeutral_between_tvals_unc001.nii")
# )
# view_img(painmapfdr, bg_img=bgimg)
# # PLot slices
# to_plot = {"x": [-34, 25, 51, 39], "y": [], "z": [5, 51]}

# cmap = plotting.cm.cold_hot
# for axis, coord in to_plot.items():
#     for c in coord:
#         fig, ax = plt.subplots(figsize=(1.5, 1.5))
#         disp = plot_stat_map(
#             painmapfdr,
#             cmap=cmap,
#             colorbar=False,
#             bg_img=bgimg,
#             dim=-0.3,
#             black_bg=False,
#             display_mode=axis,
#             axes=ax,
#             vmax=8,
#             cut_coords=(c,),
#             alpha=1,
#             annotate=False,
#         )
#         disp.annotate(size=ticksfontsize, left_right=False)
#         fig.savefig(
#             opj(outpath, "between_001_" + axis + str(c) + ".svg"),
#             transparent=True,
#             bbox_inches="tight",
#             dpi=600,
#         )


# # Plot a last random one to get the colorbar
# fig, ax = plt.subplots(figsize=(1.5, 1.5))
# thr = np.min(np.abs(painmapfdr.get_fdata()[painmapfdr.get_fdata() != 0]))
# disp = plot_stat_map(
#     painmapfdr,
#     cmap=cmap,
#     colorbar=True,
#     bg_img=bgimg,
#     dim=-0.3,
#     black_bg=False,
#     symmetric_cbar=True,
#     display_mode="x",
#     threshold=thr,
#     axes=ax,
#     vmax=6,
#     cut_coords=(27,),
#     alpha=1,
#     annotate=False,
# )
# disp.annotate(size=ticksfontsize, left_right=False)
# disp._colorbar_ax.set_ylabel("T value", rotation=90, fontsize=labelfontsize, labelpad=5)
# lab = disp._colorbar_ax.get_yticklabels()
# disp._colorbar_ax.set_yticklabels(lab, fontsize=ticksfontsize)
# disp._colorbar_ax.yaxis.set_tick_params(pad=-0.5)

# fig.savefig(
#     opj(outpath, "between_001_slicescbar.svg"),
#     dpi=600,
#     bbox_inches="tight",
#     transparent=True,
# )
