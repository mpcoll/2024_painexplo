# -*- coding:utf-8 -*-
# @Script: 02-painexplo_behav.py
# @Description: Makes figures and analyses for behavioural data
# TODO : This version uses a simple TD model, add dedicated script for Bayesian model using pymc

import pandas as pd
from os.path import join as opj
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import os
from mne.report import Report
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import urllib.request
import matplotlib.font_manager as fm
import matplotlib

# Root
from painexplo_config import global_parameters  # noqa

param = global_parameters()

# Get nice font
urllib.request.urlretrieve(
    "https://github.com/gbif/analytics/raw/master/fonts/Arial%20Narrow.ttf",
    opj(param.bidspath, "external", "arialnarrow.ttf"),
)
fm.fontManager.addfont("arialnarrow.ttf")
matplotlib.rc("font", family="Arial Narrow")

# Define path to data
dat_path = param.bidspath

# Get participants
exclusion = ["sub-001"]  # Exclude participants with diffrent data structure
part = [p for p in os.listdir(dat_path) if "sub" in p and p not in exclusion]
part.sort()

# Initialize a report
report = Report(title="Pain exploration MRI pilot analyses")
# report.add_code(
#     title="Code for analyses", code=Path("code/02-painexplo_behav.py")
# )

# Intialize a dataframe to store results
out_frame = pd.DataFrame(index=[f.split("_")[0].replace("pilot", "") for f in part])
all_frames_bins_ratings = []
all_frames_bins_intensities = []
for p in part:

    # Create output folder if not existing
    out_path = opj(dat_path, "derivatives", "behav", p)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Get main task file for each plot
    files = [
        f
        for f in os.listdir(opj(dat_path, p, "func"))
        if "events" in f and ".tsv" in f and "._" not in f
    ]
    files.sort()

    # Get temperature recordings
    temp_files = [
        f
        for f in os.listdir(opj(param.bidspath, "sourcedata", p, "maintask"))
        if "_temp_" in f and "._" not in f
    ]
    temp_files.sort()
    targets, peaks = [], []
    for f in temp_files:
        # Read file
        temp_data = pd.read_csv(opj(param.bidspath, "sourcedata", p, "maintask", f))
        targets.append(float(f.split("_")[-1].replace(".csv", "")))
        # Peak temp
        peaks.append(np.max(temp_data[["z1", "z2", "z3", "z4", "z5"]]))

    mean_abs_diff = np.abs(np.asarray(peaks) - np.asarray(targets))
    min_diff = np.min(mean_abs_diff)
    max_diff = np.max(mean_abs_diff)

    # Peaks vs targets regplot
    fig = plt.figure()
    plt.title(p + " - Peaks vs. targets", fontdict={"fontsize": 15})
    sns.regplot(x=targets, y=peaks)
    plt.xlabel("Temperature target", fontsize=12)
    plt.ylabel("Recorded peak temperature", fontsize=12)
    # Add mean and min max as text
    plt.text(
        s="Mean abs diff: " + str(np.round(np.mean(mean_abs_diff), 2)),
        x=0.5,
        y=0.9,
        transform=plt.gca().transAxes,
    )
    plt.text(
        s="Min diff: " + str(np.round(min_diff, 2)),
        x=0.5,
        y=0.85,
        transform=plt.gca().transAxes,
    )
    plt.text(
        s="Max diff: " + str(np.round(max_diff, 2)),
        x=0.5,
        y=0.8,
        transform=plt.gca().transAxes,
    )
    plt.savefig(opj(out_path, p + "_temp_peaks_vs_targets.png"))

    data = []
    for f in files:
        data.append(pd.read_csv(opj(dat_path, p, "func", f), sep="\t"))
    data = pd.concat(data)
    # BIDS files have multiple events per trial, keep only one row per trial
    data = data[data["trial_type"] == "cues"]
    # assert len(data) == 200  # Check we have 200 trials

    # Remove trials where no ratings recorded
    tot_trials = len(data)
    data = data[~(data["rating"] == "None")]
    data = data[~(data["rating"].isna())]

    # Save number of trials without ratings
    out_frame.loc[p, "n_noratings"] = tot_trials - len(data)
    out_frame.loc[p, "instance"] = data["trial_type_instance"].iloc[0]

    n_trials = len(data)

    # Get selected option 0-3
    choices = LabelEncoder().fit_transform(data["selected"])

    # Get detection and tolerance thresholds
    detection = np.asarray(data["pain_detection_ma"])[0]
    tolerance = np.asarray(data["pain_tolerance_ma"])[0]
    out_frame.loc[p, "detection_threshold"] = detection
    out_frame.loc[p, "tolerance_threshold"] = tolerance

    # Get pain intensities between 0 and 1 according to detection/tolerance
    stim_selected = (np.asarray(data["stim_intensite"]) - detection) / (
        tolerance - detection
    )

    # In new version:
    # Get all pain options and scale between 0 and 1

    stim_options_ = np.round(
        np.asarray(data[["intense_1", "intense_2", "intense_3", "intense_4"]]), 1
    )

    stim_options = (stim_options_ - detection) / (tolerance - detection)

    # Get pain ratings
    data["rating"] = data["rating"].astype(float)
    pain_ratings = np.asarray(data["rating"])
    slope, intercept, rvalue = scipy.stats.linregress(stim_selected, pain_ratings)[:3]
    # Save correlation between intensity and ratings
    out_frame.loc[p, "intensity_ratings_r"] = rvalue

    # Use multiple regression to remove variance explained by intensity, trial numbers and previous intensity
    # Use polynomials for potential non-linear effects
    data["stim_intensite_previous"] = np.concatenate([[0], data["stim_intensite"][:-1]])
    data["trial"] = np.arange(len(data))
    reg_frame = data[["stim_intensite", "trial", "stim_intensite_previous"]]
    x = np.asarray(data["stim_intensite"]).reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=5)
    xp = polynomial_features.fit_transform(reg_frame)
    model = sm.OLS(data["rating"], xp).fit()
    model.summary()
    out_frame.loc[p, "olspoly_adjr2"] = model.rsquared_adj
    pain_ratings_residuals = model.resid

    fig = plt.figure()
    sns.regplot(x=data["stim_intensite"], y=model.resid)
    plt.title(
        p + " - Pain ratings residuals vs. selected intensity",
        fontdict={"fontsize": 15},
    )
    plt.xlabel("Stimulus intensity (mA)", fontsize=12)
    plt.ylabel("Pain rating residuals", fontsize=12)
    report.add_figure(title=p + " - Intensity vs. rating residuals", fig=fig, section=p)
    plt.savefig(opj(out_path, p + "_intensity_vs_ratingsresiduals.png"))

    pain_ratings_01 = data["rating"].values / 100  # Pain ratings in 0-1 range

    # SANITY CHECK 1: Plot stim_itensity vs pain ratings

    fig = plt.figure()
    sns.regplot(x=stim_selected, y=pain_ratings)
    plt.title(p + " - Pain ratings vs. selected intensity", fontdict={"fontsize": 15})
    plt.xlabel("Intensity selected (0-1)", fontsize=12)
    plt.ylabel("Pain rating (0-100)", fontsize=12)
    plt.savefig(opj(out_path, p + "_intensity_vs_ratings.png"))
    report.add_figure(
        title=p + " - Pain ratings vs. selected intensity", fig=fig, section=p
    )

    # SANITY CHECK 2: Plot stim_intensity choices
    fig = plt.figure()
    plt.plot(stim_options)
    plt.title(p + " - Intensity for each option", fontdict={"fontsize": 15})
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("Intensity for each option (0-1)", fontsize=12)
    plt.savefig(opj(out_path, p + "_intensityforeachoption.png"))
    report.add_figure(title=p + " - Intensity for each option", fig=fig, section=p)

    # FIT COMPUTATIONAL MODEL (ONLY TD + SOFTMAX FOR NOW)
    # https://www.pymc.io/projects/examples/en/latest/case_studies/reinforcement_learning.html
    # Try reset of initial condiitons
    # Add discount factor
    def llik_td(x, *args):
        """Fit a TD model to the data, to use with scipy.optimize.minimize

        Args:
            x (list): initial guess for alpha and beta

        Returns:
            float : negative log likelihood of the data
        """
        # Extract the arguments as they are passed by scipy.optimize.minimize
        alpha, beta = x
        choices, pains = args

        # Initialize values at 0.5 (no priors)
        Q = np.array([0.5, 0.5, 0.5, 0.5])
        logp_choices = np.zeros(len(choices))

        for t, (a, r) in enumerate(zip(choices, pains)):
            # Apply the softmax transformation
            Q_ = Q * beta
            logp_choice = Q_ - scipy.special.logsumexp(Q_)

            # Store the log probability of the observed action
            logp_choices[t] = logp_choice[a]

            # Update the Q values for the next trial
            Q[a] = Q[a] + alpha * (r - Q[a])

        # Return the negative log likelihood of all observed actions
        return -np.sum(logp_choices[1:])

    def generate_values(choices, outcome, alpha, n=200):
        """Generate values for each option based on TD learning fitted values

        Args:
            choices (1D array): choices made
            stim_selected (1D array): pain intensity received
            alpha (float): learning rate
            n (int, optional): number of trials. Defaults to 200.

        Returns:
            n x 4 : array of values for each option at each trial
        """

        Qs = np.zeros((n + 1, 4))
        delta = np.zeros(n)

        # Initialize Q table
        Q = np.array([0.5, 0.5, 0.5, 0.5])  # Start with no priors
        Qs[0] = Q.copy()
        for i in range(n):
            # Add chocse and pain choice and reward
            a = choices[i]
            r = outcome[i]

            # Update Q table
            delta[i] = r - Q[a]
            Q[a] = Q[a] + alpha * (r - Q[a])

            # Store values
            Qs[i + 1] = Q.copy()

        Qs = Qs[:n]  # Remove the last trial
        return Qs, delta

    # Bayesian model
    # TODO

    # Fit the model
    x0 = [0.5, 5]
    bounds = [(0.0001, 0.9999), (0.0001, np.inf)]  # alpha should be between 0 and 1
    # NOTE pain is negative here to reflect avoidance

    result = scipy.optimize.minimize(
        llik_td, x0, args=(choices, -pain_ratings_01), method="L-BFGS-B", bounds=bounds
    )

    # Make sure the optimization converged
    assert result.success

    # Use the fitted parameters to generate values
    Qs, pes = generate_values(choices, pain_ratings_01, alpha=result.x[0], n=n_trials)

    # Ratings residuals vs prediction error
    fig = plt.figure(figsize=(3, 3))
    sns.regplot(x=pes, y=pain_ratings_residuals)
    plt.title(p, fontdict={"fontsize": 15})
    plt.xlabel("Prediction errors", fontsize=14)
    plt.ylabel("Ratings residuals", fontsize=14)
    plt.tick_params(labelsize=11)
    report.add_figure(title=p + " PEs vs residuals", fig=fig, section=p)
    plt.tight_layout()
    plt.savefig(opj(out_path, p + "_ps_vs_residuals.png"), bbox_inches="tight")

    out_frame.loc[p, "pes_resid_r"] = scipy.stats.pearsonr(pes, pain_ratings_residuals)[
        0
    ]

    # Real vs learned
    fig = plt.figure(figsize=(4, 3))
    for i, c in zip(range(4), sns.color_palette("deep")[:4]):
        plt.plot(Qs[:, i], c=c, label="Learned" if i == 0 else None)
        plt.plot(
            stim_options[:, i],
            c=c,
            alpha=0.6,
            linestyle="--",
            label="Real" if i == 0 else None,
        )
    plt.title(p, fontdict={"fontsize": 15})
    plt.legend(fontsize=11, frameon=False)
    plt.xlabel("Trials", fontsize=14)
    plt.ylabel("Stimulus intensity (0-1)", fontsize=14)
    plt.tick_params(labelsize=11)

    report.add_figure(title=p + " Real vs. learned intensities", fig=fig, section=p)
    plt.savefig(opj(out_path, p + "_real_vs_learned.png"))

    # Values vs choices
    fig = plt.figure()
    for i, c in zip(range(4), sns.color_palette("deep")[:4]):
        plt.plot(
            stim_options[:, i],
            c=c,
            alpha=0.6,
            linestyle="--",
            label="Real" if i == 0 else None,
        )
    plt.title(p + " Actual values and choices", fontdict={"fontsize": 15})
    plt.xlabel("Trials", fontsize=12)
    plt.ylabel("Stimulus intensity (0-1)", fontsize=12)

    count_explo = 0
    count_exploit = 0
    for i in range(len(choices)):

        if np.min(Qs[i]) == Qs[i, choices[i]]:
            color = "gray"
            symbol = "o"
            label = "Exploitation"
            plt.scatter(
                [i],
                [stim_options[i, choices[i]]],
                color=color,
                marker=symbol,
                s=20,
                label=label if count_explo == 0 else None,
                alpha=0.8,
            )
            count_explo += 1

        else:
            color = "gray"
            symbol = "x"
            label = "Exploration"
            plt.scatter(
                [i],
                [stim_options[i, choices[i]]],
                color=color,
                marker=symbol,
                label=label if count_exploit == 0 else None,
                alpha=0.8,
            )
            count_exploit += 1

    plt.legend()
    plt.savefig(opj(out_path, p + "_choices_vs_real.png"))
    report.add_figure(title=p + " Actual values and choices", fig=fig, section=p)

    # Actual intensity vs choices
    fig = plt.figure()
    for i, c in zip(range(4), sns.color_palette("deep")[:4]):
        plt.plot(Qs[:, i], c=c, label="Learned values" if i == 0 else None)
    plt.title(p + " Learned values and choices", fontdict={"fontsize": 15})
    plt.xlabel("Trials", fontsize=12)
    plt.ylabel("Stimulus intensity (0-1)", fontsize=12)

    count_explo = 0
    count_exploit = 0
    for i in range(len(choices)):

        if np.min(Qs[i]) == Qs[i, choices[i]]:
            color = "gray"
            symbol = "o"
            label = "Exploitation"
            plt.scatter(
                [i],
                [Qs[i, choices[i]]],
                color=color,
                marker=symbol,
                label=label if count_explo == 0 else None,
                alpha=0.8,
            )
            count_explo += 1

        else:
            color = "gray"
            symbol = "x"
            label = "Exploration"
            plt.scatter(
                [i],
                [Qs[i, choices[i]]],
                color=color,
                marker=symbol,
                label=label if count_exploit == 0 else None,
                alpha=0.8,
            )
            count_exploit += 1

    plt.legend()
    plt.savefig(opj(out_path, p + "_choices_vs_values.png"))
    report.add_figure(title=p + " Learned values and choices", fig=fig, section=p)

    # % of exploration
    fig = plt.figure()
    trial_type = np.zeros(len(Qs))
    for i in range(len(Qs)):
        trial_type[i] = np.min(Qs[i]) == Qs[i, choices[i]]
    sns.barplot(
        x=["Exploitation", "Exploration"],
        y=[np.mean(trial_type), 1 - np.mean(trial_type)],
    )
    plt.title(p + " % of exploration", fontdict={"fontsize": 15})
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("% of trials", fontsize=12)
    plt.ylim([0, 1])
    report.add_figure(title=p + " % of exploration", fig=fig, section=p)

    trial_type[0] = 0  # First trial is always exploration
    # Pain depending on trial type
    fig = plt.figure()
    sns.barplot(
        x=["Exploitation", "Exploration"],
        y=[
            np.mean(pain_ratings[trial_type == 1]),
            np.mean(pain_ratings[trial_type == 0]),
        ],
    )
    plt.title(p + " Pain ratings depending on trial type", fontdict={"fontsize": 15})
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("Pain ratings", fontsize=12)
    report.add_figure(
        title=p + " Pain ratings depending on trial type", fig=fig, section=p
    )

    data["trial_type"] = trial_type

    # Divide by quantiles to compare pain ratings and intensity
    data["stim_intensite_bin"] = pd.qcut(data["stim_intensite"], q=5, labels=False)

    data_explore = data[data["trial_type"] == 0]
    data_exploit = data[data["trial_type"] == 1]

    data_exploit["stim_intensite"]

    n_bins = len(data["stim_intensite_bin"].unique())
    for bins in range(n_bins):

        data_exploit[data_exploit["stim_intensite_bin"] == bins]["rating"].values
        out_frame.loc[p, "mean_pain_ratings_explore_bin" + str(bins)] = np.mean(
            data_explore[data_explore["stim_intensite_bin"] == bins]["rating"].values
        )
        out_frame.loc[p, "mean_pain_ratings_exploit_bin" + str(bins)] = np.mean(
            data_exploit[data_exploit["stim_intensite_bin"] == bins]["rating"].values
        )
        out_frame.loc[p, "mean_intensity_explore_bin" + str(bins)] = np.mean(
            data_explore[data_explore["stim_intensite_bin"] == bins][
                "stim_intensite"
            ].values
        )
        out_frame.loc[p, "mean_intensity_exploit_bin" + str(bins)] = np.mean(
            data_exploit[data_exploit["stim_intensite_bin"] == bins][
                "stim_intensite"
            ].values
        )

    frame_explore = pd.DataFrame(
        out_frame.loc[
            p, ["mean_pain_ratings_explore_bin" + str(bins) for bins in range(n_bins)]
        ]
    )
    frame_explore["type"] = "Exploration"
    frame_explore["bin"] = np.arange(n_bins)
    frame_explore.rename(columns={p: "mean_pain_ratings"}, inplace=True)

    frame_exploit = pd.DataFrame(
        out_frame.loc[
            p, ["mean_pain_ratings_exploit_bin" + str(bins) for bins in range(n_bins)]
        ]
    )
    frame_exploit["type"] = "Exploitation"
    frame_exploit["bin"] = np.arange(n_bins)
    frame_exploit.rename(columns={p: "mean_pain_ratings"}, inplace=True)

    frame_all_bins_ratings = pd.concat([frame_explore, frame_exploit])
    frame_all_bins_ratings["participant"] = p
    plt.figure()
    sns.barplot(x="bin", y="mean_pain_ratings", hue="type", data=frame_all_bins_ratings)
    plt.savefig(opj(out_path, p + "_mean_ratings_bybin.png"))

    frame_explore = pd.DataFrame(
        out_frame.loc[
            p, ["mean_intensity_explore_bin" + str(bins) for bins in range(n_bins)]
        ]
    )
    frame_explore["type"] = "Exploration"
    frame_explore["bin"] = np.arange(n_bins)
    frame_explore.rename(columns={p: "mean_intensity"}, inplace=True)

    frame_exploit = pd.DataFrame(
        out_frame.loc[
            p, ["mean_intensity_exploit_bin" + str(bins) for bins in range(n_bins)]
        ]
    )
    frame_exploit["type"] = "Exploitation"
    frame_exploit["bin"] = np.arange(n_bins)
    frame_exploit.rename(columns={p: "mean_intensity"}, inplace=True)

    frame_all_bins_intensity = pd.concat([frame_explore, frame_exploit])
    frame_all_bins_intensity["participant"] = p

    all_frames_bins_ratings.append(frame_all_bins_ratings)
    all_frames_bins_intensities.append(frame_all_bins_intensity)

    plt.figure()
    sns.barplot(x="bin", y="mean_intensity", hue="type", data=frame_all_bins_intensity)
    plt.ylim(42, 50)
    plt.savefig(opj(out_path, p + "_mean_intensity_bybin.png"))

    # Correct range
    data["stim_intensite"]
    data["stim_intensite_previous"]
    data_exploit.sort_values("stim_intensite", inplace=True)
    data_explore.sort_values("stim_intensite", inplace=True)
    data_exploit = data_exploit.reset_index()
    data_explore = data_explore.reset_index()

    data_exploit = data_exploit[
        (data_exploit["stim_intensite"] >= data_explore["stim_intensite"].min())
        & (data_exploit["stim_intensite"] <= data_explore["stim_intensite"].max())
    ]

    data_explore = data_explore[
        (data_explore["stim_intensite"] >= data_exploit["stim_intensite"].min())
        & (data_explore["stim_intensite"] <= data_exploit["stim_intensite"].max())
    ]

    out_frame.loc[p, "mean_pain_ratings_exploit_rangecorrected"] = np.mean(
        data_exploit["rating"].values
    )
    out_frame.loc[p, "mean_pain_ratings_explore_rangecorrected"] = np.mean(
        data_explore["rating"].values
    )

    data_explore = data[data["trial_type"] == 0]
    data_exploit = data[data["trial_type"] == 1]
    data_explore["stim_intensite2"] = data_explore["stim_intensite"]
    data_exploit["stim_intensite2"] = data_exploit["stim_intensite"]
    data_explore.sort_values("stim_intensite", inplace=True)
    data_exploit.sort_values("stim_intensite", inplace=True)
    data_exploit = data_exploit.reset_index()
    data_explore = data_explore.reset_index()

    # Keep only trials in the same range
    indexes_to_keep, indexes_to_keep2 = [], []
    if len(data_explore) > len(data_exploit):
        for i in range(len(data_exploit)):
            intensity = data_exploit.loc[i, "stim_intensite"]
            # Find closest intensity
            if np.min(np.abs(data_explore["stim_intensite"] - intensity)) > 0.1:
                pass
            else:
                closest = np.argmin(np.abs(data_explore["stim_intensite"] - intensity))
                indexes_to_keep.append(closest)
                indexes_to_keep2.append(i)
                data_explore.loc[closest, "stim_intensite"] = (
                    999999  # Remove from possible
                )

        data_explore = data_explore.loc[indexes_to_keep]
        data_exploit = data_exploit.loc[indexes_to_keep2]
        assert len(data_explore) == len(data_exploit)
    else:
        for i in range(len(data_explore)):
            intensity = data_explore.loc[i, "stim_intensite"]
            # Find closest intensity
            if np.min(np.abs(data_exploit["stim_intensite"] - intensity)) > 0.1:
                pass
            else:
                closest = np.argmin(np.abs(data_exploit["stim_intensite"] - intensity))
                indexes_to_keep.append(closest)
                indexes_to_keep2.append(i)
                data_exploit.loc[closest, "stim_intensite"] = (
                    999999  # Remove from possible
                )

        data_exploit = data_exploit.loc[indexes_to_keep]

        data_explore = data_explore.loc[indexes_to_keep2]
        assert len(data_explore) == len(data_exploit)

    out_frame.loc[p, "mean_pain_ratings_exploit_matched"] = np.mean(
        data_exploit["rating"].values
    )
    out_frame.loc[p, "mean_pain_ratings_explore_matched"] = np.mean(
        data_explore["rating"].values
    )
    out_frame.loc[p, "mean_intensity_explore_matched"] = np.mean(
        data_explore["stim_intensite2"].values
    )
    out_frame.loc[p, "mean_intensity_exploit_matched"] = np.mean(
        data_exploit["stim_intensite2"].values
    )

    out_frame.loc[p, "n_ratings_matched"] = len(data_explore)

    # Pain residuals depending on trial type
    pain_ratings_residuals = scipy.stats.zscore(pain_ratings_residuals)
    fig = plt.figure()
    sns.barplot(
        x=["Exploitation", "Exploration"],
        y=[
            np.mean(pain_ratings_residuals[trial_type == 1]),
            np.mean(pain_ratings_residuals[trial_type == 0]),
        ],
    )
    plt.xlabel("Actions", fontsize=12)
    plt.ylabel("Pain ratings residuals", fontsize=12)
    report.add_figure(
        title=p
        + " Pain ratings residuals (controlled for intensity) depending on trial type",
        fig=fig,
        section=p,
    )

    out_frame.loc[p, "alpha"] = result.x[0]
    out_frame.loc[p, "beta"] = result.x[1]
    out_frame.loc[p, "detection"] = detection
    out_frame.loc[p, "tolerance"] = tolerance
    out_frame.loc[p, "n_trials"] = n_trials
    out_frame.loc[p, "model_qtd_nllik"] = result.fun
    out_frame.loc[p, "mean_pain_ratings"] = np.mean(pain_ratings)
    out_frame.loc[p, "mean_pain_ratings_residuals"] = np.mean(pain_ratings_residuals)
    out_frame.loc[p, "mean_intensity_exploration"] = np.mean(
        stim_selected[trial_type == 0]
    )
    out_frame.loc[p, "mean_intensity_explotation"] = np.mean(
        stim_selected[trial_type == 1]
    )
    out_frame.loc[p, "mean_pain_ratings_residuals_exploration"] = np.mean(
        pain_ratings_residuals[trial_type == 0]
    )
    out_frame.loc[p, "mean_pain_ratings_residuals_exploitation"] = np.mean(
        pain_ratings_residuals[trial_type == 1]
    )
    out_frame.loc[p, "mean_pain_ratings_exploration"] = np.mean(
        pain_ratings[trial_type == 0]
    )
    out_frame.loc[p, "mean_pain_ratings_exploitation"] = np.mean(
        pain_ratings[trial_type == 1]
    )
    out_frame.loc[p, "percent_exploitation"] = np.mean(trial_type)
    out_frame.loc[p, "percent_exploration"] = 1 - np.mean(trial_type)
    plt.close("all")


out_path = opj(opj(dat_path, "derivatives", "behav"))

# Plot group results
# Percent of exploration
fig = plt.figure()
sns.barplot(x=out_frame.index, y=out_frame["percent_exploitation"])
plt.title("Proportion of exploitation trials by participant", fontdict={"fontsize": 15})
plt.ylabel("% of trials exploitation", fontsize=12)
plt.xlabel("Participant", fontsize=12)
plt.savefig(opj(out_path, "percent_exploitation_group.png"), dpi=800)
plt.xticks(rotation=90)
report.add_figure(
    title="Percent of exploitation by participant", fig=fig, section="group"
)


fig = plt.figure()
sns.barplot(y=out_frame["percent_exploitation"])
sns.stripplot(
    y=out_frame["percent_exploitation"],
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
plt.title("Proportion of exploitation trials by participant", fontdict={"fontsize": 15})
plt.ylabel("% of trials exploitation", fontsize=12)
plt.xlabel("% of exploitation", fontsize=12)
plt.savefig(opj(out_path, "percent_exploitation_group.png"), dpi=800)
plt.xticks(rotation=90)
report.add_figure(title="Percent of exploitation", fig=fig, section="group")


fig = plt.figure()
sns.barplot(x=out_frame["instance"], y=out_frame["percent_exploitation"])
sns.stripplot(
    x=out_frame["instance"],
    y=out_frame["percent_exploitation"],
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
plt.title("Proportion of exploitation by instance", fontdict={"fontsize": 15})
plt.ylabel("% of trials exploitation", fontsize=12)
plt.xlabel("Instance", fontsize=12)
plt.savefig(opj(out_path, "percent_exploitation_byinstance_group.png"), dpi=800)
plt.xticks(rotation=90)
report.add_figure(title="Percent of exploitation by instance", fig=fig, section="group")


fig = plt.figure()
sns.barplot(x=out_frame["instance"], y=out_frame["percent_exploitation"])
sns.stripplot(
    x=out_frame["instance"],
    y=out_frame["percent_exploitation"],
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
plt.title("Proportion of exploitation by instance", fontdict={"fontsize": 15})
plt.ylabel("% of trials exploitation", fontsize=12)
plt.xlabel("Instance", fontsize=12)
plt.savefig(opj(out_path, "percent_exploitation_byinstance_group.png"), dpi=800)
plt.xticks(rotation=90)
report.add_figure(title="Percent of exploitation by instance", fig=fig, section="group")

# Mean pain ratings in exploration vs. exploitation
fig = plt.figure()

plot_frame = out_frame.melt(
    value_vars=[
        "mean_pain_ratings_residuals_exploration",
        "mean_pain_ratings_residuals_exploitation",
    ]
)
sns.barplot(x="variable", y="value", data=plot_frame)
sns.stripplot(
    x="variable",
    y="value",
    data=plot_frame,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
for p in list(out_frame.index):
    plt.plot(
        [0, 1],
        [
            out_frame.loc[p, "mean_pain_ratings_residuals_exploration"],
            out_frame.loc[p, "mean_pain_ratings_residuals_exploitation"],
        ],
        color="gray",
        alpha=0.5,
    )

plt.xticks([0, 1], ["Exploration", "Exploitation"], fontsize=12)
plt.ylabel("Mean z-scored pain ratings residuals", fontsize=15)
plt.xlabel("Choice type", fontsize=15)
plt.title(
    "Mean pain ratings residuals\nin exploration vs. exploitation",
    fontdict={"fontsize": 15},
)
plt.savefig(
    opj(out_path, "mean_pain_ratings_residuals_exploration_vs_exploitation.png"),
    dpi=800,
)
report.add_figure(
    title="Mean pain ratings residuals in exploration vs. exploitation",
    fig=fig,
    section="group",
)


fig = plt.figure()

plot_frame = out_frame.melt(
    value_vars=[
        "mean_pain_ratings_explore_matched",
        "mean_pain_ratings_exploit_matched",
    ]
)
sns.barplot(x="variable", y="value", data=plot_frame)
sns.stripplot(
    x="variable",
    y="value",
    data=plot_frame,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
for p in list(out_frame.index):
    plt.plot(
        [0, 1],
        [
            out_frame.loc[p, "mean_pain_ratings_explore_matched"],
            out_frame.loc[p, "mean_pain_ratings_exploit_matched"],
        ],
        color="gray",
        alpha=0.5,
    )

plt.xticks([0, 1], ["Exploration", "Exploitation"], fontsize=12)
plt.ylabel("Mean z-scored pain ratings residuals - intensity matched", fontsize=15)
plt.xlabel("Choice type", fontsize=15)
plt.title(
    "Mean pain ratings \nin exploration vs. exploitation - intensity matched",
    fontdict={"fontsize": 15},
)
plt.savefig(
    opj(
        out_path, "mean_pain_ratings_residuals_exploration_vs_exploitation_matched.png"
    ),
    dpi=800,
)
report.add_figure(
    title="Mean pain ratings in exploration vs. exploitation - matched",
    fig=fig,
    section="group",
)


fig = plt.figure()

plot_frame = out_frame.melt(
    value_vars=["mean_intensity_explore_matched", "mean_intensity_exploit_matched"]
)
sns.barplot(x="variable", y="value", data=plot_frame)
sns.stripplot(
    x="variable",
    y="value",
    data=plot_frame,
    alpha=0.5,
    size=12,
    jitter=False,
    edgecolor="black",
    linewidth=1,
)
for p in list(out_frame.index):
    plt.plot(
        [0, 1],
        [
            out_frame.loc[p, "mean_intensity_explore_matched"],
            out_frame.loc[p, "mean_intensity_exploit_matched"],
        ],
        color="gray",
        alpha=0.5,
    )

plt.xticks([0, 1], ["Exploration", "Exploitation"], fontsize=12)
plt.ylabel("Mean temperature", fontsize=15)
plt.xlabel("Choice type", fontsize=15)
plt.title(
    "Mean intensity exploration vs. exploitation matched",
    fontdict={"fontsize": 15},
)
plt.savefig(
    opj(out_path, "mean_pain_intensity_exploration_vs_exploitation_matched.png"),
    dpi=800,
)
report.add_figure(
    title="Mean intensity in exploration vs. exploitatio - intensity nmatched",
    fig=fig,
    section="group",
)


out_frame
stats.ttest_rel(
    out_frame["mean_pain_ratings_residuals_exploration"],
    out_frame["mean_pain_ratings_residuals_exploitation"],
)


stats.ttest_rel(
    out_frame["mean_pain_ratings_exploit_matched"],
    out_frame["mean_pain_ratings_explore_matched"],
)


# out_frame = out_frame[~(out_frame.index == 'sub-pilot029')]


# all_frames_bins_ratings = pd.concat(all_frames_bins_ratings)
# all_frames_bins_intensities = pd.concat(all_frames_bins_intensities)


# plt.figure()
# sns.barplot(x="bin", y="mean_pain_ratings", hue="type", data=all_frames_bins_ratings)


# plt.figure()
# sns.barplot(x="bin", y="mean_intensity", hue="type", data=all_frames_bins_intensities)


# out_frame.to_csv(opj(out_path, "pilot_analyses.csv"))
# report.save(opj(out_path, "pilot_analyses.html"), overwrite=True, open_browser=False)


# out_frame[["mean_intensity_explore_matched", "mean_intensity_exploit_matched"]]
