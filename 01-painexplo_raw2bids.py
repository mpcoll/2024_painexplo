# -*- coding:utf-8 -*-
# @Script: 01-painexplo_raw2bids.py
# @Description: Takes raw data located in sourcedata and converts it to BIDS format
# using dcm2bids and psychopy data
# TODO : Add README file and participants.tsv with questionnaires

import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json

# Get global parameters from config file
from painexplo_config import global_parameters  # noqa

param = global_parameters()

# Get sourcedata directory
raw_dir = opj(param.bidspath, "sourcedata")

# Get participants
parts = [s for s in os.listdir(raw_dir) if "sub-" in s]
parts.sort()

# Overwrite flag
overwrite = False

# Loop over participants
for p in parts:
    # Get participant directory
    part_dir = opj(raw_dir, p)
    # Dicoms to bids config file
    config_file = opj(param.bidspath, "code", "dcm2nii_config.json")

    # Generate dcm2bids command
    command = (
        "dcm2bids -d "
        + opj(part_dir, "DICOM")
        + " -p "
        + p
        + " -c "
        + config_file
        + " -o "
        + param.bidspath
    )
    # Skip flag to manage owverwriting
    skip = False

    # Check if BIDS directory already exists
    # If it does, check if we should overwrite it
    # If we should, add the forceDcm2niix flag
    if os.path.exists(opj(param.bidspath, p)):
        if overwrite:
            print("Overwriting existing BIDS directory for participant " + p)
            command = command + " --forceDcm2niix"
            out = os.system(command)
        else:
            print("BIDS directory already exists for participant " + p)
            out = 0
            # Switch skip flag to True to avoid reprocessing
            skip = True
    else:
        # If BIDS directory does not exist, run
        out = os.system(command)

    # Remove the relative bids URL in intendedor not yet supported by
    # fmriprep (see https://neurostars.org/t/fmriprep-susceptibility-distortion-correction-none/19366/11)
    # This is a temporary fix until fmriprep supports relative paths
    ################################################
    if not skip:
        fmapjson = opj(param.bidspath, p, "fmap", p + "_phasediff.json")
        with open(fmapjson, "r") as f:
            data = json.load(f)
            x = data["IntendedFor"]
            x = [i.replace("bids::" + p + "/", "") for i in x]
            data["IntendedFor"] = x

        os.remove(fmapjson)
        with open(fmapjson, "w") as f:
            json.dump(data, f, indent=4)
        #########################
        # Psychopy data to BIDS onsets
        # Get all behavioral files
        behav_file = [
            f
            for f in os.listdir(opj(part_dir, "maintask"))
            if "csv" in f
            and "maintask" in f
            and "key" not in f
            and "temp" not in f
            and "._" not in f
        ]

        # Make sure there is only one task file
        assert len(behav_file) == 1, (
            "More than one behavioral file found for participant " + p
        )
        # Get the file
        behav_file = behav_file[0]
        # Load the file
        beh = pd.read_csv(opj(part_dir, "maintask", behav_file))
        # Drop rows with NaNs in bloc_loop.thisRepN
        beh = beh.dropna(subset=["bloc_loop.thisRepN"]).reset_index(drop=True)

        # Loop over blocks
        for block in beh["bloc_loop.thisRepN"].unique():
            # Get block data
            beh_block = beh[beh["bloc_loop.thisRepN"] == block]

            # Drop rows with NaNs in trials.thisN
            beh_block = beh_block.dropna(subset=["trials.thisN"]).reset_index(drop=True)

            # Drop rows with NaNs in wait_time_routine.started
            beh_block = beh_block.dropna(subset=["trials.thisRepN"]).reset_index(
                drop=True
            )

            # Make sure there are 20 trials
            assert len(beh_block) == 20, (
                "Block " + str(block) + " has " + str(len(beh_block)) + " trials."
            )

            # Repeat each row 7 times because BIDS requires one row per event
            # and there are 7 events per trial
            beh_long = pd.DataFrame(np.repeat(beh_block.values, 7, axis=0))
            beh_long.columns = beh_block.columns

            # Rename columns
            beh_long["trial_type_instance"] = beh_long["trial_type"]
            beh_long.drop("trial_type", axis=1, inplace=True)
            # Events to get onsets and durations
            # 1 - Jitter 1
            # 2 - cue
            # 3 - response/feedback
            # 4- Jitter 2
            # 5 - stimulation
            # 6- Jitter 3
            # 7 - rating
            # Onsets
            # Jitter 1
            # The first onset is the first cross after the scanner trigger
            # After that the onsets of the ISI are the eval.residual + jitter1
            beh_block["j1_b4trial_cross_onset"] = [
                beh_block["wait_trial_routine.started"][0]
            ] + list(beh_block["eval_residuel.started"][:-1])
            # Cues
            beh_block["cues_onset"] = beh_block["cue_1.started"]
            # Feedback
            beh_block["feedback_onset"] = beh_block["red_feedback_apparition"]
            # Jitter 2
            beh_block["j2_b4pain_cross_onset"] = beh_block[
                "j2_b4pain_cross_textcomponents.started"
            ]
            # Pain
            beh_block["pain_onset"] = beh_block["pain_cross.started"]
            # Jitter 3
            beh_block["j3_aftpain_cross_onset"] = beh_block[
                "j3_aftpain_cross_textcomponents.started"
            ]
            # Rating
            beh_block["rating_onset"] = beh_block["Rating.started"]

            # Durations
            # Jitter 1 duration is end of cross - beginning of cross already calculated
            beh_block["j1_b4trial_cross_duration"] = (
                beh_block["j1_b4trial_cross_textcomponents.stopped"]
                - beh_block["j1_b4trial_cross_onset"]
            )

            # Feedback duration is always 1
            beh_block["feedback_duration"] = 1
            # Jitter 2 duration
            beh_block["j2_b4pain_cross_duration"] = (
                beh_block["j2_b4pain_cross_textcomponents.stopped"]
                - beh_block["j2_b4pain_cross_textcomponents.started"]
            )
            # Jitter 3 duration
            beh_block["j3_aftpain_cross_duration"] = (
                beh_block["j3_aftpain_cross_textcomponents.stopped"]
                - beh_block["j3_aftpain_cross_textcomponents.started"]
            )
            # Cues duration
            beh_block["cues_duration"] = (
                beh_block["feedback_onset"] - beh_block["cue_1.started"]
            )
            # Pain duration is always 3.5
            beh_block["pain_duration"] = 3.5
            # Rating duration
            beh_block["rating_duration"] = (
                beh_block["Rating.stopped"] - beh_block["Rating.started"]
            )

            # Merge all onsets and durations
            beh_onsets = beh_block[
                [
                    "j1_b4trial_cross_onset",
                    "cues_onset",
                    "feedback_onset",
                    "j2_b4pain_cross_onset",
                    "pain_onset",
                    "j3_aftpain_cross_onset",
                    "rating_onset",
                ]
            ].melt(var_name="trial_type", value_name="onset")

            beh_onsets["trial_type"] = beh_onsets["trial_type"].str.replace(
                "_onset", ""
            )
            beh_onsets.reset_index(drop=True, inplace=True)

            beh_durations = beh_block[
                [
                    "j1_b4trial_cross_duration",
                    "cues_duration",
                    "feedback_duration",
                    "j2_b4pain_cross_duration",
                    "pain_duration",
                    "j3_aftpain_cross_duration",
                    "rating_duration",
                ]
            ].melt(var_name="trial_type", value_name="duration")

            beh_durations.reset_index(drop=True, inplace=True)

            beh_onsets["duration"] = beh_durations["duration"]

            # Recenter to first cross after scanner trigger
            beh_onsets["onset"] = beh_onsets["onset"] - beh_onsets["onset"][0]
            beh_onsets = beh_onsets.sort_values(by=["onset"]).reset_index(drop=True)

            # Add to the long dataframe
            beh_long.insert(0, "onset", beh_onsets["onset"])
            beh_long.insert(1, "duration", beh_onsets["duration"])
            beh_long.insert(2, "trial_type", beh_onsets["trial_type"])
            beh_long.insert(3, "participant_id", p)
            beh_long.insert(4, "rawfile", behav_file)
            beh_long.sort_values(by=["onset"], inplace=True)
            beh_long.reset_index(drop=True, inplace=True)

            # Replace empty/NaN with "n/a" for bids compliance
            beh_long.replace("", "n/a", inplace=True)
            beh_long.fillna("n/a", inplace=True)
            # Remove unnamed columns
            beh_long = beh_long.loc[:, ~beh_long.columns.str.contains("^Unnamed")]

            # Save to bids path
            beh_long.to_csv(
                opj(
                    param.bidspath,
                    p,
                    "func",
                    p
                    + "_task-painexplo_run-"
                    + str(int(block) + 1).zfill(2)
                    + "_events.tsv",
                ),
                sep="\t",
                index=False,
            )

# Write dataset_description.json for BIDS compliance
# Data set level file
dataset_description = {
    "Name": "PainExploration fMRI",
    "BIDSVersion": "v1.8.0",
    "Authors": ["Roy N", "Coll MP"],
    "EthicsApprovals": ["CIUSS-CN 2024-2928"],
    "License": "CC-BY-4.0",
}

with open(opj(param.bidspath, "dataset_description.json"), "w") as f:
    json.dump(dataset_description, f, indent=4)
