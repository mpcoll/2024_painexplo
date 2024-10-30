# -*- coding:utf-8 -*-
# @Script: 03-painexplo_preprocess.py
# @Description: Runs fmriprep and mriqc on BIDS data
# Runs four participants at a time until all participants are processed;
# this seems like the optimal number of participants to run at a time but
# can be changed below.
# MRIQC is run first at the participant level and then at the group level
# Fmriprep is run at the participant level only
# This script necessitates a functional docker installation with the nipreps
# images available for mriqc and fmriprep

import os
from os.path import join as opj

# import global parameters
from painexplo_config import global_parameters  # noqa

param = global_parameters()
# Number of participants to run at a time
num_part_run = 4


def run_mriqc(param):
    """
    Run mriqc on 4 participants at a time only if html output not found.
    """

    # Set paths
    outpath = opj(param.bidspath, "derivatives/mriqc")
    # License file for freesurfer
    ls_path = opj(param.bidspath, "external/fs_license.txt")

    # Get participants
    participants = [s for s in os.listdir(opj(param.bidspath)) if "sub-" in s]

    # IF first run make dir
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        finished_participants = []
    else:
        # Else check which participants are finished by checking for html report files
        finished_participants = [
            s.split(".")[0] for s in os.listdir(opj(outpath)) if ".html" in s
        ]

    # Participants to run
    to_run = [s for s in participants if s not in finished_participants]

    # Create a list of lists of participants to run
    participants = [
        to_run[i : i + num_part_run] for i in range(0, len(to_run), num_part_run)
    ]
    # Run four at a time
    for p in participants:
        # Dirty way to take into acount last bit of list if not a multiple of 4

        print("Running mriqc on " + " ".join(p))

        # Command to run
        command = (
            "docker run --rm -v "
            + param.bidspath
            + ":/data "
            + "-v "
            + outpath
            + ":/out "
            + "-v "
            + ls_path
            + ":/opt/freesurfer/license.txt "
            + "nipreps/mriqc:latest /data /out participant "
            "--participant-label "
            + " ".join(p)
            + " --nprocs "
            + str(param.ncpus)
            + " --mem-gb "
            + str(param.mem_gb)
            + " --no-sub",
        )

        # Run participants
        print(command)
        os.system(command)

    # After all participants are run at the group level to get group reports
    command = (
        "docker run --rm -v "
        + param.bidspath
        + ":/data "
        + "-v "
        + outpath
        + ":/out "
        + "-v "
        + ls_path
        + ":/opt/freesurfer/license.txt "
        + "nipreps/mriqc:latest /data /out group "
        "--nprocs " + str(param.ncpus) + " --mem-gb " + str(param.mem_gb) + " --no-sub",
    )


def run_fmriprep(param):
    """
    Run fmriprep on 4 participants at a time only if html output not found.
    """

    outpath = opj(param.bidspath, "derivatives/fmriprep")
    ls_path = opj(param.bidspath, "external/fs_license.txt")

    participants = [s for s in os.listdir(opj(param.bidspath)) if "sub-" in s]
    # IF first run make dir
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        finished_participants = []
    else:
        # Else check which participants are finished
        finished_participants = [
            s.split(".")[0] for s in os.listdir(opj(outpath)) if ".html" in s
        ]

    to_run = [s for s in participants if s not in finished_participants]
    # Command to run and print output

    participants = [
        to_run[i : i + num_part_run] for i in range(0, len(to_run), num_part_run)
    ]

    # Run four at a time
    for p in participants:
        print("Running fmriprep on " + " ".join(p))

        command = (
            "docker run --rm -v "
            + param.bidspath
            + ":/data "
            + "-v "
            + outpath
            + ":/out "
            + "-v "
            + ls_path
            + ":/opt/freesurfer/license.txt "
            + "nipreps/fmriprep:latest /data /out participant "
            "--participant-label " + " ".join(p)
            # Output space is in MNI152NLin2009cAsym with 2mm resolution
            + " --output-spaces MNI152NLin2009cAsym:res-2 --fs-no-reconall --skip-bids-validation"
        )

        # Run participants
        print(command)
        os.system(command)


run_fmriprep(param)
run_mriqc(param)
