import os
from os.path import join as opj
import subprocess
import sys
from pathlib import Path

# import global parameters
from painexplo_config import global_parameters  # noqa

param = global_parameters()


def run_mriqc(param):
    """
    Run mriqc on 4 participants at a time only if html output not found.
    """

    outpath = opj(param.bidspath, "derivatives/mriqc")
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
    participants = [to_run[i : i + 4] for i in range(0, len(to_run), 4)]
    # Run four at a time
    for p in participants:
        # Dirty way to take into acount last bit of list if not a multiple of 4

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
            + "nipreps/mriqc:latest /data /out participant "
            "--participant-label "
            + " ".join(p)
            + " --nprocs "
            + str(param.ncpus)
            + " --mem-gb 64"
            + " --no-sub",
        )

        # Run participants
        print(command)
        os.system(command)

    # Run at the group level to get group reports
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
        "--nprocs " + str(param.ncpus) + " --mem-gb 64" + " --no-sub",
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

    participants = [to_run[i : i + 4] for i in range(0, len(to_run), 4)]

    # Run four at a time
    for p in participants:
        # Dirty way to take into acount last bit of list if not a multiple of 4

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
            "--participant-label "
            + " ".join(p)
            + " --output-spaces MNI152NLin2009cAsym:res-2 --fs-no-reconall --skip-bids-validation"
        )

        # Run participants
        print(command)
        os.system(command)


run_fmriprep(param)
# run_mriqc(param)
