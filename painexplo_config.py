# -*- coding:utf-8 -*-
# @Script: painexplo_config.py
# @Description: Global parameters and helper functions
# for the pain exploration analyses

from os.path import join as opj
import nimare
import os
import numpy as np
from scipy.spatial.distance import cosine


# Global parameters
class global_parameters:
    def __init__(self):
        # Path
        self.bidspath = (
            "/Volumes/mirexplo/2024_painexplorationMRI"  # Path to BIDS directory
        )
        # Neurosynth decoder path
        self.nsynth_decoder_path = opj(
            self.bidspath,
            "external",
            "neurosynth",
            "CorrelationDecoder_allterms.pkl",
        )

        # Ressources
        self.ncpus = 4  # Number of CPUs to use
        self.mem_gb = 64  # Memory in GB

        # GLM
        self.fwhm = 6  # Smoothing kernel size in GLM
        self.highpass = 1 / 100  # High pass filter in GLM
        self.confounds = [
            "trans_x",  # Friston 24 motion parameters
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
            "csf",  # CSF signal
            "motion_outliers",
        ]  # Confounds to use at first level in GLM


# Neurosynth data preparation
def neurosynth_prep(basepath):
    """
    Download Neurosynth data and create correlation decoder if does not exist
    """
    if not os.path.exists(
        opj(basepath, "external", "neurosynth", "CorrelationDecoder_allterms.pkl")
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
            opj(basepath, "external", "neurosynth", "neurosynth_dataset_terms.pkl.gz")
        )

        decoder = nimare.decode.continuous.CorrelationDecoder(
            feature_group=None,
            features=None,
            frequency_threshold=0.005,
        )
        decoder.fit(neurosynth_dset)
        decoder.save(
            opj(basepath, "external", "neurosynth", "CorrelationDecoder_allterms.pkl")
        )


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
