from os.path import join as opj
import nimare
import os


class global_parameters:
    def __init__(self):
        self.bidspath = (
            "/Volumes/mirexplo/2024_painexplorationMRI"  # Path to BIDS directory
        )
        self.ncpus = 4  # Number of CPUs to use
        self.fwhm = 6  # Smoothing kernel size
        self.nsynth_decoder_path = opj(
            self.bidspath,
            "external",
            "neurosynth",
            "CorrelationDecoder_allterms.pkl",
        )


param = global_parameters()


def neurosynth_prep(basepath=param.bidspath):
    """
    Download Neurosynth data and prepare decoder if does not exist
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
