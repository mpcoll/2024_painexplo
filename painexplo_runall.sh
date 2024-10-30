# Run all analyses for the PainExplo project

# Directory structure prior to running this script:
# - root
#   - sourcedata (raw data) (optional)
#   - sub-xx (BIDS data) (if sourcedata is not present)
#   - external (external data and files)

# Requires:
# - Raw data in sourcedata folder (raw data is not shared, but see the 01-painexplo_raw2bids.py script for how to convert raw data to BIDS format)
# - BIDS data in root folder
# - Python environment with requirements installed (see requirements.txt)
# - Docker installed and running and nipreps images for fmriprep and mriqc pulled
# - Brain signatures in external/wager_maps (see https://github.com/canlab/Neuroimaging_Pattern_Masks, contact Tor Wager for NPS)
# - Free surfer license file in external folder (see https://surfer.nmr.mgh.harvard.edu/fswiki/License)
# - Path and ressources correctly set in painexplo_config.py

# Steps:

# 1. Convert raw data to BIDS format (if not already done)
# python 01-painexplo_raw2bids.py || exit 1 # exit if error

# 2. Run behavioural analyses
# python 02-painexplo_behav.py || exit 1 # exit if error

# 2. Run fmriprep mriqc on BIDS data and make groupmask
# python 03-painexplo_preprocess || exit 1 # exit if error
# python 04-painexplo_makegroupmask.py || exit 1 # exit if error

# 3. Run GLMs
python 05-painexplo_1stlevel_univariate.py || exit 1 # exit if error
python 06-painexplo_1stlevel_strials.py || exit 1 # exit if error

# 4. Apply brain signatures
python 07-painexplo_signatures.py || exit 1 # exit if error