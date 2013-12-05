"""

`config.py` contains configuration variables that are used throughout
the software.

Some of these variables are used when computing the canonicalized or
normalized WAVE audio:

    * FREQUENCY
    * BITRATE
    * NUM_CHANNELS
    * SAMPLE_WIDTH
    * COMP_FRAME_SIZE
    * FRAME_OVERLAP_FACTOR

And others are used as controlling parameters for comparisons and
distance calculations:

    * SIGNIFICANT_MFCC
    * MFCC_MATCH_THRESHOLD
    * FFT_MATCH_THRESHOLD

"""


# Configuration normalized form
FREQUENCY = 44100
BITRATE = 16
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2
COMP_FRAME_SIZE = 0.040 # Compare 40 ms frames

# The factor by which frames should overlap
FRAME_OVERLAP_FACTOR = .25

# The fraction of MFCCs to consider when calculating difference
SIGNIFICANT_MFCC = .25

# Thresholds for matching.  These are upper bounds to distances tolerated
MFCC_MATCH_THRESHOLD = .4
FFT_MATCH_THRESHOLD = 5
