FREQUENCY = 44100
BITRATE = 16
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2
COMP_FRAME_SIZE = 0.040 # Compare 40 ms frames

# The factor by which frames should overlap
FRAME_OVERLAP_FACTOR = .25

# Thresholds for matching.  These are upper bounds to distances tolerated
# MFCC_MATCH_THRESHOLD = 0.1
MFCC_MATCH_THRESHOLD = .4
FFT_MATCH_THRESHOLD = 5
