#!/usr/bin/python

import audioop
import os
import re
import wave
import sys

import numpy
import numpy.fft
import scipy.io.wavfile

# Local modules
import config
import utils
import comparator
import normalize


FREQUENCY = 44100
BITRATE = 16
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2
COMP_CHUNK_SIZE = 0.5 # Compare a half-second at a time
NORMALIZED_MATCH_THRESHOLD = 5

def check_args():
  try:
    assert len(sys.argv) == 3
    wave.open(sys.argv[1]).close()
    wave.open(sys.argv[2]).close()
    return True
  except (AssertionError, IndexError, IOError):
    sys.stderr.write('ERROR Usage: {0} <path> <path>\n'.format(sys.argv[0]))
    return False
  except wave.Error:
    sys.stderr.write('ERROR Both files must be of WAVE format\n')
    return False


def main():
  # Check arguments and check if WAVE file
  if not check_args():
    sys.exit(1)

  # Returns a file path, "/tmp/newfile.wav"
  audio_file1 = normalize.normalize_wave_file(sys.argv[1])
  audio_file2 = normalize.normalize_wave_file(sys.argv[2])

  ffts1 = normalize.get_fft(audio_file1)
  ffts2 = normalize.get_fft(audio_file2)

  match = comparator.compare(ffts1, ffts2)

  if match:
    print "MATCH"
  else:
    print "NO MATCH"

if __name__ == '__main__':
  main()
