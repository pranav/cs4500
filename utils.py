import os
import sys
import re

import scipy.io.wavfile


# Reads a WAVE file
# Returns:
#  rate : Int (sample rate of WAVE file)
#  data : numpy array (data read from WAVE file)
def read_wave_from_file(path):
  with open(os.devnull, 'w') as sys.stdout: # work-around for SciPy bug
    (rate, data) = scipy.io.wavfile.read(path)
  sys.stdout = sys.__stdout__
  return (rate, data)


# Write a WAVE file
def write_wave_to_file(path, rate, data):
  scipy.io.wavfile.write(path, rate, data)

# Get a path for the given file within /tmp/
def get_tmp_path(audio_file):
  filename = re.search(r'[^/]+$', audio_file).group()
  tmp_path = '/tmp/' + filename
  return tmp_path
