import os
import sys
import re
import subprocess
import wave

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

# Check if the given file is a WAVE file
def is_wave(path):
  try:
    wave.open(path).close()
    return True
  except (IOError, wave.Error):
    return False

# Check if the given file is an MP3 file
def is_mp3(path):
  return 'MPEG ADTS, layer III' in subprocess.check_output(['file', '-b',
                                                            path])

# Write a WAVE file
def write_wave_to_file(path, rate, data):
  scipy.io.wavfile.write(path, rate, data)

# Get a path for the given file within /tmp/
def get_tmp_path(audio_file):
  filename = re.search(r'[^/]+$', audio_file).group()
  tmp_path = '/tmp/' + filename
  return tmp_path

# Escape a string for use as a single shell token
def quote(s):
  return "'" + s.replace("'", "'\\''") + "'"
