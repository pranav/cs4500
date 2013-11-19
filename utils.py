import os
import sys
import re
import subprocess
import wave

import scipy.io.wavfile


def read_wave_from_file(path):
  """Reads a WAVE file.

  Args:
    path: A file path to a WAVE file.

  Returns:
    A tuple of a rate (an integer) and data (a NumPy array). The rate is
    the sample rate of WAVE file in hertz. The data is the sound data
    read from the WAVE file.
  """
  with open(os.devnull, 'w') as sys.stdout: # work-around for SciPy bug
    (rate, data) = scipy.io.wavfile.read(path)
  sys.stdout = sys.__stdout__
  return (rate, data)

def write_wave_to_file(path, rate, data):
  """Writes a WAVE file.

  Args:
    path: The output file path.
    rate: The sample rate in hertz.
    data: A 1-D or 2-D NumPy array of either integer or float data-type.
  """
  scipy.io.wavfile.write(path, rate, data)


def get_tmp_path(audio_file):
  """Gets a temporary file path.

  Derives the temporary file name from the given file path.

  Args:
    audio_file: A file path.

  Returns:
    A path for the given file within /tmp.
  """
  filename = re.search(r'[^/]+$', audio_file).group()
  tmp_path = '/tmp/' + filename
  return tmp_path


def quote(s):
  """Escapes a string for use as a single shell token.

  Args:
    s: A string.

  Returns:
    The escaped string.
  """
  return "'" + s.replace("'", "'\\''") + "'"


def is_wave(path):
  """Checks whether a file is in WAVE format.

  Args:
    path: A string.

  Returns:
    True if it the file is a WAVE; otherwise False.
  """
  try:
    wave.open(path).close()
    return True
  except (IOError, wave.Error):
    return False


def is_mp3(path):
  """Checks whether a file is in MP3 format.

  Args:
    path: A string.

  Returns:
    True if it the file is an MP3; otherwise False.
  """
  fileb = subprocess.check_output(['file', '-b', path])

  return ("MPEG ADTS, layer III" in fileb) or ("ID3" in fileb)
