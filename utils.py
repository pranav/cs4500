"""

`utils.py` defines utility functions that are used throughout the project
and are logically separate from other modules.

"""


import os
import sys
import subprocess
import tempfile
import wave

from mutagen.mp3 import MP3
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


def get_tmp_file(path):
  """Gets a new temporary file.

  Derives the temporary file name from the given file path.

  Args:
    path: A file path.

  Returns:
    A temporary file within /tmp.
  """
  tmp_file = tempfile.NamedTemporaryFile(suffix=os.path.basename(path),
                                         dir='/tmp')
  try:
    os.chmod(tmp_file.name, 0o660)
  except OSError:
    pass
  return tmp_file


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
    path: The path to a file.

  Returns:
    True if the file is a WAVE; otherwise False.
  """
  fileb = subprocess.check_output(['file', '-b', path])
  if 'WAVE audio' in fileb:
      return True;
  else:
      try:
          wave.open(path).close()
          return True
      except (IOError, EOFError, wave.Error):
          return False


def is_mp3(path):
  """Checks whether a file is in MP3 format.

  Will first use unix `file` program to check for obvious header.
  If not definite match, use Python library mutagen.MP3 to get the
  file's MIME type.

  Args:
    path: A string.

  Returns:
    True if it the file is an MP3; otherwise False.
  """
  fileb = subprocess.check_output(['file', '-b', path])
  if 'MPEG ADTS, layer III' in fileb:
      return True;
  elif 'ID3' in fileb:
      audio = MP3(path)
      return 'audio/mp3' in audio.mime
  else:
      return False
