import numpy

from config import *

def compare(ffts1, ffts2):
  """Compares the FFT decompositions of two files.

  Args:
    ffts1: An iterable of FFT decompositions as NumPy arrays.
    ffts2: Another iterable of FFT decompositions as NumPy arrays.

  Returns:
    True if the entirety of the shorter file can be located in
    sequential order within the longer file; otherwise False.
  """
  # Determine which FFT data is longer
  shorter = ffts1
  longer = ffts2
  if len(ffts1) > len(ffts2):
    shorter = ffts2
    longer = ffts1
  # Set the threshold for a match
  match_threshold = (max(numpy.amax(ffts1[0]), numpy.amax(ffts2[0])) *
                     NORMALIZED_MATCH_THRESHOLD)
  i = 0 # Current chunk of FFT data in smaller file
  j = 0 # Current chunk of FFT data in longer file
  j_prev = 0 # Chunk of FFT data in longer file where sequential matching began
  # Loop invariant
  #  0 <= i <= len(shorter)
  #  0 <= j <= len(longer)
  #  0 <= j_prev <= j
  #  compares FFTs of each of the two FFTs by finding FFTs that sit within the
  #   match_threshold
  while i < len(shorter):
    # Bottom of longer file reached, no match found
    if j == len(longer) or len(shorter) - i > len(longer) - j:
      return False
    # Current examined chunks match
    if distance(shorter[i], longer[j]) < match_threshold:
      if i == 0:
        j_prev = j
      i += 1
      j += 1
    # Current examined chunks do not match
    else:
      i = 0
      j = j_prev + 1
      j_prev = j
  # If here, bottom of smaller file reached, match found
  return True


def euclidean_distance(arr1, arr2):
  """Computes the Euclidean distance between two NumPy arrays.

  Args:
    arr1: A NumPy array.
    arr2: Another NumPy array.

  Returns:
    The distance between the two arrays.
  """
  return numpy.linalg.norm(arr1 - arr2)


def distance(a, b):
  """Computes the mean squared error between two NumPy arrays.

  Args:
    arr1: A NumPy array.
    arr2: Another NumPy array.

  Returns:
    The distance between the two arrays.
  """
  return ((a - b) ** 2).mean()
