"""

`comparator.py` contains functions that are used to compare two of the
software's normalized WAVE audio files.

"""


import numpy

from config import *


def compare(arrs1, arrs2, match_threshold):
  """Finds a fuzzy match of one iterable of NumPy arrays within another.

  Args:
    arrs1: An iterable of NumPy arrays.
    arrs2: Another iterable of NumPy arrays.
    match_threshold: The maximum distance between two matching arrays.

  Returns:
    -1 if the entirety of the first iterable can be located in
    sequential order within the longer iterable; 1 if the second is
    within the first; otherwise 0.
  """
  shorter = arrs1
  longer = arrs2
  if len(arrs1) > len(arrs2):
    shorter = arrs2
    longer = arrs1
  # match_threshold = match_threshold * max(numpy.amax(arrs1[0]),
  #                                        numpy.amax(arrs2[0]))
  i = 0 # Current index in shorter iterable
  j = 0 # Current index in longer iterable
  j_prev = 0 # Index in longer file where sequential matching began
  # Loop invariant
  #  0 <= i <= len(shorter)
  #  0 <= j <= len(longer)
  #  0 <= j_prev <= j
  #  Compares arrays from each of the two iterables by comparing their
  #   distance to match_threshold
  while i < len(shorter):
    # End of longer iterable reached: no match found
    if j == len(longer) or len(shorter) - i > len(longer) - j:
      return 0
    # Current examined arrays match
    if distance(shorter[i], longer[j]) < match_threshold:
      if i == 0:
        j_prev = j
      i += 1
      j += 1
    # Current examined arrays do not match
    else:
      i = 0
      j = j_prev + 1
      j_prev = j
  # End of shorter iterable reached: match found
  if shorter is arrs1:
    return -1
  return 1


def euclidean_distance(arr1, arr2):
  """Computes the Euclidean distance between two NumPy arrays.

  Args:
    arr1: A NumPy array.
    arr2: Another NumPy array.

  Returns:
    The distance between the two arrays.
  """
  return numpy.linalg.norm(arr1 - arr2)


def distance(arr1, arr2):
  """Computes the mean squared error between two NumPy arrays.

  Args:
    arr1: A NumPy array.
    arr2: Another NumPy array.

  Returns:
    The distance between the two arrays.
  """
  return ((arr1 - arr2) ** 2).mean()
