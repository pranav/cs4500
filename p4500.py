#!/usr/bin/python

import sys
import wave
import numpy
import scipy.io.wavfile
import math

def check_args():
  try:
    return len(sys.argv) == 3 and wave.open(sys.argv[1]) and wave.open(sys.argv[2])
  except IOError:
    sys.stderr.write('ERROR Usage: ./p4500 <path> <path>\n')
    return False
  except Exception:
    sys.stderr.write('ERROR Both files must be of WAV format\n')
    return False


# Normalize to single channel WAV
# Normalize tempo to 100 bpm
# Returns a file path string, example: "/tmp/newfile.wav"
def normalize_wave_file(wavfile):
  # Read the file
  raw_file = read_wav_from_file( wavfile )

  sample_rate = raw_file[0]
  wave_data = raw_file[1]

  print( wave_data )


# numpy.fft should break the file into chunks and perform an fft
# return the array/fft
def get_fft(wavfile):
  pass

# Compare the 2 FFTs using crazy linear distance thingys
#TODO: Better comment
def match_comparator(wav1, wav2):
  pass

# Reads a WAV file from file
# Returns:
#  rate : Int (sample rate of wave file)
#  data : numpy array (data read from wave file)
def read_wav_from_file(path):
  return scipy.io.wavfile.read( path )

# Write a WAV file to file
def write_wav_to_file(path, rate, data):
  scipy.io.wavfile.write( path, rate, data )

# Compute the euclidean distance between two numpy arrays
# Note: if the argument arrays contain complex numbers, the imaginary distance
#       will be eliminated
# Returns:
#   distance: Float (real distance between the two arrays)
def euclidean_distance(arr_1, arr_2):
  if (arr_1.size != arr_2.size):
    sys.stderr.write('INTERNAL ERROR Arrays must be of same size\n')
    return

  # Array for holding differences
  differences = numpy.zeros(arr_1.size, dtype=numpy.complex128)

  # Calculate differences
  for i in range(arr_1.size):
    differences[i] = arr_2[i] - arr_1[i]

  # Square and sum differences
  sq_diffs = [diff * diff for diff in differences]
  sum_sq_diffs = sum(sq_diffs)

  # Return distance value
  # This is where the imaginary portion goes away
  return math.sqrt(sum_sq_diffs)

def main():
  # Check arguments and check if WAV file
  if not check_args():
    sys.exit(1)

  # Returns a file path, "/tmp/newfile.wav"
  wav1 = normalize_wave_file(sys.argv[1])
  wav2 = normalize_wave_file(sys.argv[2])

  fft_wav1 = get_fft(wav1)
  fft_wav2 = get_fft(wav2)

  match = match_comparator(fft_wav1, fft_wav2)

  if match:
    print "MATCH"
  else:
    print "NO MATCH"

if __name__ == '__main__':
  main()
