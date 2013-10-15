#!/usr/bin/python

import audioop
import os
import re
import wave
import sys

import numpy
import numpy.fft
import scipy.io.wavfile


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


# Normalize to single channel WAVE
# TODO: Normalize tempo to 100 bpm
# Returns a file path string, example: "/tmp/newfile.wav"
def normalize_wave_file(audio_file):
  global FREQUENCY, NUM_CHANNELS, SAMPLE_WIDTH

  filename = re.search(r'[^/]+$', audio_file).group()
  output_path = '/tmp/' + filename + '_norm'

  # Read the file
  wf = wave.open(audio_file, 'rb')

  # Get the WAVE file parameters and read data
  (nchannels, sampwidth, framerate, nframes, comptype, compname) = \
    wf.getparams()

  frames = wf.readframes(nframes)

  wf.close()

  # Convert to mono if file is stereo
  if nchannels == 2:
    frames = audioop.tomono(frames, sampwidth, 1, 1)

  # Create a copy of it with new parameters
  wf = wave.open(output_path, 'wb')
  wf.setparams((NUM_CHANNELS, SAMPLE_WIDTH, FREQUENCY, nframes, "NONE", "NONE"))
  wf.writeframes(frames)

  wf.close()

  return output_path


# numpy.fft should break the file into chunks and perform an fft
# return the array/fft
def get_fft(audio_file):
  global COMP_CHUNK_SIZE

  # Read the file, and determine its length in 'chunks'
  (sample, data) = read_wave_from_file(audio_file)
  total_chunks = (data.size / sample) / COMP_CHUNK_SIZE

  # Allocate space for the FFT decompsitions of each chunk of sound data
  fft_out = numpy.ndarray(shape=(total_chunks, sample*COMP_CHUNK_SIZE), 
                          dtype=numpy.complex128)

  # Loop through all chunks, computing their FFT decompositions
  chunk = 0
  while chunk < total_chunks:
    fft = numpy.fft.fft(data[chunk*COMP_CHUNK_SIZE*sample : (chunk+1)*COMP_CHUNK_SIZE*sample])
    fft_out[chunk] = fft
    chunk += 1

  return fft_out


# Compare the FFT decomposition of one file to the FFT decompsition of another
# Return True if the entirety of the shorter file can be located in sequential
# order within the longer file
# Otherwise return False
def compare(ffts1, ffts2):
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
  while i < len(shorter):
    # Bottom of longer file reached, no match found
    if j == len(longer) or len(shorter) - i > len(longer) - j:
      return False
    # Current examined chunks match
    if euclidean_distance(shorter[i], longer[j]) < match_threshold:
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


# Compute the Euclidean distance between two numpy arrays
# Returns:
#   distance: Float (real distance between the two arrays)
def euclidean_distance(arr1, arr2):
  return numpy.linalg.norm(arr1 - arr2)


def main():
  # Check arguments and check if WAVE file
  if not check_args():
    sys.exit(1)

  # Returns a file path, "/tmp/newfile.wav"
  audio_file1 = normalize_wave_file(sys.argv[1])
  audio_file2 = normalize_wave_file(sys.argv[2])

  ffts1 = get_fft(audio_file1)
  ffts2 = get_fft(audio_file2)

  match = compare(ffts1, ffts2)

  if match:
    print "MATCH"
  else:
    print "NO MATCH"

if __name__ == '__main__':
  main()
