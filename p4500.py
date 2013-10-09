#!/usr/bin/python

import sys
import wave
import numpy
import scipy.io.wavfile

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

main()
