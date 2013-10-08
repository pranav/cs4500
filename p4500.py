#!/usr/bin/python

import sys
import wave
import numpy


def check_args():
  try:
    return len(sys.argv) == 3 and wave.open(sys.argv[1]) and wave.open(sys.argv[2])
  except EOFError:
    sys.stderr.write('ERROR Both files must be of WAV format\n')
    return True
  except IOError:
    return False


# Normalize to single channel WAV
# Normalize temp to 100 bpm
# Returns a file path string, example: "/tmp/newfile.wav"
def normalize_wave_file(wavfile):
  pass


# numpy.fft should break the file into chunks and perform an fft
# return the array/fft
def get_fft(wavfile):
  pass

# Compare the 2 FFTs using crazy linear distance thingys
#TODO: Better comment
def match_comparator(wav1, wav2):
  pass


def main():
  # Check arguments and check if WAV file
  if not check_args():
    sys.stderr.write('ERROR Usage: ./p4500 <path> <path>\n')
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
