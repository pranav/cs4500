import audioop
import os
import re
import sys
import wave

import numpy
import numpy.fft

import utils
from config import *


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
  try:
    os.chmod(output_path, 0666)
  except OSError:
    pass

  return output_path


# numpy.fft should break the file into chunks and perform an fft
# return the array/fft
def get_fft(audio_file):
  global COMP_CHUNK_SIZE

  # Read the file, and determine its length in 'chunks'
  (sample, data) = utils.read_wave_from_file(audio_file)
  total_chunks = (data.size / sample) / COMP_CHUNK_SIZE

  # Allocate space for the FFT decompsitions of each chunk of sound data
  fft_out = numpy.ndarray(shape=(total_chunks, sample*COMP_CHUNK_SIZE),
                          dtype=numpy.complex128)

  # Loop through all chunks, computing their FFT decompositions

  # Loop invariant:
  #   0 <= chunk <= total_chunks
  #   results in an array (fft_out) of FFTs that correspond to the chunks of the
  #    audio file

  chunk = 0
  while chunk < total_chunks:
    fft = numpy.fft.fft(data[chunk*COMP_CHUNK_SIZE*sample : (chunk+1)*COMP_CHUNK_SIZE*sample])
    fft_out[chunk] = fft
    chunk += 1

  return fft_out
