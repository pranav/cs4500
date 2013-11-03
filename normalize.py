import audioop
import os
import sys
import wave

import numpy
import numpy.fft

import utils
from config import *


# TODO: Normalize tempo to 100 bpm
def normalize_wave_file(audio_file):
  """Normalizes a WAVE file to single channel WAVE.

  Args:
    audio_file: A WAVE file name.

  Returns:
    The path to the normalized WAVE file.
  """
  global FREQUENCY, NUM_CHANNELS, SAMPLE_WIDTH
  # Get path in /tmp to write to
  output_path = utils.get_tmp_path(audio_file) + '_norm'
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
  wf.setparams((NUM_CHANNELS, SAMPLE_WIDTH, FREQUENCY, nframes, 'NONE',
                'NONE'))
  wf.writeframes(frames)
  wf.close()
  try:
    os.chmod(output_path, 0666)
  except OSError:
    pass
  return output_path


def mp3_to_wav(mp3_file):
  """Converts an MP3 file to a WAVE file.

  Args:
    mp3_file: An MP3 file name.

  Returns:
    The path to the generated WAVE file.
  """
  # Get path in /tmp/ to write to
  output_path = utils.get_tmp_path(mp3_file) + '_wav'
  # Run lame to decode the MP3 file to WAVE
  os.system('/course/cs4500f13/bin/lame --decode --mp3input --quiet {0} {1}'
            .format(utils.quote(mp3_file), utils.quote(output_path)))
  try:
    os.chmod(output_path, 0666)
  except OSError:
    pass
  return output_path

def get_ffts(audio_file):
  """Computes the FFT of each chunk of a WAVE file.

  Splits the WAVE file into chunks of equal temporal length and performs
  an FFT on each.

  Args:
    audio_file: A WAVE file.

  Returns:
    An iterable of the FFTs of the chunks of the WAVE file.
  """
  global COMP_CHUNK_SIZE
  # Read the file, and determine its length in 'chunks'
  (sample, data) = utils.read_wave_from_file(audio_file)
  total_chunks = (data.size / sample) / COMP_CHUNK_SIZE
  # Allocate space for the FFT decompsitions of each chunk of sound data
  fft_out = numpy.ndarray(shape=(total_chunks, sample*COMP_CHUNK_SIZE),
                          dtype=numpy.complex128)
  # Loop invariant:
  #   0 <= chunk <= total_chunks
  #   results in an array (fft_out) of FFTs that correspond to the chunks of
  #    the audio file
  chunk = 0
  while chunk < total_chunks:
    fft = numpy.fft.fft(data[chunk * COMP_CHUNK_SIZE * sample
                             :(chunk + 1) * COMP_CHUNK_SIZE * sample])
    fft_out[chunk] = fft
    chunk += 1
  return fft_out
