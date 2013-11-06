import audioop
import os
import sys
import wave

import numpy
import numpy.fft

import scipy.signal
from scipy.signal import lfilter, hamming
from scipy.fftpack.realtransforms import dct

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

def triangular_filters(sample, nfft):
  """ Compute the Mel triangular filters for the frame.

  Args:
    sample: the sample rate of the frame
    nfft: the size of the fft

  Returns:
    The Mel filterbank
  """
  lowfreq = 133.3333
  linsc = 200/3.
  logsc = 1.0711703

  nlinfilt = 13
  nlogfilt = 27
  nfilt = nlinfilt + nlogfilt

  # Begin computing the coefficients
  freqs = numpy.zeros(nfilt+2)
  freqs[:nlinfilt] = lowfreq + numpy.arange(nlinfilt) * linsc
  freqs[nlinfilt:] = \
          freqs[nlinfilt-1] * logsc ** numpy.arange(1, nlogfilt + 3)
  heights = 2./(freqs[2:] - freqs[0:-2])

  # Generate the filterbank coefficients
  fbank = numpy.zeros((nfilt, nfft))
  nfreqs = numpy.arange(nfft) / (1. * nfft) * sample

  # Calculate the filters
  for i in range(nfilt):
      low = freqs[i]
      cen = freqs[i+1]
      hi = freqs[i+2]

      lid = numpy.arange(numpy.floor(low * nfft / sample) + 1,
              numpy.floor(cen * nfft / sample) + 1, dtype=numpy.int)
      lslope = heights[i] / (cen - low)
      rid = numpy.arange(numpy.floor(cen * nfft / sample) + 1,
              numpy.floor(hi * nfft / sample) + 1, dtype=numpy.int)
      rslope = heights[i] / (hi - cen)
      fbank[i][lid] = lslope * (nfreqs[lid] - low)
      fbank[i][rid] = rslope * (hi - nfreqs[rid])

  return fbank

def get_mfcc(audio_file):
  """Finds the MFCCs of the audio file.

  Args:
    audio_file: A WAVE file.

  Returns:
    An interable of the MFCCs of the chunks of the WAVE file.
  """
  global COMP_CHUNK_SIZE
  # Read the file, and determine its length in 'chunks'
  (sample, data) = utils.read_wave_from_file(audio_file)
  total_chunks = (data.size / sample) / COMP_CHUNK_SIZE

  STEP = COMP_CHUNK_SIZE * sample
  prefactor = 0.97
  window = hamming(STEP)

  # Allocate space for the FFT decompsitions of each chunk of sound data
  fft_out = list()
  mel_out = list()

  # Loop invariant:
  #   0 <= chunk <= total_chunks
  #   results in an array (fft_out) of FFTs that correspond to the chunks of
  #    the audio file
  filterbank_cache = {}
  chunk = 0
  while chunk < total_chunks:
    # Obtain the chunkth frame from the data
    frame = data[chunk * STEP:(chunk + 1) * STEP]
    frame = pre_emphasis( frame, prefactor )

    # Generate the FFT of the frame windowed by the hamming window
    frame_fft = numpy.fft.fft(frame * window)

    nfft = len(frame_fft)

    # Compute the Mel triangular filterbank or get a cached version
    fb_key = (sample, nfft)
    if fb_key in filterbank_cache:
        filterbank = filterbank_cache[fb_key]
    else:
        filterbank = triangular_filters(sample, nfft)
        filterbank_cache[fb_key] = filterbank

    # The power spectrum of the frame
    power_spectrum = numpy.abs(frame_fft)
    # Filtered by the Mel filterbank
    mel_power_spectrum = numpy.log10(numpy.dot(power_spectrum, filterbank.T))
    # With the Discrete Cosine Transform to find the cepstrum
    cepstrum = dct(mel_power_spectrum, type=2, norm='ortho', axis=-1)

    fft_out.append( frame_fft )
    mel_out.append( cepstrum )

    chunk = chunk + 1

  return (fft_out, mel_out)

def pre_emphasis(frame, factor):
    return lfilter([1., -factor], 1, frame)

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
    chunk = chunk + 1
  return fft_out
