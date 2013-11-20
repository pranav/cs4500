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
def normalize_wave_file(path):
  """Normalizes a WAVE file to single channel WAVE.

  Args:
    path: The path to a WAVE file.

  Returns:
    The normalized WAVE file.
  """
  global FREQUENCY, NUM_CHANNELS, SAMPLE_WIDTH
  # Read WAVE data
  wf = wave.open(path, 'rb')
  (nchannels, sampwidth, framerate, nframes, comptype, compname) = \
    wf.getparams()
  frames = wf.readframes(nframes)
  wf.close()
  # Create a temporary copy of it with new parameters
  tmp_file = utils.get_tmp_file(path)
  wf = wave.open(tmp_file.name, 'wb')
  wf.setparams((NUM_CHANNELS, SAMPLE_WIDTH, FREQUENCY, nframes, 'NONE',
                'NONE'))
  if nchannels != 1:
    frames = audioop.tomono(frames, sampwidth, 1, 1)
  wf.writeframes(frames)
  wf.close()
  return tmp_file


def mp3_to_wav(path):
  """Converts an MP3 file to a WAVE file.

  Args:
    path: The path to an MP3 file.

  Returns:
    The generated WAVE file.
  """
  tmp_file = utils.get_tmp_file(path)
  os.system('/course/cs4500f13/bin/lame --decode --mp3input --quiet {0} {1}'
            .format(utils.quote(path), utils.quote(tmp_file.name)))
  return tmp_file


def triangular_filters(sample, nfft):
  """Compute the mel triangular filters for the frame.

  Args:
    sample: the sample rate of the frame
    nfft: the size of the FFT

  Returns:
    The mel filterbank.
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


def get_mfcc(path):
  """Finds the MFCCs and FFTs of a WAVE file.

  Args:
    path: The path to a WAVE file.

  Returns:
    A tuple of two iterables, the FFTs and MFCCs of the frames of the
    WAVE file.
  """
  global COMP_FRAME_SIZE
  # Read the file, and determine its length in frames
  (sample, data) = utils.read_wave_from_file(path)
  total_frames = (data.size / sample) / COMP_FRAME_SIZE

  step = COMP_FRAME_SIZE * sample
  prefactor = 0.97
  window = hamming(step)

  # Allocate space for the FFT decompositions of each frame of sound data
  fft_out = []
  mfcc_out = []

  # Loop invariant:
  #   0 <= frame_index <= total_frames
  #   results in an array (fft_out) of FFTs that correspond to the
  #    frames of the WAVE file
  filterbank_cache = {}
  frame_index = 0
  while frame_index < total_frames:
    # Obtain the frame_indexth frame from the data
    frame = data[frame_index * step:(frame_index + 1) * step]
    frame = pre_emphasis(frame, prefactor)

    # Generate the FFT of the frame windowed by the hamming window
    frame_fft = numpy.fft.fft(frame * window)

    nfft = len(frame_fft)

    # Compute the mel triangular filterbank or get a cached version
    fb_key = (sample, nfft)
    if fb_key in filterbank_cache:
        filterbank = filterbank_cache[fb_key]
    else:
        filterbank = triangular_filters(sample, nfft)
        filterbank_cache[fb_key] = filterbank

    # The power spectrum of the frame
    power_spectrum = numpy.abs(frame_fft)
    # Filtered by the mel filterbank
    mel_power_spectrum = numpy.log10(numpy.dot(power_spectrum, filterbank.T))
    # With the Discrete Cosine Transform to find the cepstrum
    cepstrum = dct(mel_power_spectrum, type=2, norm='ortho', axis=-1)
    fft_out.append(frame_fft)
    mfcc_out.append(cepstrum)
    frame_index = frame_index + 1
  return (fft_out, mfcc_out)


def pre_emphasis(frame, factor):
    return lfilter([1., -factor], 1, frame)

def get_ffts(audio_file):
  """Computes the FFT of each frame of a WAVE file.

  Splits the WAVE file into frames of equal temporal length and performs
  an FFT on each.

  Args:
    audio_file: A WAVE file.

  Returns:
    An iterable of the FFTs of the frames of the WAVE file.
  """
  global COMP_FRAME_SIZE
  # Read the file, and determine its length in frames
  (sample, data) = utils.read_wave_from_file(audio_file)
  total_frames = (data.size / sample) / COMP_FRAME_SIZE
  # Allocate space for the FFT decompsitions of each frame of sound data
  fft_out = numpy.ndarray(shape=(total_frames, sample*COMP_FRAME_SIZE),
                          dtype=numpy.complex128)
  # Loop invariant:
  # 0 <= frame_index <= total_frames
  # results in an array (fft_out) of FFTs that correspond to the frames of
  #  the audio file
  frame_index = 0
  while frame_index < total_frames:
    fft = numpy.fft.fft(data[frame_index * COMP_FRAME_SIZE * sample
                             :(frame_index + 1) * COMP_FRAME_SIZE * sample])
    fft_out[frame_index] = fft
    frame_index = frame_index + 1
  return fft_out
