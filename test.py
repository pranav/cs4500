#!/usr/bin/python

import math
import os
import os.path
import random
import struct
import unittest
import wave

import numpy

import comparator
from config import *
import normalize
import utils

# Test that WAVE, MP3, and other files are properly detected
class TestFileDetection(unittest.TestCase):
  # Generate files to call methods on
  def setUp(self):
    # Generate noise to store in sound files
    values = []
    for i in range(0, 100000):
      value = random.randint(-32767, 32767)
      packed_value = struct.pack('h', value)
      values.append(packed_value)
      values.append(packed_value)
    value_str = ''.join(values)

    # Write noise to WAVE file
    self.wav_file = '/tmp/noise.wav'
    wav_output = wave.open(self.wav_file, 'w')
    wav_output.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
    wav_output.writeframes(value_str)
    wav_output.close()

    # Write noise to WAVE file with different extension
    self.hidden_wav_file = '/tmp/hidden_wav_noise.txt'
    hidden_wav_output = wave.open(self.hidden_wav_file, 'w')
    hidden_wav_output.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
    hidden_wav_output.writeframes(value_str)
    hidden_wav_output.close()

    # Write noise to MP3 file (use lame to convert WAVE to MP3)
    self.mp3_file = '/tmp/noise.mp3'
    os.system('/course/cs4500f13/bin/lame --quiet {0} {1}'
              .format(self.wav_file, self.mp3_file))

    # Write noise to MP3 file with different extension (use lame)
    self.hidden_mp3_file = '/tmp/hidden_mp3_noise.txt'
    os.system('/course/cs4500f13/bin/lame --quiet {0} {1}'
              .format(self.wav_file, self.hidden_mp3_file))

    # Write noise to text file
    self.text_file = '/tmp/noise.txt'
    text_output = open(self.text_file, 'w')
    text_output.write(value_str)
    text_output.close()

    # Write noise to fake WAVE file
    self.fake_wav_file = '/tmp/fake_noise.wav'
    fake_wav_output = open(self.fake_wav_file, 'w')
    fake_wav_output.write(value_str)
    fake_wav_output.close()

    # Write noise to fake MP3 file
    self.fake_mp3_file = '/tmp/fake_noise.mp3'
    fake_mp3_output = open(self.fake_mp3_file, 'w')
    fake_mp3_output.write(value_str)
    fake_mp3_output.close()

  # Test WAVE detection
  def test_is_wave(self):
    self.assertTrue(utils.is_wave(self.wav_file))
    self.assertTrue(utils.is_wave(self.hidden_wav_file))
    self.assertFalse(utils.is_wave(self.mp3_file))
    self.assertFalse(utils.is_wave(self.hidden_mp3_file))
    self.assertFalse(utils.is_wave(self.text_file))
    self.assertFalse(utils.is_wave(self.fake_wav_file))
    self.assertFalse(utils.is_wave(self.fake_mp3_file))

  # Test MP3 detection
  def test_is_mp3(self):
    self.assertTrue(utils.is_mp3(self.mp3_file))
    self.assertTrue(utils.is_mp3(self.hidden_mp3_file))
    self.assertFalse(utils.is_mp3(self.wav_file))
    self.assertFalse(utils.is_mp3(self.hidden_wav_file))
    self.assertFalse(utils.is_mp3(self.text_file))
    self.assertFalse(utils.is_mp3(self.fake_wav_file))
    self.assertFalse(utils.is_mp3(self.fake_mp3_file))

  # Remove generated files after tests are done
  def tearDown(self):
    os.remove(self.wav_file)
    os.remove(self.hidden_wav_file)
    os.remove(self.mp3_file)
    os.remove(self.hidden_mp3_file)
    os.remove(self.text_file)
    os.remove(self.fake_wav_file)
    os.remove(self.fake_mp3_file)


# Test that WAV files are being normalized properly, and that MP3 files are
# being successfully converted to WAV format
class TestFileNormalization(unittest.TestCase):
  # Generate files to call methods on
  def setUp(self):
    # Generate noise to store in sound files
    values = []
    for i in range(0, 100000):
      value = random.randint(-32767, 32767)
      packed_value = struct.pack('h', value)
      values.append(packed_value)
      values.append(packed_value)
    value_str = ''.join(values)

    # Write noise to WAVE file
    self.noise_file = '/tmp/noise.wav'
    noise_output = wave.open(self.noise_file, 'w')
    noise_output.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
    noise_output.writeframes(value_str)
    noise_output.close()

    # Write noise to WAVE file with slower framerate
    self.slow_noise_file = '/tmp/slow_noise.wav'
    slow_noise_output = wave.open(self.slow_noise_file, 'w')
    slow_noise_output.setparams((2, 2, 30000, 0, 'NONE', 'NONE'))
    slow_noise_output.writeframes(value_str)
    noise_output.close()

    # Normalize both normal and slow WAVE files
    self.noise_file_norm = normalize.normalize_wave_file(self.noise_file)
    self.slow_noise_file_norm = normalize.normalize_wave_file(
      self.slow_noise_file)

    # Get params of normalized files
    nfn = wave.open(self.noise_file_norm)
    nfnparam = nfn.getparams()
    self.noise_norm_channels = nfnparam[0]
    self.noise_norm_hz = nfnparam[2]
    nfn.close()
    snfn = wave.open(self.slow_noise_file_norm)
    snfnparam = snfn.getparams()
    self.slow_noise_norm_channels = snfnparam[0]
    self.slow_noise_norm_hz = snfnparam[2]
    snfn.close()

  # Test to make sure file generation was successful
  def test_output_file_exists(self):
    self.assertTrue(os.path.isfile(self.noise_file_norm))
    self.assertTrue(os.path.isfile(self.slow_noise_file_norm))

  # Test to make sure generated files have been placed in /tmp
  def test_output_file_is_in_tmp(self):
    self.assertTrue(self.noise_file_norm.startswith('/tmp'))
    self.assertTrue(self.slow_noise_file_norm.startswith('/tmp'))

  # Test to make sure generated files only have one channel
  def test_output_file_is_in_mono(self):
    self.assertEqual(self.noise_norm_channels, NUM_CHANNELS)
    self.assertEqual(self.slow_noise_norm_channels, NUM_CHANNELS)

  # Test to make sure generated files have a sampling frequency of 44100 Hz
  def test_output_file_hz_is_44100(self):
    self.assertEqual(self.noise_norm_hz, FREQUENCY)
    self.assertEqual(self.slow_noise_norm_hz, FREQUENCY)

  # Remove generated files after tests are done
  def tearDown(self):
    os.remove(self.noise_file)
    os.remove(self.slow_noise_file)
    os.remove(self.noise_file_norm)
    os.remove(self.slow_noise_file_norm)


# Test conversion of sound data to FFTs
class TestGetFFT(unittest.TestCase):
  # Generate sound files and calculate their FFTs
  def setUp(self):
    # Construct strings of values to store in files
    one_s_zeros = ''.join([struct.pack('h', 0) for i in range(0, 44100)])
    two_s_zeros = ''.join([struct.pack('h', 0) for i in range(0, 88200)])

    # Generated file names
    self.short_zeros = '/tmp/short_zeros.wav'
    self.long_zeros = '/tmp/long_zeros.wav'

    # Write data to files
    wave_params = (1, 2, 44100, 1, 'NONE', 'NONE')
    one_s_z_out = wave.open(self.short_zeros, 'wb')
    one_s_z_out.setparams(wave_params)
    one_s_z_out.writeframes(one_s_zeros)
    one_s_z_out.close()
    two_s_z_out = wave.open(self.long_zeros, 'wb')
    two_s_z_out.setparams(wave_params)
    two_s_z_out.writeframes(two_s_zeros)
    two_s_z_out.close()

    # Get the FFTs of each file
    self.one_s_z_fft = normalize.get_fft(self.short_zeros)
    self.two_s_z_fft = normalize.get_fft(self.long_zeros)

  # FFT output should have be of dimensions
  # [seconds / chunk_size, 44100 * chuck_size]
  def test_output_dimensions(self):
    self.assertEqual(self.one_s_z_fft.shape, 
                     (1 / COMP_CHUNK_SIZE, 44100 * COMP_CHUNK_SIZE))
    self.assertEqual(self.two_s_z_fft.shape,
                     (2 / COMP_CHUNK_SIZE, 44100 * COMP_CHUNK_SIZE))

  # FFTs of no sound should return arrays of zero
  def test_zero_output(self):
    self.assertFalse(numpy.any(self.one_s_z_fft))
    self.assertFalse(numpy.any(self.two_s_z_fft))

  # Remove generated sound files after tests are done
  def tearDown(self):
    os.remove(self.short_zeros)
    os.remove(self.long_zeros)

# Test the euclidean distance function
class TestEuclideanDistance(unittest.TestCase):
  # Create vectors to calculate the distances between
  def setUp(self):
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3, 4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1, 1, 1, 1, 1], dtype=int)
    self.complex_ones = numpy.array([1, 1, 1, 1, 1], dtype=numpy.complex128)

  # The distance between the origin should be 0
  def test_zeros(self):
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.two_zeros),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.five_zeros),
                     0)

  # The distance between any two vectors at the same point in space 
  # should be 0
  def test_distance_ones_to_ones(self):
    self.assertEqual(comparator.euclidean_distance(self.int_ones,
                                                   self.int_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.complex_ones,
                                                   self.complex_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.int_ones,
                                                   self.complex_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.complex_ones,
                                                   self.int_ones),
                     0)

  # The distance between points (0, 0) and (3, 4) should be 5
  def test_hypotenuse_is_five(self):
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.three_four),
                     5)
    self.assertEqual(comparator.euclidean_distance(self.three_four,
                                                   self.two_zeros),
                     5)
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.complex_three_four),
                     5)
    self.assertEqual(comparator.euclidean_distance(self.complex_three_four,
                                                   self.two_zeros),
                     5)

  # The distance between (0, 0, 0, 0, 0) and (1, 1, 1, 1, 1) should be sqrt(5)
  def test_hypotenuse_is_sqrt_five(self):
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.int_ones),
                     math.sqrt(5))


# Test mean squared error distance calculation
class TestMeanSquaredError(unittest.TestCase):
  # Create vectors to calculate the distances between
  def setUp(self):
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3, 4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1, 1, 1, 1, 1], dtype=int)
    self.complex_ones = numpy.array([1, 1, 1, 1, 1], dtype=numpy.complex128)

  # The distance between the origin should be 0
  def test_zeros(self):
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.two_zeros),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.five_zeros),
                     0)

  # The distance between any two vectors at the same point in space 
  # should be 0
  def test_distance_ones_to_ones(self):
    self.assertEqual(comparator.euclidean_distance(self.int_ones,
                                                   self.int_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.complex_ones,
                                                   self.complex_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.int_ones,
                                                   self.complex_ones),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.complex_ones,
                                                   self.int_ones),
                     0)

  # The mean squared error between (0, 0, 0, 0, 0) and (1, 1, 1, 1, 1) 
  # should be 1
  def test_error_should_be_one(self):
    self.assertEqual(comparator.distance(self.five_zeros, self.int_ones), 1)
    self.assertEqual(comparator.distance(self.int_ones, self.five_zeros), 1)
    self.assertEqual(comparator.distance(self.five_zeros, self.complex_ones), 1)
    self.assertEqual(comparator.distance(self.complex_ones, self.five_zeros), 1)

  # The mean squared error between (0, 0) and (3, 4) should be 12.5
  def test_error_should_be_25_over_2(self):
    self.assertEqual(comparator.distance(self.two_zeros, self.three_four), 
                     12.5)
    self.assertEqual(comparator.distance(self.three_four, self.two_zeros), 
                     12.5)

# Test matches between arrays
class TestMatching(unittest.TestCase):
  # Generate arrays to match
  def setUp(self):
    self.fft1 = numpy.array([[1, 5, 8], [2, 4, 0], [2, 2, 1]])
    self.fft2 = numpy.array([[2, 4, 0], [2, 2, 1]])
    self.fft3 = numpy.array([[0.9, 5.1, 7.9], [2, 4, 0.1], [1.9, 2.2, 1]])
    self.fft4 = numpy.array([[7, 2, 4], [5, 25, 125], [1, 2, 5]])

  # An array compared against itself should match
  def test_exact_match(self):
    self.assertTrue(comparator.compare(self.fft1, self.fft1))

  # An array compared against a different array that contains all of its
  # elements in order should match
  def test_exact_partial_match(self):
    self.assertTrue(comparator.compare(self.fft1, self.fft2))

  # Arrays containing different 
  def test_fuzzy_match(self):
    self.assertTrue(comparator.compare(self.fft1, self.fft3))
    self.assertTrue(comparator.compare(self.fft3, self.fft1))

  def test_not_a_match(self):
    self.assertFalse(comparator.compare(self.fft1, self.fft4))

# If this script is run from the command line, run the above tests
if __name__ == '__main__':
  unittest.main()
