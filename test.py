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

class TestFileDetection(unittest.TestCase):
  """Test that WAVE, MP3, and other files are properly detected"""
  
  def setUp(self):
    """Generate files to call methods on"""
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

  def test_is_wave(self):
    """Test WAVE detection"""
    self.assertTrue(utils.is_wave(self.wav_file))
    self.assertTrue(utils.is_wave(self.hidden_wav_file))
    self.assertFalse(utils.is_wave(self.mp3_file))
    self.assertFalse(utils.is_wave(self.hidden_mp3_file))
    self.assertFalse(utils.is_wave(self.text_file))
    self.assertFalse(utils.is_wave(self.fake_wav_file))
    self.assertFalse(utils.is_wave(self.fake_mp3_file))

  def test_is_mp3(self):
    """Test MP3 detection"""
    self.assertTrue(utils.is_mp3(self.mp3_file))
    self.assertTrue(utils.is_mp3(self.hidden_mp3_file))
    self.assertFalse(utils.is_mp3(self.wav_file))
    self.assertFalse(utils.is_mp3(self.hidden_wav_file))
    self.assertFalse(utils.is_mp3(self.text_file))
    self.assertFalse(utils.is_mp3(self.fake_wav_file))
    self.assertFalse(utils.is_mp3(self.fake_mp3_file))

  def tearDown(self):
    """Remove generated files after tests are done"""
    os.remove(self.wav_file)
    os.remove(self.hidden_wav_file)
    os.remove(self.mp3_file)
    os.remove(self.hidden_mp3_file)
    os.remove(self.text_file)
    os.remove(self.fake_wav_file)
    os.remove(self.fake_mp3_file)


class TestFileNormalization(unittest.TestCase):
  """Test that WAV files are being normalized properly
  and that MP3 files are being successfully converted to WAV format"""

  def setUp(self):
    """Generate files to call methods on"""
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

  def test_output_file_exists(self):
    """Test to make sure file generation was successful"""
    self.assertTrue(os.path.isfile(self.noise_file_norm))
    self.assertTrue(os.path.isfile(self.slow_noise_file_norm))

  def test_output_file_is_in_tmp(self):
    """Test to make sure generated files have been placed in /tmp"""
    self.assertTrue(self.noise_file_norm.startswith('/tmp'))
    self.assertTrue(self.slow_noise_file_norm.startswith('/tmp'))

  def test_output_file_is_in_mono(self):
    """Test to make sure generated files only have one channel"""
    self.assertEqual(self.noise_norm_channels, NUM_CHANNELS)
    self.assertEqual(self.slow_noise_norm_channels, NUM_CHANNELS)

  def test_output_file_hz_is_44100(self):
    """Test to make sure generated files 
    have a sampling frequency of 44100 Hz"""
    self.assertEqual(self.noise_norm_hz, FREQUENCY)
    self.assertEqual(self.slow_noise_norm_hz, FREQUENCY)

  def tearDown(self):
    """Remove generated files after tests are done"""
    os.remove(self.noise_file)
    os.remove(self.slow_noise_file)
    os.remove(self.noise_file_norm)
    os.remove(self.slow_noise_file_norm)


class TestGetFFT(unittest.TestCase):
  """Test conversion of sound data to FFTs"""

  def setUp(self):
    """Generate sound files and calculate their FFTs"""
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
    self.one_s_z_fft = normalize.get_ffts(self.short_zeros)
    self.two_s_z_fft = normalize.get_ffts(self.long_zeros)

  def test_output_dimensions(self):
    """FFT output should have be of dimensions 
    [seconds / chunk_size, 44100 * chunk_size]"""
    self.assertEqual(self.one_s_z_fft.shape, 
                     (1 / COMP_CHUNK_SIZE, 44100 * COMP_CHUNK_SIZE))
    self.assertEqual(self.two_s_z_fft.shape,
                     (2 / COMP_CHUNK_SIZE, 44100 * COMP_CHUNK_SIZE))

  def test_zero_output(self):
    """FFTs of no sound should return arrays of zero"""
    self.assertFalse(numpy.any(self.one_s_z_fft))
    self.assertFalse(numpy.any(self.two_s_z_fft))

  def tearDown(self):
    """Remove generated sound files after tests are done"""
    os.remove(self.short_zeros)
    os.remove(self.long_zeros)


class TestEuclideanDistance(unittest.TestCase):
  """Test the euclidean distance function"""

  def setUp(self):
    """Create vectors to calculate the distances between"""
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3, 4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1, 1, 1, 1, 1], dtype=int)
    self.complex_ones = numpy.array([1, 1, 1, 1, 1], dtype=numpy.complex128)

  def test_zeros(self):
    """The distance between the origin and itself should be 0"""
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.two_zeros),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.five_zeros),
                     0)

  def test_distance_ones_to_ones(self):
    """The distance between any two vectors at 
    the same point in space should be 0"""
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

  def test_hypotenuse_is_five(self):
    """The distance between points (0, 0) and (3, 4) should be 5"""
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

  def test_hypotenuse_is_sqrt_five(self):
    """The distance between (0, 0, 0, 0, 0) 
    and (1, 1, 1, 1, 1) should be sqrt(5)"""
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.int_ones),
                     math.sqrt(5))


class TestMeanSquaredError(unittest.TestCase):
  """Test mean squared error distance calculation"""

  def setUp(self):
    """Create vectors to calculate the distances between"""
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3, 4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1, 1, 1, 1, 1], dtype=int)
    self.complex_ones = numpy.array([1, 1, 1, 1, 1], dtype=numpy.complex128)

  def test_zeros(self):
    """The distance between the origin and itself should be 0"""
    self.assertEqual(comparator.euclidean_distance(self.two_zeros,
                                                   self.two_zeros),
                     0)
    self.assertEqual(comparator.euclidean_distance(self.five_zeros,
                                                   self.five_zeros),
                     0)

  def test_distance_ones_to_ones(self):
    """The distance between any two vectors at 
    the same point in space should be 0"""
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

  def test_error_should_be_one(self):
    """The mean squared error between (0, 0, 0, 0, 0) 
    and (1, 1, 1, 1, 1) should be 1"""
    self.assertEqual(comparator.distance(self.five_zeros, self.int_ones), 1)
    self.assertEqual(comparator.distance(self.int_ones, self.five_zeros), 1)
    self.assertEqual(comparator.distance(self.five_zeros, self.complex_ones), 1)
    self.assertEqual(comparator.distance(self.complex_ones, self.five_zeros), 1)

  def test_error_should_be_25_over_2(self):
    """The mean squared error between (0, 0) and (3, 4) should be 12.5"""
    self.assertEqual(comparator.distance(self.two_zeros, self.three_four), 
                     12.5)
    self.assertEqual(comparator.distance(self.three_four, self.two_zeros), 
                     12.5)


class TestMatching(unittest.TestCase):
  """Test matches between arrays"""

  def setUp(self):
    """Generate arrays to match"""
    self.fft1 = numpy.array([[1, 5, 8], [2, 4, 0], [2, 2, 1]])
    self.fft2 = numpy.array([[2, 4, 0], [2, 2, 1]])
    self.fft3 = numpy.array([[0.9, 5.1, 7.9], [2, 4, 0.1], [1.9, 2.2, 1]])
    self.fft4 = numpy.array([[7, 2, 4], [5, 25, 125], [1, 2, 5]])

  def test_exact_match(self):
    """An array compared against itself should match"""
    self.assertTrue(comparator.compare(self.fft1, self.fft1))

  def test_exact_partial_match(self):
    """An array compared against a different array that contains 
    all of its elements in order should match"""
    self.assertTrue(comparator.compare(self.fft1, self.fft2))

  def test_fuzzy_match(self):
    """Arrays containing different elements, where one is not a slice of the 
    other, should match if their elements are numerically close together"""
    self.assertTrue(comparator.compare(self.fft1, self.fft3))
    self.assertTrue(comparator.compare(self.fft3, self.fft1))

  def test_not_a_match(self):
    """Arrays containing different elements, where one is not a slice of the
    other, should not match if their elements are numerically far apart"""
    self.assertFalse(comparator.compare(self.fft1, self.fft4))


class TestQuote(unittest.TestCase):
  """Test string quoting"""

  def test_quote_empty(self):
    """Quoting the empty string should return a pair of quotes"""
    self.assertEqual(utils.quote(''), '\'\'')

  def test_quote_no_space(self):
    """Quoting should work on strings without spaces"""
    self.assertEqual(utils.quote('quote'), '\'quote\'')
    self.assertEqual(utils.quote('12345$'), '\'12345$\'')

  def test_quote_with_space(self):
    """Quoting should work on strings containing spaces"""
    self.assertEqual(utils.quote('quote me'), '\'quote me\'')
    self.assertEqual(utils.quote('123 45$'), '\'123 45$\'')

# If this script is run from the command line, run the above tests
if __name__ == '__main__':
  unittest.main()
