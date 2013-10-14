#!/usr/bin/python

import math
import numpy
import os
import os.path
import random
import struct
import unittest
import wave

import p4500


class TestEuclideanDistance(unittest.TestCase):
  def setUp(self):
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3, 4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1, 1, 1, 1, 1], dtype=int)
    self.complex_ones = numpy.array([1, 1, 1, 1, 1], dtype=numpy.complex128)

  def test_zeros(self):
    self.assertEqual(p4500.euclidean_distance(self.two_zeros, self.two_zeros), 0)
    self.assertEqual(p4500.euclidean_distance(self.five_zeros, self.five_zeros), 0)

  def test_distance_ones_to_ones(self):
    self.assertEqual(p4500.euclidean_distance(self.int_ones, self.int_ones), 0)
    self.assertEqual(p4500.euclidean_distance(self.complex_ones,self.complex_ones), 0)
    self.assertEqual(p4500.euclidean_distance(self.int_ones, self.complex_ones), 0)
    self.assertEqual(p4500.euclidean_distance(self.complex_ones, self.int_ones), 0)

  def test_hypotenuse_is_five(self):
    self.assertEqual(p4500.euclidean_distance(self.two_zeros, self.three_four), 5)
    self.assertEqual(p4500.euclidean_distance(self.three_four, self.two_zeros), 5)
    self.assertEqual(p4500.euclidean_distance(self.two_zeros, self.complex_three_four), 5)
    self.assertEqual(p4500.euclidean_distance(self.complex_three_four, self.two_zeros), 5)

  def test_hypotenuse_is_sqrt_five(self):
    self.assertEqual(p4500.euclidean_distance(self.five_zeros, self.int_ones), math.sqrt(5))


class TestFileNormalization(unittest.TestCase):
  def setUp(self):
    values = []
    for i in range(0, 100000):
      value = random.randint(-32767, 32767)
      packed_value = struct.pack('h', value)
      values.append(packed_value)
      values.append(packed_value)

    value_str = ''.join(values)
    self.noise_file = 'noise.wav'
    self.slow_noise_file = 'slow_noise.wav'

    noise_output = wave.open(self.noise_file, 'w')
    noise_output.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
    noise_output.writeframes(value_str)
    noise_output.close()

    slow_noise_output = wave.open(self.slow_noise_file, 'w')
    slow_noise_output.setparams((2, 2, 30000, 0, 'NONE', 'NONE'))
    slow_noise_output.writeframes(value_str)
    noise_output.close()

    self.noise_file_norm = p4500.normalize_wave_file(self.noise_file)
    self.slow_noise_file_norm = p4500.normalize_wave_file(self.slow_noise_file)

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

  def tearDown(self):
    os.remove(self.noise_file)
    os.remove(self.slow_noise_file)

  def output_file_exists(self):
    self.assertTrue(os.path.isfile(self.noise_file_norm))
    self.assertTrue(os.path.isfile(self.slow_noise_file_norm))

  def test_output_file_is_in_tmp(self):
    self.assertTrue(self.noise_file_norm.startswith('/tmp'))
    self.assertTrue(self.slow_noise_file_norm.startswith('/tmp'))

  def output_file_is_in_mono(self):
    self.assertEqual(self.noise_norm_channels, p4500.NUM_CHANNELS)
    self.assertEqual(self.slow_noise_norm_channels, p4500.NUM_CHANNELS)

  def output_file_hz_is_44100(self):
    self.assertEqual(self.noise_norm_hz, p4500.FREQUENCY)
    self.assertEqual(self.slow_noise_norm_hz, p4500.FREQUENCY)


class TestMatching(unittest.TestCase):
  def setUp(self):
    self.fft1 = numpy.array([[1, 5, 8], [2, 4, 0], [2, 2, 1]])
    self.fft2 = numpy.array([[2, 4, 0], [2, 2, 1]])
    self.fft3 = numpy.array([[0.9, 5.1, 7.9], [2, 4, 0.1], [1.9, 2.2, 1]])
    self.fft4 = numpy.array([[7, 2, 4], [5, 5, 5], [1, 2, 5]])

  def test_exact_match(self):
    self.assertTrue(p4500.compare(self.fft1, self.fft1))

  def test_exact_partial_match(self):
    self.assertTrue(p4500.compare(self.fft1, self.fft2))

  def test_fuzzy_match(self):
    self.assertTrue(p4500.compare(self.fft1, self.fft3))
    self.assertTrue(p4500.compare(self.fft3, self.fft1))

  def test_not_a_match(self):
    self.assertFalse(p4500.compare(self.fft1, self.fft4))


if __name__ == '__main__':
  unittest.main()
