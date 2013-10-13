#!/usr/bin/python

import math
import numpy
import unittest

import p4500


class TestEuclideanDistance(unittest.TestCase):
  def setUp(self):
    self.two_zeros = numpy.zeros(2, dtype=int)
    self.five_zeros = numpy.zeros(5)
    self.three_four = numpy.array([3,4], dtype=int)
    self.complex_three_four = numpy.array([-3, 4j], dtype=numpy.complex128)
    self.int_ones = numpy.array([1,1,1,1,1], dtype=int)
    self.complex_ones = numpy.array([1,1,1,1,1], dtype=numpy.complex128)

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
    pass

  def output_file_exists(self):
    pass

  def test_output_file_is_in_tmp(self):
    pass

  def output_file_is_in_mono(self):
    pass

  def output_file_hz_is_44100(self):
    pass


class TestMatching(unittest.TestCase):
  def setUp(self):
    self.fft1 = [numpy.array(arr) for arr in [[1, 5, 8], [2, 4, 0], [2, 2, 1]]]
    self.fft2 = [numpy.array(arr) for arr in [[2, 4, 0], [2, 2, 1]]]
    self.fft3 = [numpy.array(arr)
      for arr in [[0.9, 5.1, 7.9], [2, 4, 0.1], [1.9, 2.2, 1]]]
    self.fft4 = [numpy.array(arr) for arr in [[7, 2, 4], [5, 5, 5], [1, 2, 5]]]

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
