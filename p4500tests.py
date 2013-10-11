#!/usr/bin/python

import unittest
import p4500
import numpy
import math

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
    self.assertEqual(p4500.euclidean_distance(self.complex_ones, self.complex_ones), 0)
    self.assertEqual(p4500.euclidean_distance(self.int_ones, self.complex_ones), 0)
    self.assertEqual(p4500.euclidean_distance(self.complex_ones, self.int_ones), 0)

  def test_hypotenuse_is_five(self):
    self.assertEqual(p4500.euclidean_distance(self.two_zeros, self.three_four), 5)
    self.assertEqual(p4500.euclidean_distance(self.three_four, self.two_zeros), 5)
    self.assertEqual(p4500.euclidean_distance(self.two_zeros, self.complex_three_four), 5)
    self.assertEqual(p4500.euclidean_distance(self.complex_three_four, self.two_zeros), 5)

  def test_hypotenuse_is_sqrt_five(self):
    self.assertEqual(p4500.euclidean_distance(self.five_zeros, self.int_ones), math.sqrt(5))

if __name__ == '__main__':
  unittest.main()
