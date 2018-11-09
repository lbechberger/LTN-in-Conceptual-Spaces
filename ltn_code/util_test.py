# -*- coding: utf-8 -*-
"""
Some unit tests for the functions in util.py

Created on Mon Nov  5 12:29:56 2018

@author: lbechberger
"""

import unittest
import util

class TestUtil(unittest.TestCase):

    def setUp(self):
        self.vectors = [(['A'], None), (['B'], None), (['C', 'A'], None)]
        self.binary_predictions = {'A' : [1.0, 0.0, 0.0], 'B' : [0.0, 0.0, 1.0], 'C' : [0.0, 1.0, 1.0], 'D' : [0.0, 0.0, 0.0]}
        self.continuous_predictions = {'A' : [1.0, 0.1, 0.2], 'B' : [0.5, 0.2, 0.9], 'C' : [0.1, 1.0, 1.0], 'D' : [0.0, 0.0, 0.0]}
        self.all_labels = ['A', 'B', 'C', 'D']

        
    # one_error()
    def test_one_error_binary(self):
        one_error = util.one_error(self.binary_predictions, self.vectors)
        self.assertAlmostEqual(one_error, 0.5)

    def test_one_error_continuous(self):
        one_error = util.one_error(self.continuous_predictions, self.vectors)
        self.assertAlmostEqual(one_error, 1/3)


    # coverage()
    def test_coverage_binary(self):
        coverage = util.coverage(self.binary_predictions, self.vectors)
        self.assertAlmostEqual(coverage, 3)

    def test_coverage_continuous(self):
        coverage = util.coverage(self.continuous_predictions, self.vectors)
        self.assertAlmostEqual(coverage, 2)


    # ranking_loss()
    def test_ranking_loss_binary(self):
        ranking_loss = util.ranking_loss(self.binary_predictions, self.vectors, self.all_labels)
        self.assertAlmostEqual(ranking_loss, 7/36)

    def test_ranking_loss_continuous(self):
        ranking_loss = util.ranking_loss(self.continuous_predictions, self.vectors, self.all_labels)
        self.assertAlmostEqual(ranking_loss, 7/36)


    # average_precision()
    def test_average_precision_binary(self):
        average_precision = util.average_precision(self.binary_predictions, self.vectors)
        self.assertAlmostEqual(average_precision, 7/12)

    def test_average_precision_continuous(self):
        average_precision = util.average_precision(self.continuous_predictions, self.vectors)
        self.assertAlmostEqual(average_precision, 7/9)


    # exact_match_prefix()
    def test_exact_match_prefix_binary(self):
        exact_match_prefix = util.exact_match_prefix(self.binary_predictions, self.vectors)
        self.assertAlmostEqual(exact_match_prefix, 1/3)

    def test_exact_match_prefix_binary_additional(self):
        predictions = {'A' : [1.0], 'B': [1.0], 'C': [0.0]}
        vectors = [('A', None)]
        exact_match_prefix = util.exact_match_prefix(predictions, vectors)
        self.assertAlmostEqual(exact_match_prefix, 0.0)

    def test_exact_match_prefix_continuous(self):
        exact_match_prefix = util.exact_match_prefix(self.continuous_predictions, self.vectors)
        self.assertAlmostEqual(exact_match_prefix, 1/3)


    # cross_entropy_loss()
    def test_cross_entropy_loss_binary(self):
        cross_entropy_loss = util.cross_entropy_loss(self.binary_predictions, self.vectors, self.all_labels)
        self.assertAlmostEqual(cross_entropy_loss, 4000/3)

    def test_cross_entropy_loss_continuous(self):
        cross_entropy_loss = util.cross_entropy_loss(self.continuous_predictions, self.vectors, self.all_labels)
        self.assertAlmostEqual(cross_entropy_loss, 336.4232634905174)



    # label_wise_precision()
    def test_label_wise_precision_binary(self):
        expectation = {'contents' : ['min', 'mean', 'A', 'B', 'C', 'D'], 'min': 0, 'mean' : 1/6, 'A': 0.5, 'B': 0, 'C': 0, 'D': None}
        label_wise_precision = util.label_wise_precision(self.binary_predictions, self.vectors, self.all_labels)
        self.assertEqual(expectation, label_wise_precision)

    def test_label_wise_precision_continuous(self):
        expectation = {'contents' : ['min', 'mean', 'A', 'B', 'C', 'D'], 'min': 0, 'mean' : 0.5, 'A': 0.5, 'B': 0, 'C': 1, 'D': None}
        label_wise_precision = util.label_wise_precision(self.continuous_predictions, self.vectors, self.all_labels)
        self.assertEqual(expectation, label_wise_precision)

unittest.main()