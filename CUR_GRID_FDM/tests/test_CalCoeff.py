import unittest
from CUR_GRID_FDM.DiscreteSchemes import CalCoeff
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from CUR_GRID_FDM.Geometry import DonutMesh
from CUR_GRID_FDM.Geometry import RectangularMesh

import numpy as np
import math


class CalCoeffTestCase(unittest.TestCase):
    def setUp(self):
        self.mesh = DonutMesh(2.0, 5.0, 4, 12, 2.0, 3)
        self.coeff = CalCoeff(self.mesh)

    # def tearDown(self):

    def test_covariantBasis(self):
        self.assertAlmostEqual(self.coeff.get_co_basis(
            0)[4][0], math.sqrt(3.0)/2.0)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            0)[4][1], 0.5)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            0)[4][2], 0.0)

        self.assertAlmostEqual(self.coeff.get_co_basis(
            1)[0][0], math.sqrt(3.0)-2.0)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            1)[0][1], 1.0)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            1)[0][2], 0.0)

        self.assertAlmostEqual(self.coeff.get_co_basis(
            2)[0][0], 0.0)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            2)[0][1], 0.0)
        self.assertAlmostEqual(self.coeff.get_co_basis(
            2)[0][2], 1.0)

    def test_contravariantBasis(self):
        self.assertAlmostEqual(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            0)[10]), 1.0)
        self.assertAlmostEqual(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            1)[10]), 0.0)
        self.assertAlmostEqual(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            2)[10]), 0.0)

    def test_metricTensor(self):
        ids = [0, 3, 6]

        for id in ids:
            diff = self.coeff.get_metric_tensor(0, 0)[id] - np.dot(
                self.coeff.get_co_basis(0)[id], self.coeff.get_co_basis(0)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_metric_tensor(1, 1)[id] - np.dot(
                self.coeff.get_co_basis(1)[id], self.coeff.get_co_basis(1)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_metric_tensor(2, 2)[id] - np.dot(
                self.coeff.get_co_basis(2)[id], self.coeff.get_co_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_metric_tensor(0, 1)[id] - np.dot(
                self.coeff.get_co_basis(0)[id], self.coeff.get_co_basis(1)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_metric_tensor(0, 2)[id] - np.dot(
                self.coeff.get_co_basis(0)[id], self.coeff.get_co_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_metric_tensor(1, 2)[id] - np.dot(
                self.coeff.get_co_basis(1)[id], self.coeff.get_co_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

    def test_invMetricTensor(self):
        ids = [0, 3, 6]

        for id in ids:
            diff = self.coeff.get_inv_metric_tensor(0, 0)[id] - np.dot(
                self.coeff.get_con_basis(0)[id], self.coeff.get_con_basis(0)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_inv_metric_tensor(1, 1)[id] - np.dot(
                self.coeff.get_con_basis(1)[id], self.coeff.get_con_basis(1)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_inv_metric_tensor(2, 2)[id] - np.dot(
                self.coeff.get_con_basis(2)[id], self.coeff.get_con_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_inv_metric_tensor(0, 1)[id] - np.dot(
                self.coeff.get_con_basis(0)[id], self.coeff.get_con_basis(1)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_inv_metric_tensor(0, 2)[id] - np.dot(
                self.coeff.get_con_basis(0)[id], self.coeff.get_con_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

            diff = self.coeff.get_inv_metric_tensor(1, 2)[id] - np.dot(
                self.coeff.get_con_basis(1)[id], self.coeff.get_con_basis(2)[id])
            self.assertTrue(abs(diff) < 10e-8)

    def test_christoffelSymbol(self):
        pass
        # print(self.coeff._christoffel_symbol[..., 1, 0, 1])
        # print("*************************")
        # print(self.coeff._christoffel_symbol[..., 1, 1, 0])
        # print("*************************")
        # print(self.coeff._christoffel_symbol[..., 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
