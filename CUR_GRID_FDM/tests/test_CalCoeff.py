import unittest
from CUR_GRID_FDM.DiscreteSchemes import CalCoeff
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from CUR_GRID_FDM.Geometry import DonutMesh
from CUR_GRID_FDM.Geometry import RectangularMesh

import numpy as np
import math


class CalCoeffTestCase(unittest.TestCase):
    def setUp(self):
        mesh = DonutMesh(2.0, 5.0, 4, 12, 2.0, 3)
        self.coeff = CalCoeff(mesh)
        # mesh.plot_nodeIJK(0, 0, 0)
        # mesh.plot_node(4)

    # def tearDown(self):

    def test_covariantBasis(self):
        self.assertTrue(abs(self.coeff.get_co_basis(
            0)[4][0] - (math.sqrt(3.0)/2.0)) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            0)[4][1] - (0.5)) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            0)[4][2] - 0.0) < 10e-8)

        self.assertTrue(abs(self.coeff.get_co_basis(
            1)[0][0] - (math.sqrt(3.0)-2.0)) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            1)[0][1] - 1.0) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            1)[0][2] - 0.0) < 10e-8)

        self.assertTrue(abs(self.coeff.get_co_basis(
            2)[0][0] - 0.0) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            2)[0][1] - 0.0) < 10e-8)
        self.assertTrue(abs(self.coeff.get_co_basis(
            2)[0][2] - 1.0) < 10e-8)

    def test_contravariantBasis(self):
        self.assertTrue(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            0)[10]) == 1.0)
        self.assertTrue(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            1)[10]) == 0.0)
        self.assertTrue(np.dot(self.coeff.get_co_basis(
            0)[10], self.coeff.get_con_basis(
            2)[10]) == 0.0)

    def test_metricTensor(self):
        id = 1

        print(self.coeff.get_metric_tensor(0, 0)[id])
        print(np.dot(self.coeff.get_con_basis(0)[
              id], self.coeff.get_con_basis(0)[id]))

        diff = self.coeff.get_metric_tensor(0, 0)[id] - np.dot(
            self.coeff.get_con_basis(0)[id], self.coeff.get_con_basis(0)[id])
        self.assertTrue(abs(diff) < 10e-8)

        print(self.coeff.get_metric_tensor(1, 1)[id])
        print(np.dot(self.coeff.get_con_basis(1)[
              id], self.coeff.get_con_basis(1)[id]))

        # diff = self.coeff.get_metric_tensor(1, 1)[id] - np.dot(
        #     self.coeff.get_con_basis(1)[id], self.coeff.get_con_basis(1)[id])
        # self.assertTrue(abs(diff) < 10e-8)

        diff = self.coeff.get_metric_tensor(2, 2)[id] - np.dot(
            self.coeff.get_con_basis(2)[id], self.coeff.get_con_basis(2)[id])
        self.assertTrue(abs(diff) < 10e-8)

        diff = self.coeff.get_metric_tensor(0, 1)[id] - np.dot(
            self.coeff.get_con_basis(0)[id], self.coeff.get_con_basis(1)[id])
        self.assertTrue(abs(diff) < 10e-8)

        diff = self.coeff.get_metric_tensor(0, 2)[id] - np.dot(
            self.coeff.get_con_basis(1)[id], self.coeff.get_con_basis(0)[id])
        self.assertTrue(abs(diff) < 10e-8)

        diff = self.coeff.get_metric_tensor(1, 2)[id] - np.dot(
            self.coeff.get_con_basis(1)[id], self.coeff.get_con_basis(2)[id])
        self.assertTrue(abs(diff) < 10e-8)


if __name__ == "__main__":
    unittest.main()
