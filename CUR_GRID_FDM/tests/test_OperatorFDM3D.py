import unittest
from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D
from CUR_GRID_FDM.Geometry import RectangularMesh
import numpy as np


class OperatorFDM3DTestCase(unittest.TestCase):
    def setUp(self):
        mesh = RectangularMesh(5.0, 6, 5.0, 11, 6.0, 4)
        self.op = OperatorFDM3D(mesh)
    # def tearDown(self):

    def test_Der_1(self):

        dense_i = self.op.der_1('i').todense()
        dense_j = self.op.der_1('j').todense()
        dense_k = self.op.der_1('k').todense()

        self.assertTrue(dense_i[0, 0] == -1.5)
        self.assertTrue(dense_i[0, 1] == 2.0)
        self.assertTrue(dense_i[0, 2] == -0.5)

        self.assertTrue(dense_i[1, 0] == -0.5)
        self.assertTrue(dense_i[1, 1] == 0.0)
        self.assertTrue(dense_i[1, 2] == 0.5)

        self.assertTrue(dense_j[0, 0] == -1.5)
        self.assertTrue(dense_j[0, 6] == 2.0)
        self.assertTrue(dense_j[0, 12] == -0.5)

        self.assertTrue(dense_k[0, 0] == -1.5)
        self.assertTrue(dense_k[0, 66] == 2.0)
        self.assertTrue(dense_k[0, 132] == -0.5)

        self.assertTrue(self.op.der_1('i').shape[0]
                        == self.op.der_1('i').shape[1] == 264)
        self.assertTrue(self.op.der_1('j').shape[0]
                        == self.op.der_1('j').shape[1] == 264)
        self.assertTrue(self.op.der_1('k').shape[0]
                        == self.op.der_1('k').shape[1] == 264)

    def test_Der_2(self):
        mesh = RectangularMesh(5.0, 6, 5.0, 11, 6.0, 4)
        op = OperatorFDM3D(mesh)

        dense_i = self.op.der_2('i').todense()
        dense_j = self.op.der_2('j').todense()
        dense_k = self.op.der_2('k').todense()

        self.assertTrue(dense_i[0, 0] == 2.0)
        self.assertTrue(dense_i[0, 1] == -5.0)
        self.assertTrue(dense_i[0, 2] == 4.0)
        self.assertTrue(dense_i[0, 3] == -1.0)

        self.assertTrue(dense_i[1, 0] == 1.0)
        self.assertTrue(dense_i[1, 1] == -2.0)
        self.assertTrue(dense_i[1, 2] == 1.0)

        self.assertTrue(dense_j[0, 0] == 2.0)
        self.assertTrue(dense_j[0, 6] == -5.0)
        self.assertTrue(dense_j[0, 12] == 4.0)
        self.assertTrue(dense_j[0, 18] == -1.0)

        self.assertTrue(dense_k[0, 0] == 2.0)
        self.assertTrue(dense_k[0, 66] == -5.0)
        self.assertTrue(dense_k[0, 132] == 4.0)
        self.assertTrue(dense_k[0, 198] == -1.0)

        self.assertTrue(self.op.der_2('i').shape[0]
                        == self.op.der_2('i').shape[1] == 264)
        self.assertTrue(self.op.der_2('j').shape[0]
                        == self.op.der_2('j').shape[1] == 264)
        self.assertTrue(self.op.der_2('k').shape[0]
                        == self.op.der_2('k').shape[1] == 264)


if __name__ == "__main__":
    unittest.main()
