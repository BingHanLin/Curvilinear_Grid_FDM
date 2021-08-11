import unittest
from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D
from CUR_GRID_FDM.Geometry import RectangularMesh
import numpy as np


class OperatorFDM3DTestCase(unittest.TestCase):
    # def setUp(self):
    # def tearDown(self):

    def test_Der_1(self):
        mesh = RectangularMesh(5.0, 6, 5.0, 11, 6.0, 4)
        op = OperatorFDM3D(mesh)

        self.assertTrue(np.all(op.der_1('i')*mesh.x_flatten() == 1.0))
        self.assertTrue(np.all(op.der_1('j')*mesh.y_flatten() == 0.5))
        self.assertTrue(np.all(op.der_1('k')*mesh.z_flatten() == 2.0))

    # def test_Der_2(self):
    #     mesh = RectangularMesh(5.0, 6, 5.0, 6, 5.0, 6)
    #     op = OperatorFDM3D(mesh)

    #     print(op.der_2('i')*mesh.x_flatten())
        # self.assertTrue(np.all(op.der_2('i')*mesh.x_flatten() == 1.0))
        # self.assertTrue(np.all(op.der_1('j')*mesh.y_flatten() == 0.5))
        # self.assertTrue(np.all(op.der_1('k')*mesh.z_flatten() == 2.0))


if __name__ == "__main__":
    unittest.main()
