import unittest
from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D
from CUR_GRID_FDM.Geometry import RectangularMesh


class OperatorFDM3DTestCase(unittest.TestCase):
    # def setUp(self):
    # def tearDown(self):

    def test_OperatorFDM3D(self):
        mesh = RectangularMesh(1.0, 5, 1.0, 5, 1.0, 5)
        operator = OperatorFDM3D(mesh)


if __name__ == "__main__":
    unittest.main()
