import unittest
from CUR_GRID_FDM.DiscreteSchemes import CalCoeff
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from CUR_GRID_FDM.Geometry import DonutMesh
from CUR_GRID_FDM.Geometry import RectangularMesh


class CalCoeffTestCase(unittest.TestCase):
    def setUp(self):
        mesh = DonutMesh(5.0, 1.0, 5, 5, 1.0, 2)
        self.coeff = CalCoeff(mesh)
    # def tearDown(self):

    def test_1(self):
        self.coeff.get_co_basis(0)
        self.coeff.get_con_basis(0)


if __name__ == "__main__":
    unittest.main()
