import unittest
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from CUR_GRID_FDM.Geometry import DonutMesh
from CUR_GRID_FDM.Geometry import RectangularMesh


class CurveRectangularMeshTestCase(unittest.TestCase):
    # def setUp(self):
    # def tearDown(self):

    def test_CurveRectangularMeshInfo(self):
        geom = CurveRectangularMesh(2.0, 40, 1.0, 20)

        self.assertEqual(geom.node_number(), 40*20*1)
        self.assertEqual(geom.mesh_size(), (40, 20, 1))

    def test_DonutMeshInfo(self):
        geom = DonutMesh(5.0, 1.0, 5, 10)

        self.assertEqual(geom.node_number(), 5*10*1)
        self.assertEqual(geom.mesh_size(), (5, 10, 1))

    def test_RectangularMeshInfo(self):
        geom = RectangularMesh(1.0, 5, 1.0, 10)

        self.assertEqual(geom.node_number(), 5*10*1)
        self.assertEqual(geom.mesh_size(), (5, 10, 1))


if __name__ == "__main__":
    unittest.main()
