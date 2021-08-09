from CUR_GRID_FDM.Geometry.BaseMesh import NODELOC
import unittest
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from CUR_GRID_FDM.Geometry import DonutMesh
from CUR_GRID_FDM.Geometry import RectangularMesh


class CurveRectangularMeshTestCase(unittest.TestCase):
    # def setUp(self):
    # def tearDown(self):

    def test_CurveRectangularMeshInfo(self):
        mesh = CurveRectangularMesh(2.0, 10, 1.0, 10, 1.0, 5)

        self.assertEqual(mesh.node_number(), 10*10*5)
        self.assertEqual(mesh.mesh_size(), (10, 10, 5))

    def test_DonutMeshInfo(self):
        mesh = DonutMesh(2.0, 5.0, 5, 10, 1.0, 6)

        self.assertEqual(mesh.node_number(), 5*10*6)
        self.assertEqual(mesh.mesh_size(), (5, 10, 6))

    def test_RectangularMeshInfo(self):
        mesh = RectangularMesh(1.0, 4, 1.0, 4, 1.0, 4)

        mesh.get_node_index_list(NODELOC.INTERIOR)

        self.assertEqual(mesh.node_number(), 4*4*4)
        self.assertEqual(mesh.mesh_size(), (4, 4, 4))


if __name__ == "__main__":
    unittest.main()
