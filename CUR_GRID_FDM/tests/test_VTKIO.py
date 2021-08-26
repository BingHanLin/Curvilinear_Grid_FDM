import unittest
import numpy as np
from CUR_GRID_FDM.IO import vtkStructuredGridWrapper
from CUR_GRID_FDM.Geometry import RectangularMesh


class CalCoeffTestCase(unittest.TestCase):
    # def setUp(self):
    # def tearDown(self):

    def test_vtkStructuredGridWrapper(self):
        mesh = RectangularMesh(1.0, 4, 1.0, 4, 1.0, 4)

        data = np.linspace(0.0, 100, mesh.node_number())

        writer = vtkStructuredGridWrapper(mesh)
        writer.addScalrDataArray("var", data)
        # writer.interact()
        # writer.write("test.vtk")


if __name__ == "__main__":
    unittest.main()
