from ..BaseMesh import BaseMesh
import numpy as np


class CurveRectangularMesh(BaseMesh):

    def __init__(self, lx: float, nx: int, ly: float, ny: int, lz: float, nz: int):
        self._lx = lx
        self._ly = ly
        self._lz = lz

        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._create_grid()

    def _create_grid(self):

        self._node_number = self._nx*self._ny*self._nz
        self._mesh_size = (self._nx, self._ny, self._nz)

        x = np.linspace(0, self._lx, self._nx)
        y = np.linspace(0, self._ly, self._ny)
        z = np.linspace(0, self._lz, self._nz)

        self._x, self._y, self._z = np.meshgrid(x, y, z, indexing='ij')

        amp = self._ly*0.2

        for i in range(len(self._y[:, 0, 0])):
            self._y[i, :, :] = self._y[i, :, :] + \
                amp*np.sin(self._x[i, 0, 0]*3.14)
