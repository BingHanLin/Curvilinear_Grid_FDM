from ..BaseMesh import BaseMesh
import numpy as np


class RectangularMesh(BaseMesh):

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
        z = np.linspace(0, self._ly, self._nz)

        self._x, self._y, self._z = np.meshgrid(x, y, z, indexing='ij')
