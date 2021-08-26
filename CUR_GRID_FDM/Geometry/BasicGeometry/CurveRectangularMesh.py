from ..BaseMesh import BaseMesh
import numpy as np


class CurveRectangularMesh(BaseMesh):

    def __init__(self, lx: float, nx: int, amp_x: float, ly: float, ny: int, amp_y: float, lz: float, nz: int, amp_z: float):
        self._lx = lx
        self._ly = ly
        self._lz = lz

        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._amp_x = amp_x
        self._amp_y = amp_y
        self._amp_z = amp_z

        self._create_grid()

    def _create_grid(self):

        self._node_number = self._nx*self._ny*self._nz
        self._mesh_size = (self._nx, self._ny, self._nz)

        x = np.linspace(0, self._lx, self._nx)
        y = np.linspace(0, self._ly, self._ny)
        z = np.linspace(0, self._lz, self._nz)

        self._x, self._y, self._z = np.meshgrid(x, y, z, indexing='ij')

        for i in range(len(self._x[0, 0, :])):
            self._x[:, :, i] += self._amp_x*np.sin(self._z[0, 0, i]*3.14)

        for i in range(len(self._y[:, 0, 0])):
            self._y[i, :, :] += self._amp_y*np.sin(self._x[i, 0, 0]*3.14)

        for i in range(len(self._z[0, :, 0])):
            self._z[:, i, :] += self._amp_z*np.sin(self._y[0, i, 0]*3.14)
